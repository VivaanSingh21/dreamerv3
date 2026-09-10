#!/usr/bin/env python
"""py3.7 CARLA bridge worker. Runs in the `carla96` conda env (or the vibes-carla96
image); the DreamerV3 process (py3.9+/JAX) talks to it over a TCP socket because the
CARLA 0.9.6 client egg caps its process at Python 3.7 and cannot share an interpreter
with modern JAX.

Responsibilities:
  * launch + supervise the CarlaUE4 server (vendored carla_supervisor.CarlaServer)
  * build the DBC PatchedCarlaEnv (5 cameras -> 84x420, 'sar' reward, Town04, 10 NPCs)
  * resize the stitched frame to --out-size and serve reset()/step() over the socket
  * survive a hung/dead server: restart it, rebuild the env, and hand DreamerV3 a
    terminal observation so the episode ends cleanly

Nothing DreamerV3-specific lives here. The is_terminal mapping (real terminal vs
time-limit truncation) is the one piece of protocol knowledge and it is spelled out
in _pack_obs.

Usage (normally invoked by embodied/envs/carla.py, not by hand):
  python worker.py --listen-port 2610 --carla-rpc-port 3000 --gpu 0 \
      --carla-root /home/carla/CARLA_0.9.6 \
      --egg /home/carla/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.7-linux-x86_64.egg
"""
from __future__ import print_function

import argparse
import os
import socket
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, '_vendor')


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--listen-host', default='127.0.0.1')
  p.add_argument('--listen-port', type=int, default=2610)
  p.add_argument('--carla-rpc-port', type=int, default=3000)
  p.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES for server + client')
  p.add_argument('--carla-root', default=os.environ.get(
      'CARLA_ROOT', os.path.expanduser('~/CARLA_0.9.6')))
  p.add_argument('--egg', default=os.environ.get('CARLA_EGG', ''))
  p.add_argument('--sar-root', default=os.path.join(_VENDOR, 'sar_env'))
  p.add_argument('--egl-vendor-json',
                 default='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  # DBC / VIBES protocol defaults.
  p.add_argument('--rl-image-size', type=int, default=84)
  p.add_argument('--num-cameras', type=int, default=5)
  p.add_argument('--fov', type=int, default=60)
  p.add_argument('--frame-skip', type=int, default=4)
  p.add_argument('--max-episode-steps', type=int, default=1000,
                 help='in simulator frames; 1000 = 250 agent steps at frame-skip 4')
  p.add_argument('--changing-weather-speed', type=float, default=0.1)
  p.add_argument('--out-size', default='64,256',
                 help='"H,W" the stitched panorama is resized to before it leaves here')
  p.add_argument('--boot-timeout', type=float, default=120.0)
  p.add_argument('--restart-settle-s', type=float, default=20.0)
  p.add_argument('--max-reset-attempts', type=int, default=4)
  return p.parse_args()


def setup_environment(args):
  args.carla_root = os.path.expanduser(args.carla_root)
  args.sar_root = os.path.expanduser(args.sar_root)
  # Headless EGL: without the NVIDIA vendor ICD, libglvnd routes EGL to Mesa -> SIGSEGV.
  os.environ['DISPLAY'] = ''
  if os.path.exists(args.egl_vendor_json):
    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = args.egl_vendor_json
  else:
    print('[worker] WARNING: EGL vendor json not found at %s; CARLA may segfault'
          % args.egl_vendor_json)
  # pygame is imported at carla_env module load even headless.
  os.environ.setdefault('SDL_VIDEODRIVER', 'offscreen')
  os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
  os.environ['CARLA_ROOT'] = args.carla_root
  os.environ['SAR_ROOT'] = args.sar_root
  os.environ['CARLA_REWARD'] = 'sar'  # protocol reward; explicit so a stray env var can't flip it
  os.environ.setdefault('CARLA_RESTART_SETTLE_S', str(args.restart_settle_s))

  egg = args.egg
  if not egg:
    import glob
    cands = glob.glob(os.path.join(
        args.carla_root, 'PythonAPI', 'carla', 'dist',
        'carla-*%d.%d-linux-x86_64.egg' % (sys.version_info.major, sys.version_info.minor)))
    egg = cands[0] if cands else ''
  if egg and egg not in sys.path:
    sys.path.append(egg)
  print('[worker] python %s, egg=%s' % (sys.version.split()[0], egg or '<none>'))

  for path in (_VENDOR, args.sar_root):
    if path not in sys.path:
      sys.path.insert(0, path)


class BridgeEnv(object):
  """Owns the CarlaServer + PatchedCarlaEnv and the sick-server recovery.

  Mirrors CarlaVibesEnv.reset()'s retry pattern (a timed-out reset IS the liveness
  probe for a hung server) but drops the fixed-length padding: DreamerV3 handles
  variable-length episodes natively.
  """

  def __init__(self, args, overrides=None):
    from carla_supervisor import CarlaServer
    from sar_env_patched import PatchedCarlaEnv
    import queue
    self._PatchedCarlaEnv = PatchedCarlaEnv
    self._sick = (RuntimeError, queue.Empty)
    self.args = args
    o = dict(overrides or {})
    # The DreamerV3 adapter is authoritative for env-shape params (it sends them on
    # connect); CLI args are the fallback for standalone use / the smoke client.
    self.out_hw = tuple(int(x) for x in str(
        o.get('out_size', args.out_size)).replace(' ', '').split(','))
    self._env_kwargs = dict(
        render_display=0, display_text=0,
        changing_weather_speed=o.get('changing_weather_speed', args.changing_weather_speed),
        rl_image_size=o.get('rl_image_size', args.rl_image_size),
        max_episode_steps=o.get('max_episode_steps', args.max_episode_steps),
        frame_skip=o.get('frame_skip', args.frame_skip), is_other_cars=True,
        num_cameras=o.get('num_cameras', args.num_cameras),
        fov=o.get('fov', args.fov), port=args.carla_rpc_port)
    print('[worker] env kwargs: %s  out_size=%s' % (self._env_kwargs, self.out_hw))
    self.server = CarlaServer(
        port=args.carla_rpc_port, root=args.carla_root,
        boot_timeout=args.boot_timeout, exclusive=False)
    self.server.start()
    self.env = None
    self._build_env()

  def _build_env(self):
    if self.server.port != self._env_kwargs['port']:
      self._env_kwargs['port'] = self.server.port
    self.env = self._PatchedCarlaEnv(**self._env_kwargs)

  def _restart_and_rebuild(self):
    self.server.restart()
    time.sleep(self.args.restart_settle_s)
    self._build_env()

  def reset(self):
    last = None
    for attempt in range(self.args.max_reset_attempts):
      try:
        # Only run the fragile server-health precheck on a retry: on the happy path
        # a transient "port didn't accept within 1s" would trigger a needless
        # 2-minute server restart.
        if attempt and self.server.ensure():
          time.sleep(self.args.restart_settle_s)
          self._build_env()
        t0 = time.time()
        obs = self.env.reset()  # (C,H,W) uint8
        print('[worker] env.reset() ok in %.1fs (attempt %d)'
              % (time.time() - t0, attempt + 1))
        return self._pack_obs(obs, 0.0, False, {})
      except self._sick as e:
        last = e
        print('[worker] reset attempt %d/%d failed: %s(%s) -- restarting server'
              % (attempt + 1, self.args.max_reset_attempts, type(e).__name__, e))
        try:
          self._restart_and_rebuild()
        except Exception as re_err:  # noqa: BLE001
          last = re_err
          print('[worker] recovery failed: %s(%s)' % (type(re_err).__name__, re_err))
    raise RuntimeError('reset failed %d times; last error: %s'
                       % (self.args.max_reset_attempts, last))

  def step(self, action):
    try:
      obs, reward, done, info = self.env.step(action)
      return self._pack_obs(obs, float(reward), bool(done), info or {})
    except self._sick as e:
      # A crash mid-episode: end the episode as a real terminal so DreamerV3's cont
      # head learns P(continue)=0 here, and rebuild for the next reset().
      print('[worker] step failed: %s(%s) -- ending episode, restarting server'
            % (type(e).__name__, e))
      try:
        self._restart_and_rebuild()
      except Exception as re_err:  # noqa: BLE001
        print('[worker] post-crash rebuild failed: %s(%s)'
              % (type(re_err).__name__, re_err))
      return self._pack_obs(self._last_image_zeros(), 0.0, True,
                            {'reason_episode_ended': 'server_crash'})

  # ---- obs packing -----------------------------------------------------------
  # A real terminal (cont-head target P(continue)=0, no value bootstrap) vs the
  # time-limit truncation 'success' (is_last but NOT is_terminal -> bootstrap).
  _REAL_TERMINALS = ('off_road', 'no_waypoints', 'carla_bug', 'stuck', 'server_crash')

  def _pack_obs(self, chw, reward, done, info):
    import numpy as np
    reason = info.get('reason_episode_ended', '')
    terminal = bool(done) and reason in self._REAL_TERMINALS
    hwc = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))  # (H,W,3) uint8
    hwc = self._resize(hwc)
    keep = ('reason_episode_ended', 'crash_intensity', 'vel_s', 'dist_from_center',
            'steer', 'brake', 'distance')
    small = {k: info[k] for k in keep if k in info}
    return ('obs', dict(
        image=hwc, reward=float(reward), done=bool(done),
        terminal=bool(terminal), info=small))

  def _resize(self, hwc):
    import numpy as np
    h, w = self.out_hw
    if hwc.shape[:2] == (h, w):
      return hwc
    from PIL import Image
    return np.asarray(Image.fromarray(hwc).resize((w, h), Image.BILINEAR), dtype=np.uint8)

  def _last_image_zeros(self):
    import numpy as np
    return np.zeros((3, self.args.rl_image_size,
                     self.args.num_cameras * self.args.rl_image_size), np.uint8)

  def close(self):
    try:
      if self.env is not None:
        self.env.finish()
    except Exception:  # noqa: BLE001
      pass
    try:
      self.server.kill()
    except Exception:  # noqa: BLE001
      pass


def main():
  args = parse_args()
  setup_environment(args)
  if _HERE not in sys.path:  # protocol.py lives next to this file
    sys.path.insert(0, _HERE)
  from protocol import send_msg, recv_msg

  # Bind + accept BEFORE booting CARLA, so DreamerV3 can connect and wait, and a
  # boot failure is reported over the socket instead of as a connect timeout.
  srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  srv.bind((args.listen_host, args.listen_port))
  srv.listen(1)
  print('[worker] listening on %s:%d (CARLA not booted yet)'
        % (args.listen_host, args.listen_port))
  conn, addr = srv.accept()
  conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  print('[worker] client connected from %s' % (addr,))

  # First message is the env config from the adapter (or the smoke client sends
  # ('config', {}) / nothing-meaningful and CLI args stand).
  overrides = {}
  first = recv_msg(conn)
  if first[0] == 'config':
    overrides = first[1] or {}
    print('[worker] config from client: %s' % overrides)
  else:
    print('[worker] first msg was %r, not config; using CLI args' % (first[0],))

  bridge = None
  try:
    bridge = BridgeEnv(args, overrides)
  except Exception:  # noqa: BLE001
    tb = traceback.format_exc()
    print(tb)
    try:
      send_msg(conn, ('error', tb))
    except OSError:
      pass
    conn.close()
    srv.close()
    raise

  send_msg(conn, ('ready',))
  try:
    while True:
      msg = recv_msg(conn)
      cmd = msg[0]
      if cmd == 'close':
        break
      elif cmd == 'reset':
        send_msg(conn, bridge.reset())
      elif cmd == 'step':
        send_msg(conn, bridge.step(msg[1]))
      else:
        send_msg(conn, ('error', 'unknown command %r' % (cmd,)))
  except (ConnectionError, OSError) as e:
    print('[worker] connection lost: %s' % e)
  except Exception:  # noqa: BLE001
    tb = traceback.format_exc()
    print(tb)
    try:
      send_msg(conn, ('error', tb))
    except OSError:
      pass
  finally:
    try:
      conn.close()
    except OSError:
      pass
    srv.close()
    bridge.close()


if __name__ == '__main__':
  main()
