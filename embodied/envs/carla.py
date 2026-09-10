import functools
import os
import socket
import subprocess
import sys
import time

import elements
import embodied
import numpy as np

from .carla_bridge import protocol


class Carla(embodied.Env):
  """DreamerV3 adapter for the DBC / VIBES CARLA 0.9.6 highway-driving task.

  DreamerV3 needs Python 3.9+ (JAX); the CARLA 0.9.6 client egg is Python 3.7 only,
  so the actual env runs in a separate process (`carla_bridge/worker.py`, in the
  `carla96` conda env / vibes-carla96 image) and this class proxies reset()/step()
  to it over a TCP socket. See CARLA_RUNBOOK.md.

  Task string: `carla_town04` -> task='town04' (the only task; DBC is single-task
  infinite highway driving on Town04, 5 cameras stitched to one wide image, 'sar'
  reward, 10 NPCs).

  obs:  image (uint8, `size` HxWx3), reward, is_first/is_last/is_terminal,
        log/vel_s, log/crash_intensity
  act:  action (float32 (2,) in [-1,1]) = [steer, throttle_brake], plus reset

  Two ways to reach the worker:
    * connect-only (default): the worker is already running (its own container);
      point `bridge_addr` at it.
    * spawn: set `spawn_worker=True` and `worker_python` to a py3.7 interpreter on
      this host; this class launches worker.py itself. Handy for a single-host
      conda test, not for the two-container canebrake setup.
  """

  def __init__(
      self, task='town04', size=(64, 256), frame_skip=4, max_episode_steps=1000,
      num_cameras=5, rl_image_size=84, fov=60, changing_weather_speed=0.1,
      bridge_addr='127.0.0.1:2610', connect_timeout=600.0,
      spawn_worker=False, worker_python='python', gpu='0',
      carla_root='', egg=''):
    assert task == 'town04', task
    self._size = tuple(size)
    self._addr = bridge_addr
    self._done = True
    self._proc = None
    self._sock = None
    host, _, port = bridge_addr.partition(':')
    self._host, self._port = host, int(port)
    self._connect_timeout = connect_timeout
    self._spawn = spawn_worker
    self._config_msg = dict(
        out_size='%d,%d' % self._size,
        frame_skip=frame_skip, max_episode_steps=max_episode_steps,
        num_cameras=num_cameras, rl_image_size=rl_image_size, fov=fov,
        changing_weather_speed=changing_weather_speed)
    self._spawn_kw = dict(
        py=worker_python, gpu=gpu, carla_root=carla_root, egg=egg,
        frame_skip=frame_skip, max_steps=max_episode_steps,
        num_cameras=num_cameras, rl_image_size=rl_image_size, fov=fov,
        weather=changing_weather_speed)
    # Connect lazily on the first step(): make_agent() builds a throwaway env just
    # to read obs_space/act_space (both static here) and closes it -- we must not
    # boot CARLA for that, and the worker is single-session.

  def _ensure_connected(self):
    if self._sock is not None:
      return
    if self._spawn:
      self._spawn_worker(**self._spawn_kw)
    self._sock = self._connect(self._connect_timeout)
    protocol.send_msg(self._sock, ('config', self._config_msg))
    msg = protocol.recv_msg(self._sock)
    if msg[0] == 'error':
      raise RuntimeError('carla worker failed to start:\n' + msg[1])
    assert msg[0] == 'ready', msg
    print('[carla] bridge ready at %s' % self._addr)

  # ---- worker lifecycle -----------------------------------------------------
  def _spawn_worker(self, py, gpu, carla_root, egg, frame_skip, max_steps,
                    num_cameras, rl_image_size, fov, weather):
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        py, os.path.join(here, 'carla_bridge', 'worker.py'),
        '--listen-host', self._host, '--listen-port', str(self._port),
        '--carla-rpc-port', str(self._port - 610 + 400),  # e.g. 2610 -> 2400
        '--gpu', str(gpu),
        '--frame-skip', str(frame_skip), '--max-episode-steps', str(max_steps),
        '--num-cameras', str(num_cameras), '--rl-image-size', str(rl_image_size),
        '--fov', str(fov), '--changing-weather-speed', str(weather),
        '--out-size', '%d,%d' % self._size]
    if carla_root:
      cmd += ['--carla-root', carla_root]
    if egg:
      cmd += ['--egg', egg]
    print('[carla] spawning worker: %s' % ' '.join(cmd))
    self._proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

  def _connect(self, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
      if self._proc is not None and self._proc.poll() is not None:
        raise RuntimeError('carla worker exited with code %d before accepting'
                           % self._proc.returncode)
      try:
        s = socket.create_connection((self._host, self._port), timeout=10.0)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Generous per-RPC timeout: a reset during CARLA's respawn-retry loop, or a
        # mid-run server restart (boot ~120s + settle), can legitimately take minutes.
        s.settimeout(600.0)
        return s
      except OSError as e:  # noqa: PERF203
        last = e
        time.sleep(2.0)
    raise RuntimeError('could not connect to carla bridge at %s within %.0fs: %s'
                       % (self._addr, timeout, last))

  # ---- spaces -------------------------------------------------------------
  @functools.cached_property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/vel_s': elements.Space(np.float32),
        'log/crash_intensity': elements.Space(np.float32),
    }

  @functools.cached_property
  def act_space(self):
    return {
        'action': elements.Space(np.float32, (2,), -1.0, 1.0),
        'reset': elements.Space(bool),
    }

  # ---- stepping ---------------------------------------------------------
  def step(self, action):
    self._ensure_connected()
    if action['reset'] or self._done:
      obs = self._rpc(('reset',))
      self._done = False
      return self._obs(obs, is_first=True)
    act = np.asarray(action['action'], np.float32).reshape(2)
    obs = self._rpc(('step', act))
    self._done = obs['done']
    return self._obs(obs, is_first=False)

  def _rpc(self, msg):
    protocol.send_msg(self._sock, msg)
    reply = protocol.recv_msg(self._sock)
    if reply[0] == 'error':
      raise RuntimeError('carla bridge error:\n' + reply[1])
    assert reply[0] == 'obs', reply
    return reply[1]

  def _obs(self, o, is_first):
    img = np.asarray(o['image'], np.uint8)
    assert img.shape == self._size + (3,), (img.shape, self._size)
    info = o.get('info', {})
    return dict(
        image=img,
        reward=np.float32(0.0 if is_first else o['reward']),
        is_first=bool(is_first),
        is_last=bool(o['done']),
        is_terminal=bool(o['terminal']),
        **{
            'log/vel_s': np.float32(info.get('vel_s', 0.0)),
            'log/crash_intensity': np.float32(info.get('crash_intensity', 0.0)),
        },
    )

  def close(self):
    if self._sock is not None:
      try:
        protocol.send_msg(self._sock, ('close',))
      except OSError:
        pass
      try:
        self._sock.close()
      except OSError:
        pass
      self._sock = None
    if self._proc is not None:
      try:
        self._proc.terminate()
        self._proc.wait(timeout=30)
      except Exception:  # noqa: BLE001
        self._proc.kill()
