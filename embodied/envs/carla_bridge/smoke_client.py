#!/usr/bin/env python
"""Minimal bridge check -- no DreamerV3, no JAX. Connects to a running worker.py,
does one reset + N random steps, prints obs shapes / reward / done reasons.

  python smoke_client.py --addr 127.0.0.1:2610 --steps 30

Run this from the DreamerV3 container (or anywhere with Python 3 + numpy) once the
worker prints "client connected -- booting CARLA" ... actually it waits for a client,
so just run it: it will connect, then block ~30-60s while CARLA boots.
"""
import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from protocol import send_msg, recv_msg  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--addr', default='127.0.0.1:2610')
  ap.add_argument('--steps', type=int, default=30)
  ap.add_argument('--episodes', type=int, default=2)
  a = ap.parse_args()
  host, _, port = a.addr.partition(':')

  s = socket.create_connection((host, int(port)), timeout=30)
  s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  send_msg(s, ('config', {}))  # keep the worker's CLI args
  print('connected; waiting for CARLA to boot ...')
  t0 = time.time()
  msg = recv_msg(s)
  assert msg[0] == 'ready', msg
  print('ready in %.1fs' % (time.time() - t0))

  for ep in range(a.episodes):
    send_msg(s, ('reset',))
    _, o = recv_msg(s)
    print('ep %d reset: image %s dtype=%s' % (ep, o['image'].shape, o['image'].dtype))
    ret = 0.0
    for t in range(a.steps):
      act = np.array([np.random.uniform(-0.3, 0.3), 0.6], np.float32)
      send_msg(s, ('step', act))
      tag, o = recv_msg(s)
      assert tag == 'obs', (tag, o)
      ret += o['reward']
      if o['done']:
        print('  done at t=%d reason=%s terminal=%s return=%.2f'
              % (t, o['info'].get('reason_episode_ended'), o['terminal'], ret))
        break
    else:
      print('  survived %d steps return=%.2f last_info=%s' % (a.steps, ret, o['info']))
  send_msg(s, ('close',))
  s.close()
  print('OK')


if __name__ == '__main__':
  main()
