"""Length-prefixed pickle framing shared by the DreamerV3 side and the py3.7 worker.

Must import and run under BOTH Python 3.7 (the CARLA client egg's ceiling) and the
Python 3.9+ DreamerV3 process, so: stdlib only, pickle protocol 4 (the highest 3.7
can read), no f-strings-in-annotations, no walrus.

Wire format per message: 4-byte big-endian unsigned length, then that many bytes of
``pickle.dumps(obj, protocol=4)``. Messages are plain tuples:

  DreamerV3 -> worker : ('reset',)  |  ('step', <np.ndarray float32 (2,)>)  |  ('close',)
  worker -> DreamerV3 : ('ready',)  |  ('obs', <dict>)  |  ('error', <str traceback>)

The ``obs`` dict has: image (uint8 HxWx3), reward (float), done (bool), terminal (bool),
info (small dict of scalars). ``reward`` on a reset frame is 0.0 and ``done`` is False.
"""
import pickle
import struct

PROTOCOL = 4
_HEADER = struct.Struct('>I')


def send_msg(sock, obj):
  body = pickle.dumps(obj, protocol=PROTOCOL)
  sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock, n):
  chunks = []
  got = 0
  while got < n:
    chunk = sock.recv(min(n - got, 1 << 20))
    if not chunk:
      raise ConnectionError('bridge socket closed mid-message')
    chunks.append(chunk)
    got += len(chunk)
  return b''.join(chunks)


def recv_msg(sock):
  header = _recv_exactly(sock, _HEADER.size)
  (length,) = _HEADER.unpack(header)
  return pickle.loads(_recv_exactly(sock, length))
