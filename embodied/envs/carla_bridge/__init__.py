"""CARLA 0.9.6 bridge: a py3.7 worker process (worker.py) that runs the DBC/VIBES
CarlaEnv and talks to DreamerV3 over a socket. Only `protocol` is safe to import
from the DreamerV3 (py3.9+) side; worker.py and _vendor/ require Python 3.7 + the
CARLA client egg.
"""
from . import protocol  # noqa: F401
