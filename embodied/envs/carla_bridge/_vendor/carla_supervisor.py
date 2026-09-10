#!/usr/bin/env python
"""
CARLA server supervisor.

CARLA 0.9.6 crashes during long runs -- we observed one crash between short test
sessions, and CARLA's own first-ever GitHub issue is "Random crashes when running
multiple server instances in parallel". Roach pins 0.9.10.1 "because 0.9.11 crashes
more often". A training run of 400k simulator frames WILL hit this.

Usage as a context manager (the normal case):

    from carla_supervisor import CarlaServer
    with CarlaServer(port=2000) as srv:
        env = CarlaEnv(..., port=srv.port)
        ...
        if srv.is_dead():          # cheap check between episodes
            srv.restart()
            env = CarlaEnv(..., port=srv.port)   # env must be rebuilt after a restart

Or standalone:  python carla_supervisor.py --port 2000 --watch

NOTE: this manages the SERVER only. A restart invalidates every actor handle, so the
caller must rebuild the env. Pair it with agent-side checkpointing so a crash costs
minutes, not hours.
"""
from __future__ import print_function
import argparse
import os
import signal
import socket
import subprocess
import sys
import time

CARLA_ROOT = os.environ.get("CARLA_ROOT", os.path.expanduser("~/CARLA_0.9.6"))
NVIDIA_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
PROC_NAME = "CarlaUE4-Linux-"      # `pgrep -x` truncates comm at 15 chars


def _port_open(port, host="127.0.0.1", timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _pgrep():
    try:
        out = subprocess.check_output(["pgrep", "-x", PROC_NAME],
                                      stderr=subprocess.DEVNULL)
        return [int(x) for x in out.split()]
    except Exception:
        return []


class CarlaServer(object):
    def __init__(self, port=2000, root=CARLA_ROOT, quality="Low",
                 gpu_egl=NVIDIA_EGL, boot_timeout=120.0, log=None,
                 max_restarts=100, exclusive=True,
                 windowed=None, display=None, res=None):
        # windowed=True opens a real CARLA window on `display` so a run can be watched.
        # Env-var defaults so an existing run can be made visible without code changes:
        #   CARLA_WINDOWED=1  CARLA_DISPLAY=:1  CARLA_RES=960x540
        # NOTE: windowed rendering goes through GLX on a real X display, so the headless
        # EGL vendor override must NOT be set -- it is what makes the OFF-screen path work,
        # and forcing it here breaks the windowed one.
        if windowed is None:
            windowed = os.environ.get("CARLA_WINDOWED", "") not in ("", "0", "false")
        self.windowed = bool(windowed)
        self.display = display or os.environ.get("CARLA_DISPLAY", ":1")
        self.res = res or os.environ.get("CARLA_RES", "960x540")
        # exclusive=True: kill() also pkills EVERY CarlaUE4 on the box. Right for the
        # single-server case (clears strays from a crashed run), FATAL when two servers
        # coexist -- the second one's start() would kill the first. Multi-server callers
        # (carla_vibes_env: one server per training process) must pass exclusive=False,
        # which restricts kill() to this instance's own process group.
        self.exclusive = bool(exclusive)
        self.port = int(port)
        self.base_port = int(port)   # rotation wraps back near here if it ever runs out
        self.root = root
        self.quality = quality
        self.gpu_egl = gpu_egl
        self.boot_timeout = boot_timeout
        self.log_path = log or os.path.expanduser("~/carla_server_%d.log" % self.port)
        self.max_restarts = max_restarts
        self.restarts = 0
        self._proc = None

    # ---- lifecycle -------------------------------------------------------
    def _env(self):
        e = dict(os.environ)
        if self.windowed:
            # Real X display -> GLX. Must NOT set __EGL_VENDOR_LIBRARY_FILENAMES here.
            e["DISPLAY"] = self.display
            e.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
            return e
        e["DISPLAY"] = ""                       # headless
        if self.gpu_egl:
            # Without this, libglvnd dispatches EGL to Mesa -> "failed to create dri2
            # screen" -> SIGSEGV on this machine. Required, not optional.
            e["__EGL_VENDOR_LIBRARY_FILENAMES"] = self.gpu_egl
        return e

    def start(self):
        self.kill()
        cmd = ["./CarlaUE4.sh", "-opengl",
               "-quality-level=%s" % self.quality,
               "-carla-rpc-port=%d" % self.port]
        if self.windowed:
            w, _, h = self.res.partition("x")
            cmd += ["-windowed", "-ResX=%s" % w, "-ResY=%s" % h]
        logf = open(self.log_path, "ab", 0)
        self._proc = subprocess.Popen(cmd, cwd=self.root, env=self._env(),
                                      stdout=logf, stderr=subprocess.STDOUT,
                                      preexec_fn=os.setsid)
        t0 = time.time()
        while time.time() - t0 < self.boot_timeout:
            time.sleep(1.0)
            # Check OUR process, not `pgrep` -- with a second server alive, a global
            # pgrep stays non-empty even when this one has died.
            if self._proc.poll() is not None or (self.exclusive and not _pgrep()):
                raise RuntimeError("CARLA died during boot; see %s" % self.log_path)
            if _port_open(self.port):
                print("[supervisor] server up on :%d in %.1fs (restarts so far: %d)"
                      % (self.port, time.time() - t0, self.restarts))
                return self
        self.kill()
        raise RuntimeError("CARLA did not open :%d within %.0fs; see %s"
                           % (self.port, self.boot_timeout, self.log_path))

    def kill(self):
        had_proc = self._proc is not None
        if had_proc:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except Exception:
                pass
            self._proc = None
        # Surgical port-scoped kill -- safe even with 4 concurrent servers. CARLA 0.9.10's
        # CarlaUE4-Linux-Shipping detaches from the CarlaUE4.sh process group, so killpg
        # above misses it and restarts pile up pegged-at-100%-CPU zombies. The rpc-port
        # flag is on both the .sh and the binary's command line, so -f "...port=<N>"
        # catches exactly this server without touching the others.
        subprocess.call(["pkill", "-9", "-f", "carla-rpc-port=%d" % self.port],
                        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        if self.exclusive:
            subprocess.call(["pkill", "-9", "-f", "CarlaUE4-Linux"],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.call(["pkill", "-9", "-f", "CarlaUE4.sh"],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(2.0)

        # WAIT FOR THE PORT TO ACTUALLY FREE. A fixed sleep is not enough: a dying -- or
        # SIGSTOPped -- process keeps its listening socket bound, and the kernel still
        # completes the TCP handshake from the backlog. start()'s readiness check is
        # _port_open(), so without this it sees the CORPSE's socket and reports
        # "server up in 3.0s" for a server that never booted. The client then talks to
        # nothing and every sensor read raises queue.Empty. This was the real reason
        # restart-on-hang failed, not the settle delay.
        t0 = time.time()
        while _port_open(self.port, timeout=0.5) and time.time() - t0 < 30.0:
            time.sleep(0.5)
        if _port_open(self.port, timeout=0.5):
            print("[supervisor] WARNING: :%d still bound %.0fs after kill; a new server "
                  "will likely fail to bind" % (self.port, time.time() - t0))

    # ---- health ----------------------------------------------------------
    def is_dead(self):
        """True if the process is gone or the RPC port stopped accepting.

        ⚠️ THIS CANNOT DETECT A HUNG SERVER. A CARLA that has stopped answering RPC still
        holds its listening socket open, so both checks below pass and this returns False.
        That is exactly what killed the 2026-08-17 post-fix smoke run: `ensure()` reported
        healthy, the caller proceeded, and the reset died with a 60 s RPC timeout.

        Detecting a hang properly needs a real RPC round-trip (e.g. `client.get_server_
        version()` on a short timeout), which would make this module import `carla` and
        depend on the egg. Instead the recovery lives in the caller:
        `CarlaVibesEnv.reset()` treats a timed-out reset as the liveness probe and restarts
        us. Anything else using this class directly inherits the blind spot.
        """
        if self._proc is not None and self._proc.poll() is not None:
            return True
        if self._proc is None and not _pgrep():
            return True
        return not _port_open(self.port)

    # Port rotation on restart (2026-08-20). Root cause of the "fresh server, no frames"
    # failure that killed runs 11, 11b, 12, 14 and 15: the replacement server was booted on
    # the SAME port pair, and the client process still carried the dead env's sensor
    # streams / sync-mode state bound to that pair. Measured while run 15's :2004
    # replacement could not deliver a frame: a brand-new server on an unused port (:2008)
    # booted in 4 s and rendered at 92 FPS from the same process tree. So the host was
    # fine and the poison was client-side, keyed to the port. Each restart therefore moves
    # to a new pair. Stride 8 keeps the four pool servers (2000/2002/2004/2006, residues
    # 0/2/4/6 mod 8) on disjoint progressions, and port+1 (the stream port) stays odd.
    PORT_STRIDE = int(os.environ.get("CARLA_PORT_ROTATE_STRIDE", 8))
    PORT_ROTATE = os.environ.get("CARLA_PORT_ROTATE", "1") != "0"

    def _next_port(self):
        p = self.port
        for _ in range(64):
            p += self.PORT_STRIDE
            if p > 65000:
                p = self.base_port + self.PORT_STRIDE
            if not _port_open(p, timeout=0.2) and not _port_open(p + 1, timeout=0.2):
                return p
        raise RuntimeError("no free port found rotating from %d" % self.port)

    def restart(self):
        self.restarts += 1
        if self.restarts > self.max_restarts:
            raise RuntimeError("exceeded max_restarts=%d" % self.max_restarts)
        old = self.port
        self.kill()
        if self.PORT_ROTATE:
            self.port = self._next_port()
            self.log_path = os.path.expanduser("~/carla_server_%d.log" % self.port)
        print("[supervisor] server unhealthy -- restart #%d (port %d -> %d)"
              % (self.restarts, old, self.port))
        self.start()
        return self

    def ensure(self):
        """Restart if unhealthy. Returns True if a restart happened."""
        if self.is_dead():
            self.restart()
            return True
        return False

    # ---- context manager --------------------------------------------------
    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.kill()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--quality", default="Low")
    ap.add_argument("--watch", action="store_true",
                    help="stay up, restarting the server whenever it dies")
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()

    srv = CarlaServer(port=a.port, quality=a.quality)
    srv.start()
    if not a.watch:
        print("[supervisor] started; not watching (use --watch to auto-restart)")
        return
    print("[supervisor] watching every %.0fs; Ctrl-C to stop" % a.interval)
    try:
        while True:
            time.sleep(a.interval)
            if srv.ensure():
                print("[supervisor] recovered at %s" % time.strftime("%H:%M:%S"))
    except KeyboardInterrupt:
        print("\n[supervisor] shutting down")
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
