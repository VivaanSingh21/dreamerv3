#!/usr/bin/env python
"""
CarlaEnv with the spawn glitch fixed. Subclass, not a fork -- ~/SAR stays pristine.

THE PROBLEM (measured, see sar_spawn_probe.py)
-----------------------------------------------
CarlaEnv.reset() ticks twice and immediately calls step(action=None). The ego spawns
at z=0.1 over a road at z=0.0, so it is still falling; ~20% of the time the physics
engine resolves the interpenetration by ejecting it up to 3 m into the air at up to
28 m/s. _simulator_step() then trips its own guard and sets

    info['reason_episode_ended'] = 'carla_bug';  done = True

but reset() returns obs ONLY -- it discards that done. The caller receives a healthy
looking observation and gets done=True on its very first step(). Result: a length-1
episode with reward ~ -0.3 that is nonetheless recorded in the eval statistics.

Measured on our server, 20 resets each, Town04, is_other_cars=True:

    stock SAR reset()          4/20 = 20%  glitched
    + 40 settle ticks          1/20 =  5%  glitched
    + settle + retry           0/20 =  0%  glitched   <-- this class

THE FIX
-------
Two failure modes, two remedies:

  (a) interpenetration ejection -- the car is airborne but falling. Ticking lets it
      land. 40 ticks suffices and is FREE: carla_sim_test.py measured that the
      vehicle sits in gear 0 for ~35 ticks from rest anyway, so the settle overlaps
      time in which the car physically cannot move.

  (b) resting ON TOP of an NPC vehicle (loc.z ~ 2.2 with vel.z ~ 0). Settling cannot
      help -- it is stably parked on another car. Only a re-roll of the spawn does.

WHY THIS IS FAITHFUL, NOT CHEATING
----------------------------------
We are not altering the MDP. Every episode still begins from SAR's own spawn
distribution (random lane in 1-4, same pose, same NPC logic). We remove only the
physics glitches that SAR's own authors labelled "not algorithm's fault, but the
simulator sometimes throws the car in the air wierdly" (carla_env.py:396) and
explicitly wrote a guard to detect. Their handling of that detection -- return a
doomed observation and let the caller burn an episode -- is what we replace, and the
episodes we discard are exactly the ones they intended to discard.

Report `env.glitch_stats()` alongside any result so the discard rate is on the record.
"""
from __future__ import print_function

# CRITICAL: torch MUST be imported before carla. The reverse order segfaults
# (exit 139) -- CARLA 0.9.6's Boost.Python .so and torch's C++ runtime clash.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import os
import sys

SAR_ROOT = os.environ.get("SAR_ROOT", os.path.expanduser("~/SAR"))
if SAR_ROOT not in sys.path:
    sys.path.insert(0, SAR_ROOT)

# RoamingAgentModified is defined inside carla_env.py itself (line 657), not in
# agents/navigation/roaming_agent.py -- import it from the same module.
from CARLA_.PythonAPI.carla.agents.navigation.carla_env import (  # noqa: E402
    CarlaEnv,
    CarlaSyncMode,
    RoamingAgentModified,
)
import math    # noqa: E402
import random  # noqa: E402  -- SAR's reset_other_vehicles draws from the global `random`
import carla   # noqa: E402  -- safe here: torch is already imported above

# ---------------------------------------------------------------------------------------
# Sensor-tick timeout: SAR hardcodes `self.sync_mode.tick(timeout=2.0)` (carla_env.py:543).
# Two seconds is not enough to receive six queues (world tick + 5 cameras) when the server
# is still warming up or is sharing the GPU, and a miss raises a bare `_queue.Empty` that
# kills the whole training process. Observed on the first env construction right after a
# server boot.
#
# The 2.0 is passed explicitly, so raising the default on tick() would not help -- we widen
# it inside _retrieve_data instead. This never shortens a timeout, only lengthens it.
SENSOR_TIMEOUT = float(os.environ.get("CARLA_SENSOR_TIMEOUT", 30.0))

if not getattr(CarlaSyncMode, "_vibes_timeout_patched", False):
    _orig_retrieve_data = CarlaSyncMode._retrieve_data

    def _retrieve_data_patched(self, sensor_queue, timeout):
        return _orig_retrieve_data(self, sensor_queue, max(timeout, SENSOR_TIMEOUT))

    CarlaSyncMode._retrieve_data = _retrieve_data_patched
    CarlaSyncMode._vibes_timeout_patched = True

SETTLE_TICKS = 40
MAX_RESPAWN = 10
# Spawn-cleanliness thresholds (added 2026-08-20, doc 18 C3). SAR's reset_vehicle() places
# the ego at a fixed pose (yaw -90) and explicitly ZEROES its velocity and angular velocity
# (carla_env.py:429-441), so at the end of the settle a clean car is stationary and
# lane-aligned; anything else was shoved. A settled car idles at ~0.01 m/s.
SPAWN_MAX_SPEED = float(os.environ.get("CARLA_SPAWN_MAX_SPEED", 0.5))      # m/s, horizontal
SPAWN_MAX_YAW_DEG = float(os.environ.get("CARLA_SPAWN_MAX_YAW_DEG", 15.0))  # vs lane heading
SPAWN_MAX_TILT_DEG = float(os.environ.get("CARLA_SPAWN_MAX_TILT_DEG", 30.0))  # roll / pitch
# NPC autopilot timing. Stock SAR enables autopilot in the spawn batch and takes its first
# step two ticks later. Our 40-tick settle (the 2026-08-17 glitch fix) therefore gave
# autopilot NPCs 2 s to drive into a stationary ego at y=0 from up to 40 m behind -- the
# measured result (doc 25 s2) was ~half of all episodes starting shoved, rotated, or in
# collision. Default 1: NPCs spawn static, the ego settles and is checked, THEN autopilot
# is enabled and one tick runs -- i.e. the episode starts the way SAR's does, with NPCs
# ~1-2 ticks into autopilot. 0 restores the old behaviour (for A/B measurement only).
NPC_AUTOPILOT_AFTER_SETTLE = os.environ.get("CARLA_NPC_AUTOPILOT_AFTER_SETTLE", "1") != "0"


class PatchedCarlaEnv(CarlaEnv):
    """CarlaEnv whose reset() is guaranteed to return a settled, on-road, lane-aligned,
    stationary vehicle, with NPC traffic that starts moving only once the ego is settled."""

    _FAIL_KEYS = ("airborne", "on_top", "off_road", "moving", "rotated", "tilted")

    def __init__(self, *args, **kwargs):
        self.settle_ticks = kwargs.pop("settle_ticks", SETTLE_TICKS)
        self.max_respawn = kwargs.pop("max_respawn", MAX_RESPAWN)
        self._resets = 0
        self._respawns = 0
        self._fail_counts = {k: 0 for k in self._FAIL_KEYS}
        self._check_log = []            # (speed, |dyaw|) at every check instant
        self._npc_spawn_failures = 0
        self._npc_spawn_failures_total = 0
        super(PatchedCarlaEnv, self).__init__(*args, **kwargs)

    # ---- NPC spawn: single application, autopilot deferred (doc 18 C5 + C3) -----------
    def reset_other_vehicles(self):
        """SAR's reset_other_vehicles (carla_env.py:443-485) with two changes:
        (1) the spawn batch is applied ONCE and failures are counted instead of appended
            as actor id 0 (their double application never landed an extra car in 1,892
            measured episodes, but it made `len(vehicles_list)` a constant 10);
        (2) with NPC_AUTOPILOT_AFTER_SETTLE, SpawnActor is not chained to SetAutopilot --
            `_enable_npc_autopilot()` does that after the ego's settle passes.
        The two-loop structure and every `random` call are kept verbatim so a given seed
        produces the same spawn realisation as before."""
        if not self.is_other_cars:
            return

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        num_vehicles = 10
        other_car_transforms = []
        for _ in range(num_vehicles):
            lane_id = random.choice([1, 2, 3, 4])
            start_x = 1.5 + 3.5 * lane_id
            start_y = random.uniform(-40., 40.)
            transform = carla.Transform(carla.Location(x=start_x, y=start_y, z=0.1),
                                        carla.Rotation(yaw=-90))
            other_car_transforms.append(transform)

        batch = []
        for n, transform in enumerate(other_car_transforms):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            cmd = carla.command.SpawnActor(blueprint, transform)
            if not NPC_AUTOPILOT_AFTER_SETTLE:
                cmd = cmd.then(carla.command.SetAutopilot(carla.command.FutureActor, True))
            batch.append(cmd)

        self._npc_spawn_failures = 0
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                self._npc_spawn_failures += 1
            else:
                self.vehicles_list.append(response.actor_id)
        self._npc_spawn_failures_total += self._npc_spawn_failures

    def _enable_npc_autopilot(self):
        if not self.is_other_cars or not self.vehicles_list:
            return
        self.client.apply_batch_sync(
            [carla.command.SetAutopilot(aid, True) for aid in self.vehicles_list], False)

    # ---- health check on a freshly placed vehicle -------------------------
    def _spawn_failures(self):
        """Every reason the settled ego is NOT a clean start. Empty list == clean.

        (a) airborne / on_top: the vertical-glitch guard, mirroring carla_env.py:397-404.
        (b) off_road: `get_waypoint(loc, project_to_road=False)` is None -- exactly what
            the env calls `off_road` (carla_env.py:364). Added 2026-08-17 (27% of run-6
            episodes were off-road at spawn).
        (c) moving / rotated / tilted: added 2026-08-20 after the per-frame recorder showed
            ~half of episodes starting at >0.5 m/s, >=10 deg off the lane, or in collision
            (doc 25 s2). reset_vehicle() zeroes velocity and spawns at the lane heading, so
            a clean car is stationary and aligned; the lane heading is read from the
            waypoint, not hard-coded, so the check survives any future start pose.
        Also records (speed, |dyaw|) at this instant into _check_log for measurement.
        """
        v = self.vehicle.get_velocity()
        tr = self.vehicle.get_transform()
        loc = tr.location
        failed = []
        if v.z > 1.0:
            failed.append("airborne")
        if loc.z > 0.5:
            failed.append("on_top")
        wp = self.map.get_waypoint(loc, project_to_road=False)
        speed = math.hypot(v.x, v.y)
        dyaw = float("nan")
        if wp is None:
            failed.append("off_road")
        else:
            dyaw = abs((tr.rotation.yaw - wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
            if dyaw > SPAWN_MAX_YAW_DEG:
                failed.append("rotated")
        if speed > SPAWN_MAX_SPEED:
            failed.append("moving")
        if abs(tr.rotation.roll) > SPAWN_MAX_TILT_DEG or abs(tr.rotation.pitch) > SPAWN_MAX_TILT_DEG:
            failed.append("tilted")
        self._check_log.append((speed, dyaw))
        for k in failed:
            self._fail_counts[k] += 1
        self._last_check = dict(speed=speed, dyaw=dyaw, vz=v.z, z=loc.z,
                                roll=tr.rotation.roll, pitch=tr.rotation.pitch, failed=failed)
        return failed

    def _spawn_is_clean(self):
        return not self._spawn_failures()

    # ---- the actual root cause of the "shoved spawn" (found 2026-08-20) ----------------
    # SAR reuses ONE ego actor across resets (reset_vehicle: set_transform + zero velocity,
    # carla_env.py:435-441) and CARLA keeps applying a vehicle's LAST VehicleControl until
    # it is replaced. So the previous episode's final control -- full throttle, for a
    # trained policy -- is still live during our 40-tick settle, and the "settled" ego
    # drives itself to ~10 m/s and rams the traffic ahead. Measured with
    # spawn_contamination_probe.py: median 9.5-10.2 m/s at the check instant in 77-82% of
    # checks, identical with NPC autopilot on or off (so the NPCs were never the shover).
    # Stock SAR takes its first step 2 ticks after reset, where a stale throttle is
    # harmless; the settle (our 2026-08-17 glitch fix) turned it into a 2 s launch. A stale
    # BRAKE is the other face of the same bug: the car sits parked through the episode
    # until the frame-count stuck rule fires at agent step 25 -- the "stuck at exactly 24
    # steps" lock of runs 7-8.
    _HOLD = None   # set lazily: carla is importable only after torch

    def _hold_still(self):
        if PatchedCarlaEnv._HOLD is None:
            PatchedCarlaEnv._HOLD = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0,
                                                         hand_brake=True)
        self.vehicle.apply_control(PatchedCarlaEnv._HOLD)

    def _release(self):
        # Neutral control, exactly what a freshly spawned actor has. The agent's first
        # step (carla_env.py's step()) applies its own control one tick later.
        self.vehicle.apply_control(carla.VehicleControl())
        self.vehicle.set_velocity(carla.Vector3D())
        self.vehicle.set_angular_velocity(carla.Vector3D())

    def _place_and_settle(self):
        self.reset_vehicle()
        self._hold_still()          # park BEFORE the first tick: the stale control is live
        self.world.tick()
        self.reset_other_vehicles()
        self.world.tick()
        for _ in range(self.settle_ticks):
            self.world.tick()

    # ---- the override ------------------------------------------------------
    def reset(self):
        self._resets += 1
        for attempt in range(self.max_respawn):
            self._place_and_settle()
            if self._spawn_is_clean():
                break
            self._respawns += 1
            if attempt == self.max_respawn - 1:
                # Do not silently hand back a broken episode -- that is the bug we
                # are fixing. Loud failure is correct here, and it must say WHICH check.
                c = self._last_check
                raise RuntimeError(
                    "spawn still glitched after %d attempts; failing check(s): %s "
                    "[speed=%.2f m/s dyaw=%.1f deg vel.z=%.3f loc.z=%.3f roll=%.1f pitch=%.1f]. "
                    "If the only failure is off_road and it repeats, the server is probably "
                    "sick rather than the spawn unlucky -- restart it via carla_supervisor"
                    % (self.max_respawn, ", ".join(c["failed"]) or "none (race?)",
                       c["speed"], c["dyaw"], c["vz"], c["z"], c["roll"], c["pitch"]))

        # Ego settled and verified: drop the parking control so it starts neutral, like a
        # fresh spawn (and like stock SAR's first step sees it).
        self._release()
        if NPC_AUTOPILOT_AFTER_SETTLE:
            # Now let traffic go, with the same single tick stock SAR gives it between
            # spawning NPCs and the first step.
            self._enable_npc_autopilot()
            self.world.tick()

        # ---- the rest is verbatim CarlaEnv.reset() (carla_env.py:419-427) ----
        self.agent = RoamingAgentModified(self.vehicle, follow_traffic_lights=False)
        self.count = 0
        self.dist_s = 0
        self.return_ = 0
        self.velocities = []
        obs, _, _, _ = self.step(action=None)
        return obs

    # ---- accounting --------------------------------------------------------
    def glitch_stats(self):
        n = len(self._check_log)
        speeds = [s for s, _ in self._check_log]
        dyaws = [d for _, d in self._check_log if d == d]   # drop NaN (off-road)
        return {
            "resets": self._resets,
            "respawns_needed": self._respawns,
            "respawn_rate": (float(self._respawns) / self._resets) if self._resets else 0.0,
            "settle_ticks": self.settle_ticks,
            "npc_autopilot_after_settle": NPC_AUTOPILOT_AFTER_SETTLE,
            "checks": n,
            "fail_counts": dict(self._fail_counts),
            "check_speed_median": sorted(speeds)[n // 2] if n else None,
            "check_speed_p90": sorted(speeds)[int(0.9 * (n - 1))] if n else None,
            "check_dyaw_median": sorted(dyaws)[len(dyaws) // 2] if dyaws else None,
            "check_dyaw_p90": sorted(dyaws)[int(0.9 * (len(dyaws) - 1))] if dyaws else None,
            "npc_spawn_failures_total": self._npc_spawn_failures_total,
            "npc_count_last": len(self.vehicles_list),
        }


if __name__ == "__main__":
    import argparse
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()

    env = PatchedCarlaEnv(render_display=0, display_text=0, changing_weather_speed=0.1,
                          rl_image_size=84, max_episode_steps=1000, frame_skip=4,
                          is_other_cars=True, port=a.port)
    bugs = 0
    for i in range(a.episodes):
        env.reset()
        for _ in range(a.steps):
            _, _, done, info = env.step(np.array([0.0, 0.5], dtype=np.float32))
            if done:
                bugs += (info.get("reason_episode_ended") == "carla_bug")
                break
    print("\ncarla_bug in first %d steps: %d/%d episodes"
          % (a.steps, bugs, a.episodes))
    print("glitch_stats: %s" % env.glitch_stats())
    env.finish()
    print("VERDICT: %s" % ("PASS" if bugs == 0 else "FAIL"))
