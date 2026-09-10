# CARLA 0.9.6 on DreamerV3 — runbook (canebrake)

DreamerV3 (Python 3.9+/JAX) cannot `import carla`: the CARLA 0.9.6 client egg is
Python 3.7 only. So the env runs in a **separate py3.7 worker process** and DreamerV3
talks to it over a TCP socket.

```
container: vibes-carla96  (--network host, GPU)          container: dv3  (--network host, GPU)
  worker.py  (conda env carla96, py3.7)                    python dreamerv3/main.py --configs carla
    ├─ carla_supervisor.CarlaServer → CarlaUE4.sh  :3000     └─ embodied/envs/carla.py
    ├─ PatchedCarlaEnv  (5 cam → 84×420, 'sar' reward)            connects to 127.0.0.1:2610
    └─ TCP listen :2610  ──────────────── socket ────────────────┘
```

All the CARLA-version pain (egg, `agents.navigation`, spawn glitches, server crashes,
sensor timeouts) is sealed in the worker. Vendored from `adversarial_dreamer@carla`
under `embodied/envs/carla_bridge/_vendor/`.

## What lands where

| piece | path |
|---|---|
| DreamerV3 adapter | `embodied/envs/carla.py` (`suite='carla'`, task `carla_town04`) |
| socket framing | `embodied/envs/carla_bridge/protocol.py` (py3.7 **and** py3.9 safe) |
| py3.7 worker | `embodied/envs/carla_bridge/worker.py` |
| vendored DBC env | `embodied/envs/carla_bridge/_vendor/` (`sar_env_patched.py`, `carla_supervisor.py`, `sar_env/CARLA_/…/carla_env.py`) |
| bridge-only smoke test | `embodied/envs/carla_bridge/smoke_client.py` |
| config | `configs.yaml` block `carla` + `env.carla` defaults; run `--configs carla size12m` |

## canebrake facts (confirm the ⚠️ ones)

- repo: `~/vivaans/dreamerv3` → `/workspace` in the running `dv3` container (image `dv3-dcs`)
- images: `vibes-carla96` (conda env `carla96` py3.7 + CARLA baked at `/home/carla/CARLA_0.9.6`), `dv3-dcs`
- CARLA also extracted on host: `/data/vivaans/CARLA_0.9.6`
- ⚠️ py3.7 egg: 0.9.6 ships only py3.5. Confirm `/home/carla/CARLA_0.9.6/PythonAPI/carla/dist/`
  has `carla-0.9.6-py3.7-linux-x86_64.egg` inside the `vibes-carla96` image; if not:
  `cp carla-0.9.6-py3.5-linux-x86_64.egg carla-0.9.6-py3.7-linux-x86_64.egg`
- ⚠️ `ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json` on host (worker needs it; pass a
  different path with `--egl-vendor-json` if it lives elsewhere)
- `--privileged` required on canebrake (NVML/cgroup); GPU 0 for the test (per Vivaan)
- ports: CARLA rpc `3000`/`3001`, bridge listen `2610` — chosen to miss the live
  `carla-ae_ramp5_500k` run's 2000-pool. `--network host` shares localhost across containers.

## 1. Start the worker (bridge container)

```sh
docker run -d --name carla-bridge --privileged --network host --shm-size=16g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v ~/vivaans/dreamerv3:/dv3:ro \
  vibes-carla96 tail -f /dev/null

docker exec -it carla-bridge conda run --no-capture-output -n carla96 \
  python /dv3/embodied/envs/carla_bridge/worker.py \
    --listen-port 2610 --carla-rpc-port 3000 --gpu 0 \
    --carla-root /home/carla/CARLA_0.9.6 \
    --out-size 64,256
```

The worker prints `listening on 127.0.0.1:2610 (CARLA not booted yet)` and then waits
for a client. It boots CARLA only *after* DreamerV3 (or the smoke client) connects.
Server log: `~/carla_server_3000.log` inside the container.

## 2a. Bridge-only smoke test (no JAX)

```sh
docker exec -it carla-bridge conda run --no-capture-output -n carla96 \
  python /dv3/embodied/envs/carla_bridge/smoke_client.py --addr 127.0.0.1:2610 --steps 30
```

Expect: `ready in ~40s`, `image (64, 256, 3) dtype=uint8`, a couple of episodes that
either survive 30 steps or end with a reason (`off_road` / `stuck` / `carla_bug` …).
Then restart the worker (it exits after the client disconnects) before step 2b.

## 2b. DreamerV3 smoke test (pipe + learning loop, ~a few min on GPU 0)

```sh
docker exec -it dv3 sh -lc '
  cd /workspace && git fetch origin && git checkout carla && git pull &&
  XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=0 \
  python dreamerv3/main.py \
    --logdir /workspace/logdir/carla/smoke_$(date +%s) \
    --configs carla size12m \
    --run.steps 1500 --run.train_ratio 8 --batch_size 8 \
    --run.log_every 20 --run.report_every 60 --run.save_every 120
'
```

Check `<logdir>/config.yaml` shows `task: carla_town04`, `env.carla.frame_skip: 4`.
Watch for `[carla] bridge ready`, then `episode/score` and `episode/length` lines.

## 3. Real run (500k agent steps = 2M sim frames; do when GPUs free)

Start a fresh worker (step 1), then:

```sh
docker exec -d dv3 sh -lc '
  cd /workspace &&
  XLA_FLAGS=--xla_gpu_enable_command_buffer= CUDA_VISIBLE_DEVICES=0 \
  python dreamerv3/main.py \
    --logdir /workspace/logdir/carla/carla_town04_s0 \
    --configs carla size12m --seed 0 \
    > ~/carla_logs/s0.log 2>&1
'
```

- `run.steps 500000`, `train_ratio 64`, `envs 1` come from the `carla` block.
- Resumable: rerun the identical command (same `--logdir`) — checkpoint covers
  step + agent + replay. Restart the worker too (it doesn't persist).
- Multiple seeds → give each its own worker: `--carla-rpc-port 3010 --listen-port 2611`
  etc., and `--env.carla.bridge_addr 127.0.0.1:2611`, one GPU each.

## 4. Final eval (100 episodes, matches the DCS campaign convention)

Separate worker on another port, then:

```sh
docker exec -it dv3 sh -lc '
  cd /workspace &&
  CUDA_VISIBLE_DEVICES=0 python dreamerv3/main.py \
    --logdir /workspace/logdir/carla/finaleval/carla_town04_s0 \
    --configs carla size12m --script eval_only \
    --run.from_checkpoint /workspace/logdir/carla/carla_town04_s0/ckpt/ \
    --run.final_eval_eps 100 \
    --env.carla.bridge_addr 127.0.0.1:2611
'
```

## Teardown

```sh
docker exec carla-bridge pkill -9 -f carla-rpc-port   # kill the CARLA server
docker rm -f carla-bridge
```

## Gotchas (from the vendored code's own comments)

- **Start the worker before DreamerV3.** The adapter's `connect_timeout` is 900s.
- **One client per worker.** worker.py exits when the client disconnects; restart it
  between smoke tests / runs.
- CARLA 0.9.6 crashes on long runs — the worker restarts the server + rebuilds the
  env and hands DreamerV3 a terminal obs (`reason='server_crash'`, counts as a real
  terminal). Agent checkpoints every 15 min by default, bump with `--run.save_every`.
- Image is resized in the worker (`--out-size 64,256`, → `(64,256,3)`). The encoder
  (`mults [2,3,4,4]`) needs each dim divisible by 16 and the result in `[3,16]`:
  `64→4`, `256→16` (boundary). `64,192` is the safe-margin fallback.
- Reward is `'sar'` (`vel_s·dt − 1e-4·collision_impulse − |steer|`), forced by the
  worker via `CARLA_REWARD=sar`. `is_terminal` True for `off_road / no_waypoints /
  carla_bug / stuck / server_crash`; the horizon hit (`success`, 1000 sim frames =
  250 agent steps) is `is_last` only → value bootstrapped. DreamerV3's `cont` head
  (on by default) consumes this.
