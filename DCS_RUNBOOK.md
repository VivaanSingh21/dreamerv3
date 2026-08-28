# DreamerV3 on the Distracting Control Suite — runbook

DreamerV3 leg of the HRSSM / VIBES / DreamerV3 three-way DCS benchmark.
**VIBES is the ground truth**; every protocol choice below matches it.
See the decision map for the reasoning behind each number.

---

## What was added

| file | change |
|---|---|
| `embodied/envs/dmc_dcs.py` | new env — wraps `distracting_control.suite.load(..., from_pixels=False)` and renders itself, same pattern as `embodied/envs/dmc.py` |
| `dreamerv3/main.py` | hard-set `MUJOCO_GL=egl` at top; register `dcs` suite; `make_env` takes `mode` and routes train→`train` / eval→`val` DAVIS partitions |
| `dreamerv3/configs.yaml` | `env.dcs` defaults + a `dcs` config block |
| `requirements.txt` | `distracting-control==0.1.1rc3`, `dm_control==1.0.15`, `mujoco==2.3.7` |

The distraction source is the **same PyPI package VIBES uses** (`distracting-control==0.1.1rc3`,
a gym wrapper on Google's original `distracting_control`) — *not* HRSSM's modified fork.
The `easy` tier bakes in: distractor scale **0.1**, first **4** DAVIS videos per
partition, all three distractors (background + camera + color), dynamic.

---

## 1. DAVIS videos

The background distractor reads `<davis_path>/DAVIS/JPEGImages/480p/<video>/*.jpg`.
Only 8 folders are ever touched (first 4 of the DAVIS-2017 train list, first 4 of val),
but simplest is to drop in the whole set.

```bash
# rsync the verified copy from canebrake (same source HRSSM used)
rsync -avz --progress -e "ssh -p 17236" \
  canebrake@fuegodos.sbp.ri.cmu.edu:~/vivaans/HRSSM/env/data/DAVIS/ \
  /workspace/dcs_data/DAVIS/
# verify
find /workspace/dcs_data/DAVIS/JPEGImages/480p -maxdepth 1 -type d | wc -l   # expect 90+
```

Then pass `--env.dcs.davis_path /workspace/dcs_data` on every launch.

---

## 2. Container / EGL

The background distractor uploads a texture to the GL context **every physics step**,
so EGL must actually work:

- container launched with `NVIDIA_DRIVER_CAPABILITIES=all,graphics` (not just `compute,utility`)
- `MUJOCO_GL=egl` — `main.py` now hard-sets this, but the container still needs the driver caps
- if you see `Failed to create EGL context` or OSMesa fallback, that's the driver-caps issue

**Compatibility risk to check on first run:** `distracting_control/background.py` does
`from dm_control.mujoco.wrapper import mjbindings` and calls
`mjbindings.mjlib.mjr_uploadTexture`. This legacy path was removed in newer dm_control.
`requirements.txt` pins `dm_control==1.0.15` / `mujoco==2.3.7` where it still works. If the
import or that call fails, either bump the pins down until it imports, or monkeypatch the
texture upload in `dmc_dcs.py` to call `mujoco.mjr_uploadTexture` directly.

---

## 3. Smoke test (does it run at all)

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dcs_smoke/{timestamp} \
  --configs dcs debug \
  --task dcs_cheetah_run \
  --env.dcs.davis_path /workspace/dcs_data
```

`debug` shrinks everything. Watch for: env constructs without error, an `image` obs of
shape (64, 64, 3), `episode/score` lines appearing. Then check the resolved config:

```bash
cat ~/logdir/dcs_smoke/*/config.yaml | grep -A2 'dcs:'
```

---

## 4. Timing test — pick the model size

Run the **same task** at each size for a fixed step count, read the steady-state
`fps/policy` from the logs (ignore the first 1–2 report lines — JIT warmup), then:

```
hours to 500k interactions  ≈  500000 / fps_policy / 3600
```

```bash
for SIZE in size1m size12m size50m size200m; do
  python dreamerv3/main.py \
    --logdir ~/logdir/dcs_timing/$SIZE/{timestamp} \
    --configs dcs $SIZE \
    --task dcs_cheetah_run \
    --env.dcs.davis_path /workspace/dcs_data \
    --run.steps 40000 \
    --run.log_every 60 \
    --script train        # plain train (no eval) is enough for a timing number
done
```

(Run them one at a time, or on separate GPUs — packing distorts the numbers.)

Record for each size: `fps/policy`, `fps/train`, peak VRAM (`nvidia-smi`), and whether it
OOMs. `size200m` is the published default; `size1m` ≈ VIBES's ~2.5–3M params,
`size12m` ≈ HRSSM's ~10M. The decision is: run DreamerV3 at its published size and note
it's much bigger, or scale down to match — driven by how long the above says it'll take.

---

## 5. The real benchmark run (once size is chosen)

Per task — set `action_repeat` and `train_ratio = 50 × action_repeat`:

| task | `--env.dcs.action_repeat` | `--run.train_ratio` |
|---|---|---|
| `dcs_cheetah_run` | 4 | 200 |
| `dcs_walker_walk` | 2 | 100 |
| `dcs_cartpole_swingup` | 8 | 400 |
| `dcs_reacher_easy` | 4 | 200 |
| `dcs_cup_catch` | 4 | 200 |
| `dcs_finger_spin` | 2 | 100 |

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dcs/<task>_<size>_seed<N>/{timestamp} \
  --configs dcs <size> \
  --task dcs_<task> \
  --env.dcs.action_repeat <ar> \
  --run.train_ratio <50*ar> \
  --run.steps 500000 \
  --env.dcs.davis_path /workspace/dcs_data \
  --seed <N>
```

- `--configs dcs` already sets `script: train_eval`, `run.eval_eps: 10`, `envs: 4`, `eval_envs: 4`.
- 5 seeds per task (matches VIBES Table 1).
- Final number = a **100-episode deterministic eval** from the final checkpoint on the val
  videos — run separately:

```bash
python dreamerv3/main.py \
  --logdir ~/logdir/dcs/<task>_<size>_seed<N>_finaleval/{timestamp} \
  --configs dcs <size> \
  --task dcs_<task> \
  --env.dcs.action_repeat <ar> \
  --env.dcs.davis_path /workspace/dcs_data \
  --script eval_only \
  --run.eval_eps 100 \
  --from_checkpoint ~/logdir/dcs/<task>_<size>_seed<N>/*/ckpt
```

Then aggregate mean ± 95% CI across the 5 seeds.

---

## Notes / where DreamerV3 can't match VIBES

- **Image size** — VIBES 84×84, we use 64×64 (DreamerV3's encoder is built for 64).
- **Latent** — VIBES continuous Gaussian; DreamerV3 categorical. Core to each method.
- **Reconstruction** — VIBES adversarial loss, no decoder; DreamerV3 pixel reconstruction.
- **Precision** — VIBES fp32; DreamerV3 stays on its native bf16.
- **Periodic reset** — VIBES does a hard reset every 250 episodes with extra training; no
  DreamerV3 analogue, not replicated.

State these plainly in the writeup — it's a methods comparison, not a matched-architecture one.
