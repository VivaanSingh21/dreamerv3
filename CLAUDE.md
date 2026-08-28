# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A JAX reimplementation of DreamerV3 (Hafner et al.). A model-based RL agent: learns a
recurrent world model (RSSM) from replayed experience, then trains an actor-critic purely
on trajectories imagined inside that world model. Fixed hyperparameters across domains.

Note: this codebase is **JAX / ninjax**, not PyTorch. `jax`, `optax`, `ninjax` (`nj`),
and the in-repo `elements` / `embodied` libraries are the substrate.

## Commands

Run training (entry point is always `dreamerv3/main.py`):

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/{timestamp} \
  --configs dmc_vision \
  --task dmc_walker_walk \
  --run.train_ratio 256
```

- `--configs` names one or more blocks from `dreamerv3/configs.yaml`, applied in order over
  `defaults` (e.g. `--configs dmc_vision size50m`). Any leaf in the resolved config is a CLI
  flag: `--run.train_ratio`, `--batch_size`, `--env.dmc.repeat`, `--jax.platform cpu`, etc.
- `{timestamp}` in `logdir` is substituted. The resolved config is written to
  `<logdir>/config.yaml` on start — always check that file to confirm flags landed.
- `--configs debug` shrinks the net/batch/intervals for fast smoke tests (does not learn).
- Resume: rerun the exact same command with the same `--logdir`. Checkpoint at
  `<logdir>/ckpt` covers step counter, agent params/optimizer, **and** replay; replay chunks
  also stream to `<logdir>/replay/`.
- CPU: `--jax.platform cpu`. GPU is the default and expects `jax[cuda12]`.

View results:

```sh
python -m scope.viewer --basedir ~/logdir --port 8000
```

Metrics are also written as `<logdir>/metrics.jsonl` and `<logdir>/scores.jsonl`. Enable
`wandb` / `tensorboard` by adding them to `--logger.outputs`.

Tests (pytest, no config file — run from repo root):

```sh
python -m pytest embodied/tests/                                  # all
python -m pytest embodied/tests/test_train.py::TestTrain::test_run_loop   # one
```

`embodied/perf/` holds standalone perf/bandwidth scripts, not pytest tests.

Docker: `docker build -f Dockerfile -t img .` then run `dreamerv3/main.py` inside.
The image installs all optional env deps and sets `MUJOCO_GL=egl`; `entrypoint.sh` wraps
the command in `xvfb-run`.

## Architecture

Two layers: **`dreamerv3/`** is the algorithm; **`embodied/`** is a general RL harness that
knows nothing about Dreamer specifically.

### `dreamerv3/`
- `main.py` — config resolution + wiring. `make_agent`, `make_replay`, `make_env`,
  `make_stream`, `make_logger` are constructed here and handed to a run script chosen by
  `--script` (`train`, `train_eval`, `eval_only`, `parallel`). `make_env` maps a
  `<suite>_<task>` string to an env class via a dispatch dict.
- `configs.yaml` — every hyperparameter. `defaults` block + task blocks (`dmc_vision`,
  `dmc_proprio`, `atari`, ...) + `sizeNm` blocks. Size/debug blocks use **regex keys**
  (`.*\.rssm`, `.*\.units`) that rewrite matching leaves anywhere in the tree.
- `agent.py` — `Agent` ties together encoder, RSSM dynamics, decoder, reward/cont heads,
  policy, value + slow value. `loss()` has three parts: world-model loss (recon + reward +
  cont + dynamics/representation KL), imagination actor-critic loss (`imag_loss`), and a
  replay-value loss on real transitions (`repl_loss`). `train()` runs **one** `self.opt`
  call = one optimizer step over all world-model + AC modules jointly (not separate
  optimizers per component).
- `rssm.py` — `RSSM` (categorical stochastic + deterministic GRU-style recurrence, with
  `observe` / `imagine` / `loss`), plus the conv `Encoder` / `Decoder`.

### `embodied/`
- `core/` — `Driver` (steps a batch of envs, fires `on_step` callbacks), `replay.py` +
  `selectors.py` (FIFO replay with uniform / prioritized / recency mixture sampling),
  `streams.py` (batch assembly, prefetch, consec-chunk stitching), `wrappers.py`
  (action normalization/clipping, dtype unification, space checks).
- `run/` — the training-loop scripts. `train.py` is the core loop: driver collects steps
  into replay; `should_train` (a `Ratio` on `train_ratio / (batch_size*batch_length)`)
  decides how many `agent.train` calls to run per collected step; periodic report / log /
  save via `LocalClock`. `parallel.py` runs actor, replay, envs, and logger as separate
  processes (`--script parallel`, with `parallel_env` / `parallel_replay` subprocess
  entry points).
- `jax/` — the JAX agent base (`agent.py`: jit, sharding, params, policy/train device
  split, async metric fetch), `nets.py`, `heads.py` (`MLPHead` with twohot / symlog /
  categorical outputs), `opt.py` (`Optimizer` with adaptive grad clipping), `transform.py`.
- `envs/` — one adapter per suite. `dmc.py` wraps `dm_control.suite.load(domain, task)`,
  applies `ActionRepeat`, and renders a `(64,64,3)` uint8 `image` observation each step
  (camera 0 by default; `MUJOCO_GL` defaults to `egl` if unset). `from_dm.py` is the
  generic dm_env → embodied `Env` bridge.
- `dmc_dcs.py` (suite `dcs`) — Distracting Control Suite for the HRSSM/VIBES/DreamerV3
  benchmark. Same shape as `dmc.py` but loads `distracting_control.suite.load` (PyPI
  `distracting-control==0.1.1rc3`, the exact package VIBES uses) with the `easy` tier.
  `main.py::make_env` takes a `mode` arg that routes train envs to the DAVIS `train` video
  partition and eval envs to `val`. See `DCS_RUNBOOK.md`. Needs a real EGL context
  (`NVIDIA_DRIVER_CAPABILITIES=all,graphics`) — the background distractor uploads a texture
  to the GL context every physics step.

### Key config knobs
- `run.train_ratio` — replayed env-steps trained per real env-step (data efficiency vs
  compute). Per-task defaults in `configs.yaml` (`dmc_vision` = 256, `dmc_proprio` = 1024).
- `env.dmc.repeat` — action repeat (default 1). `make_logger` multiplies logged step
  counts by this so x-axis is env frames. (The `dcs` suite deliberately names its knob
  `env.dcs.action_repeat` instead, so the logger multiplier stays 1 and the x-axis is in
  environment interactions.)
- `run.steps` — training length in env steps (`dmc_*` default `1.1e6`).
- `batch_size` × `batch_length` — 16 × 64 default; sequence batch for the world model.
- `agent.dyn.rssm.{deter,hidden,stoch,classes}` — world-model capacity (overridden by size blocks).

### Baselines / scoring
`scores/*.json.gz` holds published curves for DreamerV3 and competing methods per benchmark;
`baselines.yaml` has random/human normalization bounds; `plot.py` and `scores/view.py`
render comparisons from `metrics.jsonl` outputs.
