#!/usr/bin/env bash
# Batch 1 of the DreamerV3 DCS campaign — run on the viper HOST (not inside the container).
# 8 runs, 2 per GPU across viper's 4x RTX 2080 Ti:
#   1 calibration (clean DMC walker_walk) + cheetah_run seeds 0-4 + walker_walk seeds 0-1
#
# One tmux window per run, each running `docker exec` into the container `dv3`.
# Windows are addressed by window-id (robust to base-index / tmux.conf quirks).
# Reattach:  tmux attach -t dcs      move: Ctrl-b n / Ctrl-b p      detach: Ctrl-b d
# A finished/crashed run drops back to a shell prompt in its window (scroll up for the traceback).

set -u

CONTAINER=${CONTAINER:-dv3}
SIZE=${SIZE:-size12m}
DAVIS=${DAVIS:-/workspace/dcs_data}
LOGROOT=${LOGROOT:-/workspace/logdir/dcs}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3-dcs}
OUTPUTS=${OUTPUTS:-'[jsonl,scope,wandb]'}   # override with OUTPUTS='[jsonl,scope]' to skip wandb

# --- preflight --------------------------------------------------------------
docker exec "$CONTAINER" test -d "$DAVIS/DAVIS/JPEGImages/480p" \
  || { echo "DAVIS not found at $DAVIS inside $CONTAINER"; exit 1; }
docker exec "$CONTAINER" test -f /root/.netrc \
  || echo "WARNING: /root/.netrc missing in container — wandb will fail (see runbook §1b)"
tmux has-session -t dcs 2>/dev/null && { echo "tmux session 'dcs' exists — 'tmux kill-session -t dcs' first"; exit 1; }

tmux new-session -d -s dcs
first_wid=$(tmux list-windows -t dcs -F '#{window_id}' | head -1)
_first_used=

launch() {  # launch <gpu> <name> <extra main.py args...>
  local gpu=$1 name=$2; shift 2
  local wid
  if [ -z "$_first_used" ]; then
    wid=$first_wid; _first_used=1
    tmux rename-window -t "$wid" "$name"
  else
    wid=$(tmux new-window -d -t dcs -P -F '#{window_id}' -n "$name")
  fi
  tmux set-option -t "$wid" remain-on-exit off
  tmux send-keys -t "$wid" \
    "docker exec -it -e CUDA_VISIBLE_DEVICES=$gpu -e WANDB_PROJECT=$WANDB_PROJECT $CONTAINER python dreamerv3/main.py --logdir $LOGROOT/$name/{timestamp} --logger.outputs '$OUTPUTS' $*" Enter
  echo "launched $name -> GPU $gpu ($wid)"
  sleep 2
}

# --- calibration: clean DMC (no distractors), campaign size/budget ---
launch 0 calib_walker_walk_s0 \
  --configs dmc_vision "$SIZE" --task dmc_walker_walk \
  --script train_eval --run.debug False --run.envs 4 --run.eval_envs 4 \
  --run.eval_eps 10 --run.report_every 600 \
  --env.dmc.repeat 2 --run.train_ratio 100 --run.steps 500000 --seed 0

# --- DCS: cheetah_run seeds 0-4 (dcs block: ar=4, train_ratio=200) ---
launch 0 dcs_cheetah_run_s0 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 0
launch 1 dcs_cheetah_run_s1 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 1
launch 1 dcs_cheetah_run_s2 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 2
launch 2 dcs_cheetah_run_s3 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 3
launch 2 dcs_cheetah_run_s4 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 4

# --- DCS: walker_walk seeds 0-1 (ar=2, train_ratio=100) ---
launch 3 dcs_walker_walk_s0 --configs dcs "$SIZE" --task dcs_walker_walk \
  --env.dcs.action_repeat 2 --run.train_ratio 100 --env.dcs.davis_path "$DAVIS" --seed 0
launch 3 dcs_walker_walk_s1 --configs dcs "$SIZE" --task dcs_walker_walk \
  --env.dcs.action_repeat 2 --run.train_ratio 100 --env.dcs.davis_path "$DAVIS" --seed 1

echo
tmux list-windows -t dcs
echo
echo "8 windows created. tmux attach -t dcs"
echo "After ~2 min: docker exec $CONTAINER nvidia-smi  (expect 2 python procs/GPU; watch GPU 0 for stacked EGL contexts)"
