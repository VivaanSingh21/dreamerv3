#!/usr/bin/env bash
# Batch 1 of the DreamerV3 DCS campaign — run on the viper HOST (not inside the container).
# 8 runs, 2 per GPU across viper's 4x RTX 2080 Ti:
#   1 calibration (clean DMC walker_walk) + cheetah_run seeds 0-4 + walker_walk seeds 0-1
#
# Each tmux window wraps `docker exec` into the already-running container.
# Detach with Ctrl-b d. Reattach with: tmux attach -t dcs
# A crashed run leaves its pane open (remain-on-exit) with the traceback visible.

set -u

CONTAINER=${CONTAINER:-dv3}
SIZE=${SIZE:-size12m}
DAVIS=${DAVIS:-/workspace/dcs_data}
LOGROOT=${LOGROOT:-/workspace/logdir/dcs}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3-dcs}
OUTPUTS='[jsonl,scope,wandb]'   # set to '[jsonl,scope]' to skip wandb

# --- preflight ---------------------------------------------------------------
docker exec "$CONTAINER" test -d "$DAVIS/DAVIS/JPEGImages/480p" \
  || { echo "DAVIS not found at $DAVIS inside $CONTAINER"; exit 1; }
docker exec "$CONTAINER" test -f /root/.netrc \
  || echo "WARNING: /root/.netrc missing in container — wandb will fail (see runbook)"

tmux has-session -t dcs 2>/dev/null && { echo "tmux session 'dcs' already exists"; exit 1; }
tmux new-session -d -s dcs -n scratch
tmux set-option -g remain-on-exit on

launch() {  # launch <gpu> <name> <extra main.py args...>
  local gpu=$1 name=$2; shift 2
  tmux new-window -t dcs -n "$name" \
    "docker exec -it -e CUDA_VISIBLE_DEVICES=$gpu -e WANDB_PROJECT=$WANDB_PROJECT $CONTAINER \
       python dreamerv3/main.py \
         --logdir $LOGROOT/$name/{timestamp} \
         --logger.outputs '$OUTPUTS' \
         $* ; echo EXIT $?; exec bash"
  echo "launched $name on GPU $gpu"
  sleep 3
}

# --- calibration: clean DMC (no distractors), same size/budget as the campaign ---
launch 0 calib_walker_walk_s0 \
  --configs dmc_vision "$SIZE" --task dmc_walker_walk \
  --script train_eval --run.debug False --run.envs 4 --run.eval_envs 4 \
  --run.eval_eps 10 --run.report_every 600 \
  --env.dmc.repeat 2 --run.train_ratio 100 --run.steps 500000 --seed 0

# --- DCS: cheetah_run seeds 0-4 (dcs block already sets ar=4, train_ratio=200) ---
launch 0 dcs_cheetah_run_s0 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 0
launch 1 dcs_cheetah_run_s1 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 1
launch 1 dcs_cheetah_run_s2 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 2
launch 2 dcs_cheetah_run_s3 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 3
launch 2 dcs_cheetah_run_s4 --configs dcs "$SIZE" --task dcs_cheetah_run --env.dcs.davis_path "$DAVIS" --seed 4

# --- DCS: walker_walk seeds 0-1 (ar=2, train_ratio=50*2=100) ---
launch 3 dcs_walker_walk_s0 --configs dcs "$SIZE" --task dcs_walker_walk \
  --env.dcs.action_repeat 2 --run.train_ratio 100 --env.dcs.davis_path "$DAVIS" --seed 0
launch 3 dcs_walker_walk_s1 --configs dcs "$SIZE" --task dcs_walker_walk \
  --env.dcs.action_repeat 2 --run.train_ratio 100 --env.dcs.davis_path "$DAVIS" --seed 1

tmux kill-window -t dcs:scratch 2>/dev/null || true
echo
echo "All 8 launched. tmux attach -t dcs   (Ctrl-b n/p to move between windows, Ctrl-b d to detach)"
echo "After ~2 min, check: docker exec $CONTAINER nvidia-smi   (confirm 2 python procs per GPU, watch GPU 0 for stacked EGL contexts)"
