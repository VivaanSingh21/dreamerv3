#!/usr/bin/env bash
# DreamerV3 DCS campaign — batch 1, run on the viper HOST (not inside the container).
# Uses nohup + per-run logfiles (viper's tmux.conf fights scripted windows).
# 4 runs per wave, ONE per GPU (2x size12m on an 11GB card falls back to slow
# cuDNN kernels — don't pack). 8 runs total across two waves:
#
#   WAVE=1 bash scripts/launch_dcs_batch1.sh   -> calib + cheetah_run seeds 0,1,2
#   WAVE=2 bash scripts/launch_dcs_batch1.sh   -> cheetah_run seeds 3,4 + walker_walk seeds 0,1
#
# Run wave 2 once wave 1's GPUs free (or on another box).
# Runs survive ssh disconnect (nohup). They do NOT survive a host reboot — the
# container does (docker start dv3), just relaunch.
#
# monitor:  tail -f ~/dcs_logs/*.log
# stop all: docker exec dv3 pkill -9 -f dreamerv3

set -u

WAVE=${WAVE:-1}
CONTAINER=${CONTAINER:-dv3}
SIZE=${SIZE:-size12m}
DAVIS=${DAVIS:-/workspace/dcs_data}
LOGROOT=${LOGROOT:-/workspace/logdir/dcs}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3-dcs}
OUTPUTS=${OUTPUTS:-jsonl,scope,wandb}   # OUTPUTS=jsonl,scope to skip wandb
HOSTLOGS=${HOSTLOGS:-$HOME/dcs_logs}

mkdir -p "$HOSTLOGS"
docker exec "$CONTAINER" test -d "$DAVIS/DAVIS/JPEGImages/480p" \
  || { echo "DAVIS not found at $DAVIS inside $CONTAINER"; exit 1; }
case "$OUTPUTS" in *wandb*) docker exec "$CONTAINER" test -f /root/.netrc \
  || echo "WARNING: /root/.netrc missing — wandb will fail (runbook 1b)";; esac

run() {  # run <gpu> <name> <main.py args...>
  local gpu=$1 name=$2; shift 2
  nohup docker exec -e CUDA_VISIBLE_DEVICES="$gpu" -e WANDB_PROJECT="$WANDB_PROJECT" "$CONTAINER" \
    python dreamerv3/main.py \
      --logdir "$LOGROOT/$name/{timestamp}" \
      --logger.outputs "$OUTPUTS" "$@" \
    > "$HOSTLOGS/$name.log" 2>&1 &
  echo "$name -> GPU $gpu   (host pid $!, log $HOSTLOGS/$name.log)"
  sleep 2
}

DCS=(--configs dcs "$SIZE" --env.dcs.davis_path "$DAVIS")
W2=(--env.dcs.action_repeat 2 --run.train_ratio 100)   # walker: ar=2, train_ratio=50*2

if [ "$WAVE" = 1 ]; then
  run 0 calib_walker_walk_s0 \
    --configs dmc_vision "$SIZE" --task dmc_walker_walk \
    --script train_eval --run.debug False --run.envs 4 --run.eval_envs 4 \
    --run.eval_eps 10 --run.report_every 600 \
    --env.dmc.repeat 2 --run.train_ratio 100 --run.steps 500000 --seed 0
  run 1 dcs_cheetah_run_s0 "${DCS[@]}" --task dcs_cheetah_run --seed 0
  run 2 dcs_cheetah_run_s1 "${DCS[@]}" --task dcs_cheetah_run --seed 1
  run 3 dcs_cheetah_run_s2 "${DCS[@]}" --task dcs_cheetah_run --seed 2
else
  run 0 dcs_cheetah_run_s3 "${DCS[@]}" --task dcs_cheetah_run --seed 3
  run 1 dcs_cheetah_run_s4 "${DCS[@]}" --task dcs_cheetah_run --seed 4
  run 2 dcs_walker_walk_s0 "${DCS[@]}" --task dcs_walker_walk "${W2[@]}" --seed 0
  run 3 dcs_walker_walk_s1 "${DCS[@]}" --task dcs_walker_walk "${W2[@]}" --seed 1
fi

echo
echo "monitor:  tail -f $HOSTLOGS/*.log"
echo "gpus:     docker exec $CONTAINER nvidia-smi   (expect 1 python proc/GPU after ~2 min)"
echo "stop all: docker exec $CONTAINER pkill -9 -f dreamerv3"
