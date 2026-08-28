#!/usr/bin/env bash
# DreamerV3 DCS campaign — batch 1, run on the viper HOST (not inside the container).
# nohup + per-run logfile (viper's tmux.conf breaks scripted windows).
#
# 3 GPUs (1,2,3); GPU 0 left idle — it runs hottest and dm_control's EGL render
# contexts pile onto physical GPU 0 regardless of CUDA_VISIBLE_DEVICES.
# XLA CUDA graphs disabled (--xla_gpu_enable_command_buffer=) — the NVRM Xid 13
# faults that took viper down on 2026-08-28 came from XLA's graph command stream.
#
#   WAVE=1 bash scripts/launch_dcs_batch1.sh   -> calib + cheetah_run seeds 0,1
#   WAVE=2 bash scripts/launch_dcs_batch1.sh   -> cheetah_run seeds 2,3,4
#   WAVE=3 bash scripts/launch_dcs_batch1.sh   -> walker_walk seeds 0,1
#
# Stable logdir per run (no {timestamp}): re-run the identical WAVE command after
# a crash/reboot and it resumes from the last checkpoint (save_every = 15 min).
# Runs survive ssh disconnect (nohup), NOT a host reboot (relaunch; container
# survives as Exited -> docker start dv3).
#
# monitor:  tail -f ~/dcs_logs/*.log
#           dmesg -w | grep -i xid        # in a spare terminal — Xid = a seed about to die
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
XLA=${XLA:-"--xla_gpu_enable_command_buffer="}

mkdir -p "$HOSTLOGS"
docker exec "$CONTAINER" test -d "$DAVIS/DAVIS/JPEGImages/480p" \
  || { echo "DAVIS not found at $DAVIS inside $CONTAINER"; exit 1; }
case "$OUTPUTS" in *wandb*) docker exec "$CONTAINER" test -f /root/.netrc \
  || echo "WARNING: /root/.netrc missing — wandb will fail (runbook 1b)";; esac

run() {  # run <gpu> <name> <main.py args...>
  local gpu=$1 name=$2; shift 2
  nohup docker exec \
    -e CUDA_VISIBLE_DEVICES="$gpu" -e WANDB_PROJECT="$WANDB_PROJECT" -e XLA_FLAGS="$XLA" \
    "$CONTAINER" python dreamerv3/main.py \
      --logdir "$LOGROOT/$name" \
      --logger.outputs "$OUTPUTS" "$@" \
    >> "$HOSTLOGS/$name.log" 2>&1 &
  echo "$name -> GPU $gpu   (host pid $!, log $HOSTLOGS/$name.log)"
  sleep 2
}

DCS=(--configs dcs "$SIZE" --env.dcs.davis_path "$DAVIS")
W2=(--env.dcs.action_repeat 2 --run.train_ratio 100)   # walker: ar=2, train_ratio=50*2

case "$WAVE" in
  1)
    run 1 calib_walker_walk_s0 \
      --configs dmc_vision "$SIZE" --task dmc_walker_walk \
      --script train_eval --run.debug False --run.envs 4 --run.eval_envs 4 \
      --run.eval_eps 10 --run.report_every 600 \
      --env.dmc.repeat 2 --run.train_ratio 100 --run.steps 500000 --seed 0
    run 2 dcs_cheetah_run_s0 "${DCS[@]}" --task dcs_cheetah_run --seed 0
    run 3 dcs_cheetah_run_s1 "${DCS[@]}" --task dcs_cheetah_run --seed 1
    ;;
  2)
    run 1 dcs_cheetah_run_s2 "${DCS[@]}" --task dcs_cheetah_run --seed 2
    run 2 dcs_cheetah_run_s3 "${DCS[@]}" --task dcs_cheetah_run --seed 3
    run 3 dcs_cheetah_run_s4 "${DCS[@]}" --task dcs_cheetah_run --seed 4
    ;;
  3)
    run 1 dcs_walker_walk_s0 "${DCS[@]}" --task dcs_walker_walk "${W2[@]}" --seed 0
    run 2 dcs_walker_walk_s1 "${DCS[@]}" --task dcs_walker_walk "${W2[@]}" --seed 1
    ;;
  *) echo "WAVE must be 1, 2, or 3"; exit 1 ;;
esac

echo
echo "monitor:  tail -f $HOSTLOGS/*.log   |   dmesg -w | grep -i xid"
echo "gpus:     docker exec $CONTAINER nvidia-smi   (1 python proc each on GPUs 1-3; GPU 0 idle bar EGL)"
