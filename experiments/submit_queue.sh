#!/bin/bash
# Drain a queue of sbatch submissions against the account's QOS job limit.
#
# The `critical` QOS allows a small number of *submitted* jobs per user
# (QOSMaxSubmitJobPerUserLimit, observed as 9 including jobs from other projects), so a
# 90-cell grid split into shards cannot all be submitted at once. This loop keeps the queue
# full: one line of `experiments/pending.txt` is one set of arguments to
# `sbatch experiments/sweep.slurm`, and a line is removed only once sbatch accepts it.
#
#   echo "experiments.exp4_width_transfer.spec_width --shard 4/6" >> experiments/pending.txt
#   nohup bash experiments/submit_queue.sh > queue.log 2>&1 &
#
# MAXJOBS is the number of *this project's* jobs to keep in flight; it is counted by job
# name so an unrelated job of the same user does not starve the queue or overrun the limit.
set -uo pipefail
cd /public/home/belyack/work/pt
PENDING=experiments/pending.txt
MAXJOBS=${MAXJOBS:-7}
SLEEP=${SLEEP:-60}
touch "$PENDING"

while :; do
  n_pending=$(grep -cve '^\s*$' "$PENDING" 2>/dev/null || echo 0)
  if [ "$n_pending" -eq 0 ]; then
    echo "$(date +%H:%M:%S) queue empty, exiting"
    exit 0
  fi
  running=$(squeue -u "$USER" -h -n pt-sweep 2>/dev/null | wc -l)
  if [ "$running" -lt "$MAXJOBS" ]; then
    line=$(grep -ve '^\s*$' "$PENDING" | head -1)
    # shellcheck disable=SC2086
    if out=$(sbatch --time=06:00:00 --exclude=ai_gpu32,ai_gpu33 experiments/sweep.slurm $line 2>&1); then
      echo "$(date +%H:%M:%S) submitted [$line] -> $out"
      grep -vxF "$line" "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    else
      echo "$(date +%H:%M:%S) refused [$line]: $out"
    fi
  fi
  sleep "$SLEEP"
done
