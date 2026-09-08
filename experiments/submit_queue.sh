#!/bin/bash
# Drain a queue of sbatch submissions against the account's QOS job limit.
#
# The `critical` QOS allows a small number of *submitted* jobs per user
# (QOSMaxSubmitJobPerUserLimit, observed as 9 including jobs from other projects), so a
# 90-cell grid split into shards cannot all be submitted at once. This loop keeps the queue
# full without ever tripping the limit.
#
# One line of `experiments/pending.txt` is one submission. Two forms:
#
#   <spec-module> [args...]        -> sbatch <DEFAULT_OPTS> experiments/sweep.slurm <line>
#   ! <shell command>              -> run the command verbatim (for one-off jobs that need
#                                     their own sbatch options, e.g. pinning a node)
#
# A line is removed only once the submission is accepted, so a refusal simply retries later.
#
#   nohup bash experiments/submit_queue.sh > queue.log 2>&1 &
#
# MAXJOBS counts *all* of this user's jobs except the ones named in FOREIGN, because the QOS
# limit counts them too: an earlier version counted only `pt-sweep` and therefore kept trying
# to submit while the account was already at the cap, filling the log with refusals.
set -uo pipefail
cd /public/home/belyack/work/pt
PENDING=experiments/pending.txt
MAXJOBS=${MAXJOBS:-8}
FOREIGN=${FOREIGN:-ts-llm}
SLEEP=${SLEEP:-60}
DEFAULT_OPTS=${DEFAULT_OPTS:-"--time=06:00:00 --exclude=ai_gpu32,ai_gpu33"}
touch "$PENDING"

while :; do
  n_pending=$(grep -cve '^\s*$' "$PENDING" 2>/dev/null || echo 0)
  if [ "$n_pending" -eq 0 ]; then
    echo "$(date +%H:%M:%S) queue empty, exiting"
    exit 0
  fi
  mine=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -vc "^${FOREIGN}$" || true)
  if [ "${mine:-99}" -lt "$MAXJOBS" ]; then
    line=$(grep -ve '^\s*$' "$PENDING" | head -1)
    case "$line" in
      "!"*) cmd="${line#\! }" ;;
      *)    cmd="sbatch $DEFAULT_OPTS experiments/sweep.slurm $line" ;;
    esac
    if out=$(eval "$cmd" 2>&1); then
      echo "$(date +%H:%M:%S) submitted [$line] -> $out"
      grep -vxF "$line" "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
    else
      echo "$(date +%H:%M:%S) refused [$line]: $out"
    fi
  fi
  sleep "$SLEEP"
done
