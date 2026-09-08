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
  # `grep -c` prints 0 AND exits 1 on an empty file, so `|| echo 0` appended a second zero and
  # the test below failed with "integer expression expected" -- the loop then fell through and
  # submitted `head -1` of an empty file, i.e. an empty spec. Observed 2026-09-09: one job
  # submitted with no arguments, which died on the sweep script's `set -u` without writing
  # anything. Count without the fallback, and refuse to submit a blank line whatever the count.
  n_pending=$(grep -cve '^\s*$' "$PENDING" 2>/dev/null)
  n_pending=${n_pending:-0}
  if [ "$n_pending" -eq 0 ] 2>/dev/null || [ -z "$n_pending" ]; then
    echo "$(date +%H:%M:%S) queue empty, exiting"
    exit 0
  fi
  mine=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -vc "^${FOREIGN}$" || true)
  if [ "${mine:-99}" -lt "$MAXJOBS" ]; then
    line=$(grep -ve '^\s*$' "$PENDING" | head -1)
    if [ -z "$line" ]; then
      echo "$(date +%H:%M:%S) queue empty, exiting"
      exit 0
    fi
    case "$line" in
      "!"*) cmd="${line#\! }" ;;
      *)    cmd="sbatch $DEFAULT_OPTS experiments/sweep.slurm $line" ;;
    esac
    if out=$(eval "$cmd" 2>&1); then
      echo "$(date +%H:%M:%S) submitted [$line] -> $out"
      # `grep -v` exits 1 when it filters out *every* line, which is exactly what happens on
      # the last item in the queue -- so `&& mv` was skipped, the line survived, and the queue
      # resubmitted its final entry once a minute forever. Observed on 2026-09-09: the same
      # factored shard submitted twice in 68 seconds, both refused by the shard lock.
      grep -vxF "$line" "$PENDING" > "$PENDING.tmp" || true
      mv "$PENDING.tmp" "$PENDING"
    else
      echo "$(date +%H:%M:%S) refused [$line]: $out"
    fi
  fi
  sleep "$SLEEP"
done
