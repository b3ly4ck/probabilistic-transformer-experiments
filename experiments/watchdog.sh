#!/bin/bash
# Re-queue dead shards on a timer, and regenerate the results tables.
#
# Cluster jobs die for reasons unrelated to the experiment (a node reporting a GPU it cannot
# allocate, an unmounted filesystem, an expired allocation). `experiments/requeue.py` makes
# each of those recoverable; this loop makes the recovery automatic, so a grid does not end
# with a silently missing shard that looks like a hole in its design.
set -uo pipefail
cd /public/home/belyack/work/pt
PY=.venv/bin/python
while :; do
  echo "=== $(date +%H:%M:%S) ==="
  $PY -m experiments.requeue 2>&1 | grep -E "REQUEUE|to re-queue|nothing"
  $PY -m experiments.make_report >/dev/null 2>&1 && echo "results regenerated"
  sleep 600
done
