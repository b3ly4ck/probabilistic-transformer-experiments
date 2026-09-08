#!/bin/bash
#SBATCH --job-name=pt-repro
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
#
# Is the 2026-08 record reproducible, and is the new sweep runner faithful to the old driver?
#
# The first cell of the exp4 grid (d=16, h=2, lr=0.02, frozen b, alpha_Z=1, 6000 steps,
# seed 0) is bit-for-bit the same configuration as job 940848 of 2026-08-10, recorded at
# val 430.04. Re-run through the new runner it landed at 610.88, with the first evaluation
# agreeing to 0.1 (711.61 vs 711.7) and the traces separating only after it.
#
# Three explanations; this job separates them.
#   (A) the refactor of src/ changed the mathematics -> OLD src and NEW src disagree here;
#   (B) the new harness differs from the old driver   -> old driver and sweep runner disagree
#                                                        under the same src;
#   (C) the configuration is chaotic and 430.04 was one draw -> everything agrees only to
#       within a large spread, and the spread is itself the finding.
#
# (C) is a real possibility, not an excuse: the 2026-08 report documents this very
# configuration reverting from 516.1 to 693.9 between two evaluations and then recovering,
# and names that excursion as evidence that the message/unary balance gates learning. A run
# that passes through such an excursion is not expected to reproduce across two GPUs with
# different reduction orders. If (C) holds, no single-seed number from the undamped
# configuration can go in a paper -- which is worth knowing before writing one.
set -euo pipefail
REPO=/public/home/belyack/work/pt
if ! cd "$REPO" 2>/dev/null; then
  echo "FATAL: $REPO is not reachable from $(hostname) -- exclude this node" >&2
  exit 1
fi
PY=/public/software/anaconda3/bin/python
echo "node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Allocate and multiply, not just `is_available()`: a GPU held in exclusive-process mode by
# another job reports as available and fails only at the first allocation. Job 998640 died
# that way on ai_gpu28 after three other jobs of this sweep had already taken GPUs there.
$PY -c "
import torch, sys
assert torch.cuda.is_available(), 'no CUDA on this node'
try:
    x = torch.randn(1024, 1024, device='cuda'); torch.cuda.synchronize()
    assert torch.isfinite(x @ x).all()
except RuntimeError as e:
    sys.exit('FATAL: CUDA present but unusable (%s): %s' % (torch.cuda.get_device_name(0), e))
print('torch', torch.__version__, torch.cuda.get_device_name(0), 'allocation OK')
"

# `git archive` writes a tree to a tarball without touching the index or the working tree --
# unlike `git --work-tree=... checkout`, which would stage the old files in this repo.
OLD=/tmp/pt-old-${SLURM_JOB_ID:-$$}
rm -rf "$OLD"; mkdir -p "$OLD/data"
git archive cc28e96 src experiments | tar -x -C "$OLD"
cp -r data/ptb "$OLD/data/"
mkdir -p "$OLD/checkpoints" checkpoints  # the old driver saves one, unconditionally

echo; echo "=== (A) OLD src (cc28e96), OLD driver ==========================="
( cd "$OLD" && $PY -m experiments.exp1_language_modeling.ptb_attack \
    --ladder 16 --freeze-b --steps 6000 --tag reproA ) 2>&1 | tail -4

echo; echo "=== (B) NEW src, OLD driver ====================================="
$PY -m experiments.exp1_language_modeling.ptb_attack \
    --ladder 16 --freeze-b --steps 6000 --tag reproB 2>&1 | tail -4

echo; echo "=== (C) NEW src, NEW runner, seeds 0/1/2 (+ damped controls) ===="
$PY -m experiments.sweep --spec experiments.exp3_readout.spec_repro \
    --out experiments/exp3_readout/repro_check.json 2>&1 | grep -E "^  -> |^\[[0-9]"

echo; echo "=== traces ==="
$PY - <<'PYEOF'
import json, pathlib
for tag, path in (("A(old src, old driver)", "/tmp/pt-old-%s/experiments/exp1_language_modeling/reproA.json" % __import__("os").environ.get("SLURM_JOB_ID","")),
                  ("B(new src, old driver)", "experiments/exp1_language_modeling/reproB.json")):
    p = pathlib.Path(path)
    if not p.exists():
        print(tag, "MISSING", path); continue
    r = json.load(open(p))["results"][0]
    print(f"{tag}: val {r['val_ppl']} train {r['train_ppl']} msg {r['msg_over_unary']} trace {r['val_trace']}")
p = pathlib.Path("experiments/exp3_readout/repro_check.json")
if p.exists():
    for row in json.load(open(p))["rows"]:
        if "error" in row: print("ERROR", row["cell"]["name"], row["error"]); continue
        print(f"C {row['cell']['name']}: val {row['val_ppl']} final {row['val_ppl_final']} "
              f"train {row['train_ppl']} trace {row['val_trace']}")
PYEOF
rm -rf "$OLD"
echo; echo "ALL DONE"
