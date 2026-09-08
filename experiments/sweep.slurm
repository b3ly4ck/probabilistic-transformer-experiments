#!/bin/bash
#SBATCH --job-name=pt-sweep
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#
# Generic wrapper: sbatch experiments/sweep.slurm <spec-module> [--shard i/n] [extra args]
#
# Some nodes of the `critical` partition do not have /public mounted -- ai_gpu32 and
# ai_gpu33 were both observed on 2026-08-10. There `cd` fails under `set -e` and, because
# --output points into the same filesystem, the job dies in two seconds with no output file
# at all, which looks like a transient scheduler fault and is not one. Say so explicitly.
set -euo pipefail
REPO=/public/home/belyack/work/pt
if ! cd "$REPO" 2>/dev/null; then
  echo "FATAL: $REPO is not reachable from $(hostname) -- exclude this node" >&2
  exit 1
fi
echo "node $(hostname)  job ${SLURM_JOB_ID:-none}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "nvidia-smi failed"
PY=/public/software/anaconda3/bin/python
# `torch.cuda.is_available()` and `get_device_name` both succeed on a GPU that is held in
# exclusive-process mode by somebody else; the failure only appears at the first allocation.
# On 2026-09-08 three of these jobs were placed on one node and two of them discovered that
# only after training had started. Allocate and compute here instead, so a job that cannot
# use its GPU dies in two seconds with a clear message and frees the slot.
$PY -c "
import torch, sys
assert torch.cuda.is_available(), 'no CUDA on this node -- refusing to train on CPU'
try:
    x = torch.randn(1024, 1024, device='cuda')
    torch.cuda.synchronize()
    assert torch.isfinite(x @ x).all()
except RuntimeError as e:
    sys.exit('FATAL: CUDA present but unusable on this node (%s): %s' % (torch.cuda.get_device_name(0), e))
print('torch', torch.__version__, torch.cuda.get_device_name(0), 'allocation OK')
"
SPEC=$1; shift
echo "spec: $SPEC  args: $*"
exec $PY -m experiments.sweep --spec "$SPEC" "$@"
