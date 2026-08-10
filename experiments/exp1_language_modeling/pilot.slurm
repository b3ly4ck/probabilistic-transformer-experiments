#!/bin/bash
#SBATCH --job-name=pt-pilot
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# The repository .venv carries a CPU-only torch (2.13.0+cpu). The cluster anaconda has
# torch 2.0.1+cu117, so GPU runs use that interpreter; the source is py3.8-compatible.
set -euo pipefail
REPO=/public/home/belyack/work/pt
# Some nodes of the `critical` partition do not have /public mounted — ai_gpu32 and ai_gpu33
# were both observed on 2026-08-10. There `cd` fails under `set -e` and, because --output
# points into the same filesystem, the job dies in two seconds with no output file at all,
# which looks like a transient scheduler fault and is not one. Say so explicitly.
if ! cd "$REPO" 2>/dev/null; then
  echo "FATAL: $REPO is not reachable from $(hostname) — exclude this node" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/public/software/anaconda3/bin/python -c \
  "import torch; assert torch.cuda.is_available(), 'no CUDA — refusing to train on CPU'; \
   print('torch', torch.__version__, torch.cuda.get_device_name(0))"

/public/software/anaconda3/bin/python -m experiments.exp1_language_modeling.run_pilot "$@"
