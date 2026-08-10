#!/bin/bash
#SBATCH --job-name=pt-attack
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
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
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "nvidia-smi failed"
/public/software/anaconda3/bin/python -c \
  "import torch; assert torch.cuda.is_available(), 'no CUDA on this node — refusing to train on CPU'; \
   print('torch', torch.__version__, torch.cuda.get_device_name(0))"
/public/software/anaconda3/bin/python -m experiments.exp1_language_modeling.ptb_attack "$@"
