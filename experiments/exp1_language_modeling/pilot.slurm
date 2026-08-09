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
cd /public/home/belyack/work/pt

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/public/software/anaconda3/bin/python -c \
  "import torch; assert torch.cuda.is_available(), 'no CUDA — refusing to train on CPU'; \
   print('torch', torch.__version__, torch.cuda.get_device_name(0))"

/public/software/anaconda3/bin/python -m experiments.exp1_language_modeling.run_pilot "$@"
