#!/bin/bash
#SBATCH --job-name=pt-attack
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd /public/home/belyack/work/pt
echo "node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "nvidia-smi failed"
/public/software/anaconda3/bin/python -c \
  "import torch; assert torch.cuda.is_available(), 'no CUDA on this node — refusing to train on CPU'; \
   print('torch', torch.__version__, torch.cuda.get_device_name(0))"
/public/software/anaconda3/bin/python -m experiments.exp1_language_modeling.ptb_attack "$@"
