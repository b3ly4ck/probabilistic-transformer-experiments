#!/bin/bash
#SBATCH --job-name=pt-calib
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd /public/home/belyack/work/pt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
/public/software/anaconda3/bin/python experiments/exp1_language_modeling/run_calibration.py "$@"
