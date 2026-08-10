#!/bin/bash
#SBATCH --job-name=pt-session
#SBATCH --partition=critical
#SBATCH --account=tukw-critical
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

# One allocation, both remaining diagnostics, in the order that matters:
#   (2) exonerate or implicate the shared pipeline with a GPT baseline on PTB;
#   (1) bisect the scale gap on a synthetic stream where the answer is known.
# Step 2 runs first: until the loop, the data and the metric are cleared, a bisection could
# be measuring somebody else's bug.
set -euo pipefail
cd /public/home/belyack/work/pt

PY=/public/software/anaconda3/bin/python
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA — refusing to train on CPU'; \
        print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo; echo "=============== STEP 2: GPT baseline on the identical pipeline ==============="
$PY -m experiments.exp1_language_modeling.run_pilot --model gpt \
    --n-embd 160 --n-layer 4 --n-head 4 --steps 6000 --eval-every 500 \
    --save-ckpt --tag base_gpt

echo; echo "--- Looped Transformer, same budget (Experiment 2's control, free here) ---"
$PY -m experiments.exp1_language_modeling.run_pilot --model looped \
    --n-embd 160 --n-layer 4 --n-head 4 --steps 6000 --eval-every 500 \
    --save-ckpt --tag base_looped

echo; echo "--- PT control at identical settings, MFVI readout ---"
$PY -m experiments.exp1_language_modeling.run_pilot --model pt --readout mfvi \
    --d 256 --h 8 --rank 64 --n-iters 3 --steps 6000 --eval-every 500 \
    --save-ckpt --tag base_pt

echo; echo "=============== STEP 1: scale bisection on a synthetic chain ==============="
$PY -m experiments.exp1_language_modeling.run_scale --steps 3000

echo; echo "=============== prefix ablation on every checkpoint written ==============="
for ck in checkpoints/base_gpt_gpt.pt checkpoints/base_looped_looped.pt checkpoints/base_pt_pt.pt; do
  [ -f "$ck" ] && $PY -m experiments.exp1_language_modeling.ablate_prefix "$ck" --batches 8 || true
done

echo; echo "ALL DONE"
