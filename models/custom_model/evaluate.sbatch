#!/bin/bash
#SBATCH --job-name=eval-custom
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

module load python/3.11.7-cx
source .venv/bin/activate
export HF_HUB_OFFLINE=1   # BERT weights are local in artifacts/bert on Hopper
for split in standard_test unseen_test; do
  python scripts/evaluate_registered_model.py --model-id custom_model --dataset-type $split
done
ls -l results/metrics/standardized/custom_model_*
