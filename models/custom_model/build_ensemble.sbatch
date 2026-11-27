#!/bin/bash
#SBATCH --job-name=build-ensemble
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

module load python/3.11.7-cx
source .venv/bin/activate
export HF_HUB_OFFLINE=1
python -m models.custom_model.build_ensemble /scratch/$USER/cs747/member_scores
