#!/bin/bash
#SBATCH --job-name=train-bert
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

module load python/3.11.7-cx
source .venv/bin/activate
export HF_HUB_OFFLINE=1   # bert-base-uncased is pre-downloaded on the login node
python -m models.custom_model.train_bert /scratch/$USER/cs747/bert_checkpoints
