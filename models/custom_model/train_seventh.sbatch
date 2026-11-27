#!/bin/bash
#SBATCH --job-name=train-seventh
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

module load python/3.11.7-cx
source .venv/bin/activate
python -m models.custom_model.train_seventh
