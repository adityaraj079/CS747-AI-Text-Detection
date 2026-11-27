#!/bin/bash
#SBATCH --job-name=eval-external
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
EXT=/scratch/$USER/cs747/external
python -m models.custom_model.evaluate_external \
  --corpus /scratch/$USER/cs747/CS747data.csv \
  --ghostbuster $EXT/ghostbuster-data/essay \
  --hc3 $EXT/all.jsonl \
  --wiki $EXT/GPT-wiki-intro.csv.zip \
  --out /scratch/$USER/cs747/external_custom_model.json
