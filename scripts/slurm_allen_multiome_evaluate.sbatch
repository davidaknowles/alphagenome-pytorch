#!/bin/bash
#SBATCH -J ag_allen_eval
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

: "${PRETRAINED_WEIGHTS:=/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
python scripts/evaluate_allen_multiome.py \
  --manifest "${MANIFEST:-finetuning_output/allen_multiome_v0/manifest.json}" \
  --checkpoint "${CHECKPOINT:-finetuning_output/allen_multiome_v0/run/best_model.delta.pth}" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --output "${OUTPUT:-finetuning_output/allen_multiome_v0/run/heldout_metrics.json}"
