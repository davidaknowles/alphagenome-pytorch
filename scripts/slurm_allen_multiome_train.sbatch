#!/bin/bash
#SBATCH -J ag_allen_train
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${PRETRAINED_WEIGHTS:=/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
python scripts/finetune_allen_multiome.py \
  --manifest "${MANIFEST:-finetuning_output/allen_multiome/manifest.json}" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --output-dir "${OUTPUT_DIR:-finetuning_output/allen_multiome/run}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --gradient-checkpointing
