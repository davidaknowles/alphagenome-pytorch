#!/bin/bash
#SBATCH -J ag_allen_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

pretrained="${PRETRAINED_WEIGHTS:-/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
root="${OUTPUT_ROOT:-finetuning_output/allen_multiome_smoke}"

pytest -q \
  tests/unit/test_finetuning_datasets.py \
  tests/unit/test_gene_annotation.py \
  tests/unit/test_finetune_multitask.py \
  tests/unit/test_adapters.py \
  tests/unit/test_metrics.py

python scripts/prepare_allen_multiome.py \
  --output-dir "$root" \
  --limit-train 2 \
  --limit-valid 1 \
  --limit-test 1

python scripts/finetune_allen_multiome.py \
  --manifest "$root/manifest.json" \
  --pretrained-weights "$pretrained" \
  --output-dir "$root/run" \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --track-means-samples 2 \
  --num-workers 0 \
  --gradient-checkpointing

python scripts/evaluate_allen_multiome.py \
  --manifest "$root/manifest.json" \
  --checkpoint "$root/run/best_model.delta.pth" \
  --pretrained-weights "$pretrained" \
  --output "$root/run/heldout_metrics.json" \
  --batch-size 1 \
  --num-workers 0

python scripts/score_singlebrain_finemap.py \
  --checkpoint "$root/run/best_model.delta.pth" \
  --pretrained-weights "$pretrained" \
  --fasta /gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa \
  --gtf /gpfs/commons/groups/knowles_lab/index/hg38/gencode.v49.basic.annotation.gtf.gz \
  --output-dir "$root/run/variant_effects" \
  --limit-pairs 2
