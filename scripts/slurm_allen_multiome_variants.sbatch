#!/bin/bash
#SBATCH -J ag_allen_vep
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

: "${PRETRAINED_WEIGHTS:=/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
python scripts/score_singlebrain_finemap.py \
  --checkpoint "${CHECKPOINT:-finetuning_output/allen_multiome_v0/run/best_model.delta.pth}" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --fasta "${FASTA:-/gpfs/commons/groups/knowles_lab/index/hg38/hg38.fa}" \
  --benchmark-variants "${BENCHMARK_VARIANTS:-finetuning_output/allen_multiome_v0/run/variant_effects/benchmark_variants.tsv.gz}" \
  --output-dir "${OUTPUT_DIR:-finetuning_output/allen_multiome_v0/run/variant_effects}"
