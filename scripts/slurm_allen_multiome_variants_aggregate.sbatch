#!/bin/bash
#SBATCH -J ag_allen_vep_agg
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python scripts/aggregate_singlebrain_finemap.py \
  --input-dir "${INPUT_DIR:-finetuning_output/allen_multiome_v0/run/variant_effects}" \
  --benchmark-variants "${BENCHMARK_VARIANTS:-finetuning_output/allen_multiome_v0/run/variant_effects/benchmark_variants.tsv.gz}" \
  --num-shards "${NUM_SHARDS:-8}"
