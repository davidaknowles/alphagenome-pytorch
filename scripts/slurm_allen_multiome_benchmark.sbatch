#!/bin/bash
#SBATCH -J ag_allen_cohort
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

python scripts/prepare_singlebrain_finemap_benchmark.py \
  --gtf "${GTF:-/gpfs/commons/groups/knowles_lab/index/hg38/gencode.v49.basic.annotation.gtf.gz}" \
  --output "${OUTPUT:-finetuning_output/allen_multiome_v0/run/variant_effects/benchmark_variants.tsv.gz}"
