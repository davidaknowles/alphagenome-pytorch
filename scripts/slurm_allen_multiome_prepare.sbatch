#!/bin/bash
#SBATCH -J ag_allen_prepare
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/allen_multiome/%x-%j.out
#SBATCH -e logs/allen_multiome/%x-%j.err

set -euo pipefail
cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/allen_multiome
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"

cmd=(
  python scripts/prepare_allen_multiome.py
  --output-dir "${OUTPUT_DIR:-finetuning_output/allen_multiome}"
)
[[ -n "${LIMIT_TRAIN:-}" ]] && cmd+=(--limit-train "$LIMIT_TRAIN")
[[ -n "${LIMIT_VALID:-}" ]] && cmd+=(--limit-valid "$LIMIT_VALID")
[[ -n "${LIMIT_TEST:-}" ]] && cmd+=(--limit-test "$LIMIT_TEST")
[[ -n "${REFERENCE_DIR:-}" ]] && cmd+=(--reference-dir "$REFERENCE_DIR")
"${cmd[@]}"
