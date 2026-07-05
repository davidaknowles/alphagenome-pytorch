#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/low_vram/$(date +%Y%m%d_%H%M%S)_parity_driver}"
OG_WEIGHTS="${OG_WEIGHTS:-/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"

mkdir -p "$OUTPUT_ROOT" logs/low_vram

export ROOT PYTHON OUTPUT_ROOT OG_WEIGHTS
export CHROM="${CHROM:-chr9}"
export HEAD="${HEAD:-atac}"
export MAX_WINDOWS="${MAX_WINDOWS:-}"

DRIVER_JOB=$(sbatch scripts/slurm_low_vram_parity_driver.sbatch | awk '{print $4}')

echo "driver_job_id=$DRIVER_JOB"
echo "output_root=$OUTPUT_ROOT"
echo "og_weights=$OG_WEIGHTS"
