#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch}"
cd "$ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/low_vram_batch_sweep/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_ROOT" logs/low_vram
export ROOT OUTPUT_ROOT

JOB_ID=$(sbatch scripts/slurm_low_vram_batch_sweep.sbatch | awk '{print $4}')

echo "job_id=$JOB_ID"
echo "output_root=$OUTPUT_ROOT"
