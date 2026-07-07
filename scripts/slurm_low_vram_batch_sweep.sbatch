#!/usr/bin/env bash
#SBATCH -J ag-lv-bsweep
#SBATCH -p gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH -o logs/low_vram/%x_%j.out
#SBATCH -e logs/low_vram/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/low_vram_batch_sweep/$(date +%Y%m%d_%H%M%S)}"
OG_WEIGHTS="${OG_WEIGHTS:-/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
FASTA_PATH="${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}"

mkdir -p "$OUTPUT_ROOT" logs/low_vram

args=(
  scripts/benchmark_low_vram_batch_sweep.py
  --output-root "$OUTPUT_ROOT"
  --python "$PYTHON"
  --og-weights "$OG_WEIGHTS"
  --fasta-path "$FASTA_PATH"
  --chrom "${CHROM:-chr9}"
  --head "${HEAD:-atac}"
  --vram-limit-gib "${VRAM_LIMIT_GIB:-48}"
  --gpu-sample-interval "${GPU_SAMPLE_INTERVAL:-0.25}"
  --final-examples-131kb "${FINAL_EXAMPLES_131KB:-128}"
  --final-examples-1mb "${FINAL_EXAMPLES_1MB:-32}"
)
if [[ -n "${STRATEGIES:-}" ]]; then
  args+=(--strategies "$STRATEGIES")
fi
if [[ -n "${BATCHES_131KB:-}" ]]; then
  args+=(--batches-131kb "$BATCHES_131KB")
fi
if [[ -n "${BATCHES_1MB:-}" ]]; then
  args+=(--batches-1mb "$BATCHES_1MB")
fi

"$PYTHON" "${args[@]}"

echo "completed output_root=$OUTPUT_ROOT"
