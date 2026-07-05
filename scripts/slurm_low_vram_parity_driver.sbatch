#!/usr/bin/env bash
#SBATCH -J ag-lv-driver
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH -o logs/low_vram/%x_%j.out
#SBATCH -e logs/low_vram/%x_%j.err

set -euo pipefail

ROOT="${ROOT:-/gpfs/commons/home/daknowles/projects/ag-refactor/alphagenome-pytorch}"
cd "$ROOT"

PYTHON="${PYTHON:-$HOME/venv/torchfix/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/low_vram/$(date +%Y%m%d_%H%M%S)_parity_driver}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$OUTPUT_ROOT/reference_predictions}"
OG_WEIGHTS="${OG_WEIGHTS:-/gpfs/commons/home/daknowles/projects/mpragent/outputs/models/alphagenome/model_all_folds.safetensors}"
CHROM="${CHROM:-chr9}"
HEAD="${HEAD:-atac}"

mkdir -p "$OUTPUT_ROOT" "$REFERENCE_ROOT" logs/low_vram

WINDOWS=(
  "131072:32"
  "1048576:2"
)

STRATEGIES=(
  default
  bf16_params
  triton_conv
  flexattention
  flex_lowres_bias
  no_intermediates
  triton_pool
  fused_embedder
  fused_down0
  fused_embedder_down0
  all_features
)

submit_and_wait() {
  local strategy="$1"
  local window_size="$2"
  local batch_size="$3"
  local write_reference="$4"
  local ref="$REFERENCE_ROOT/${HEAD}_${CHROM}_w${window_size}_b${batch_size}_default.npy"
  local run_dir="$OUTPUT_ROOT/${strategy}_w${window_size}_b${batch_size}"
  local reference_arg=""
  if [[ "$write_reference" == "1" ]]; then
    reference_arg="--write-reference"
  fi
  local max_windows_arg=""
  if [[ -n "${MAX_WINDOWS:-}" ]]; then
    max_windows_arg="--max-windows ${MAX_WINDOWS}"
  fi

  echo "submitting strategy=$strategy window_size=$window_size batch_size=$batch_size write_reference=$write_reference"
  sbatch --wait \
    -J "ag-lv-${strategy:0:10}-${window_size}" \
    -p gpu \
    --gres=gpu:b6k:1 \
    --cpus-per-task=8 \
    --mem=80G \
    --time=12:00:00 \
    -o "logs/low_vram/%x_%j.out" \
    -e "logs/low_vram/%x_%j.err" \
    --wrap "cd '$ROOT' && '$PYTHON' scripts/benchmark_low_vram_parity.py --strategy '$strategy' --reference-predictions '$ref' --output-dir '$run_dir' --og-weights '$OG_WEIGHTS' --window-size '$window_size' --batch-size '$batch_size' --chrom '$CHROM' --head '$HEAD' $reference_arg $max_windows_arg"
}

echo "output_root=$OUTPUT_ROOT"
echo "reference_root=$REFERENCE_ROOT"
echo "og_weights=$OG_WEIGHTS"

for window_spec in "${WINDOWS[@]}"; do
  IFS=: read -r window_size batch_size <<< "$window_spec"
  submit_and_wait default "$window_size" "$batch_size" 1
done

for strategy in "${STRATEGIES[@]}"; do
  for window_spec in "${WINDOWS[@]}"; do
    IFS=: read -r window_size batch_size <<< "$window_spec"
    submit_and_wait "$strategy" "$window_size" "$batch_size" 0
  done
done

"$PYTHON" scripts/collate_low_vram_parity.py "$OUTPUT_ROOT" \
  --output docs/low_vram/og_low_vram_parity_20260704.md

echo "completed output_root=$OUTPUT_ROOT"
