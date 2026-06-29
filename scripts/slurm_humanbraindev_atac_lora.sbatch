#!/bin/bash
#SBATCH -J ag_hbd_lora_pt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 24:00:00
#SBATCH -o logs/humanbraindev_atac_lora/%x-%j.out
#SBATCH -e logs/humanbraindev_atac_lora/%x-%j.err

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/alphagenome-pytorch
mkdir -p logs/humanbraindev_atac_lora finetuning_output/humanbraindev_atac_lora

source ~/venv/torch/bin/activate
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-daknowles}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/pycache-${USER:-daknowles}-${SLURM_JOB_ID:-manual}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${PRETRAINED_WEIGHTS:-}" ]]; then
  echo "PRETRAINED_WEIGHTS is required, e.g. PRETRAINED_WEIGHTS=/path/to/alphagenome.pth sbatch $0" >&2
  exit 2
fi

cmd=(
  python scripts/run_humanbraindev_lora_finetune.py
  --bigwig-dir "${BIGWIG_DIR:-/gpfs/commons/home/daknowles/knowles_lab/data/multiome/humanbraindev/bigwigs}"
  --fasta-path "${FASTA_PATH:-/gpfs/commons/home/daknowles/knowles_lab/index/hg38/hg38.fa}"
  --pretrained-weights "${PRETRAINED_WEIGHTS}"
  --output-dir "${OUTPUT_DIR:-finetuning_output/humanbraindev_atac_lora}"
  --split-source "${SPLIT_SOURCE:-chromosome}"
  --window-size "${WINDOW_SIZE:-131072}"
  --stride "${STRIDE:-131072}"
  --batch-size "${BATCH_SIZE:-7}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --epochs "${EPOCHS:-5}"
  --lr "${LR:-1e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.1}"
  --warmup-steps "${WARMUP_STEPS:-500}"
  --lora-rank "${LORA_RANK:-8}"
  --lora-alpha "${LORA_ALPHA:-16}"
  --lora-targets "${LORA_TARGETS:-q_proj,v_proj}"
  --dtype "${DTYPE:-bfloat16}"
  --fp8-recipe "${FP8_RECIPE:-tensorwise}"
  --fp8-min-feature-multiple "${FP8_MIN_FEATURE_MULTIPLE:-16}"
  --fp8-skip-name-patterns "${FP8_SKIP_NAME_PATTERNS:-heads,original_layer,lora_,locon_,ia3,adapter}"
  --fp4-min-feature-multiple "${FP4_MIN_FEATURE_MULTIPLE:-16}"
  --fp4-mode "${FP4_MODE:-weight-only}"
  --fp4-skip-name-patterns "${FP4_SKIP_NAME_PATTERNS:-heads,lora_,locon_,ia3,adapter}"
  --track-means-samples "${TRACK_MEANS_SAMPLES:-256}"
  --num-workers "${NUM_WORKERS:-4}"
  --max-io-workers "${MAX_IO_WORKERS:-16}"
  --profile-batches "${PROFILE_BATCHES:-0}"
)

[[ -n "${RUN_NAME:-}" ]] && cmd+=(--run-name "$RUN_NAME")
[[ -n "${LIMIT_TRAIN:-}" ]] && cmd+=(--limit-train "$LIMIT_TRAIN")
[[ -n "${LIMIT_VALID:-}" ]] && cmd+=(--limit-valid "$LIMIT_VALID")
[[ -n "${LIMIT_TEST:-}" ]] && cmd+=(--limit-test "$LIMIT_TEST")
[[ -n "${LIMIT_BIGWIGS:-}" ]] && cmd+=(--limit-bigwigs "$LIMIT_BIGWIGS")
[[ -n "${SEED:-}" ]] && cmd+=(--seed "$SEED")
[[ -n "${GRADIENT_CHECKPOINTING:-}" ]] && cmd+=(--gradient-checkpointing)
[[ -n "${CACHE_GENOME:-}" ]] && cmd+=(--cache-genome)
[[ -n "${CACHE_SIGNALS:-}" ]] && cmd+=(--cache-signals)
[[ -n "${NO_FULL_CHECKPOINT:-}" ]] && cmd+=(--no-full-checkpoint)
[[ -n "${NO_SAVE_CHECKPOINTS:-}" ]] && cmd+=(--no-save-checkpoints)
[[ -n "${WANDB:-}" ]] && cmd+=(--wandb)
[[ -n "${WANDB_PROJECT:-}" ]] && cmd+=(--wandb-project "$WANDB_PROJECT")
[[ -n "${WANDB_ENTITY:-}" ]] && cmd+=(--wandb-entity "$WANDB_ENTITY")

"${cmd[@]}"
