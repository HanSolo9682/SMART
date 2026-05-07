#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   GPUS=0,1 TRAIN_DATA=/path/to/colpali_train_set bash training_visdoc_2b.sh
#   DEBUG_PRINT_BATCH=1 DEBUG_PRINT_BATCH_EXIT=1 GPUS=0 bash training_visdoc_2b.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

count_csv_items() {
  local value="$1"
  awk -F',' '{print NF}' <<< "${value}"
}

TRAIN_DATA=${TRAIN_DATA:-/XXX/dataset/Visdoc/colpali_train_set}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs/qwen3_vl_visdoc_2b_only_residual}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-Embedding-2B}
CACHE_DIR=${CACHE_DIR:-/XXX/.cache/hf_datasets}
IMAGE_CACHE_DIR=${IMAGE_CACHE_DIR:-/XXX/.cache/visdoc_page_images}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# SELECTED_LAYERS overrides USE_LAST_N_LAYERS
USE_LAST_N_LAYERS=${USE_LAST_N_LAYERS:-1}
SELECTED_LAYERS=${SELECTED_LAYERS:-}
SELECTED_LAYERS=${SELECTED_LAYERS// /}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
GPUS=${GPUS// /}
VISIBLE_GPU_COUNT=$(count_csv_items "${GPUS}")
NUM_PROCESSES=${NUM_PROCESSES:-${VISIBLE_GPU_COUNT}}

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-qwen3_vl_visdoc_experiment}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-offline}
WANDB_RUN_NAME_PREFIX=${WANDB_RUN_NAME_PREFIX:-qwen3_vl_visdoc}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-0}
SAVE_EVERY_EPOCH=${SAVE_EVERY_EPOCH:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
PIN_MEMORY=${PIN_MEMORY:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
QUERY_ENCODE_BATCH_SIZE=${QUERY_ENCODE_BATCH_SIZE:-}
DOC_ENCODE_BATCH_SIZE=${DOC_ENCODE_BATCH_SIZE:-64}
MIN_PIXELS=${MIN_PIXELS:-4096}
MAX_PIXELS=${MAX_PIXELS:-1843200}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-100}
LAMBDA_ANCHOR=${LAMBDA_ANCHOR:-1.0}
LAMBDA_RESIDUAL=${LAMBDA_RESIDUAL:-1.0}
HYBRID_LOSS_WEIGHT=${HYBRID_LOSS_WEIGHT:-0.0}
RESIDUAL_LOSS_WEIGHT_START=${RESIDUAL_LOSS_WEIGHT_START:-1.0}
RESIDUAL_LOSS_WEIGHT_END=${RESIDUAL_LOSS_WEIGHT_END:-1.0}

TRACE_EVERY=${TRACE_EVERY:-10}
SLOW_STEP_SECONDS=${SLOW_STEP_SECONDS:-10}
HOST_MEMORY_TRIM_EVERY=${HOST_MEMORY_TRIM_EVERY:-10}
HF_PRELOAD_MODEL=${HF_PRELOAD_MODEL:-1}
FORCE_TRANSFORMERS_OFFLINE=${FORCE_TRANSFORMERS_OFFLINE:-1}
DEBUG_DUMP_NUM_BATCHES=${DEBUG_DUMP_NUM_BATCHES:-0}
DEBUG_DUMP_SAMPLES=${DEBUG_DUMP_SAMPLES:-3}
DEBUG_DUMP_INCLUDE_NEGATIVES=${DEBUG_DUMP_INCLUDE_NEGATIVES:-0}
DEBUG_PRINT_BATCH=${DEBUG_PRINT_BATCH:-0}
DEBUG_PRINT_BATCH_EXIT=${DEBUG_PRINT_BATCH_EXIT:-0}
DEBUG_TOKEN_FILTER_MAX_TOKENS=${DEBUG_TOKEN_FILTER_MAX_TOKENS:-160}
PYTHON_BIN=${PYTHON_BIN:-python}
ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "[ERROR] NUM_PROCESSES must be >= 1, got ${NUM_PROCESSES}" >&2
  exit 1
fi
if [[ "${NUM_PROCESSES}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "[ERROR] NUM_PROCESSES=${NUM_PROCESSES} exceeds visible GPU count ${VISIBLE_GPU_COUNT}" >&2
  exit 1
fi
if [[ ! -d "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA does not exist: ${TRAIN_DATA}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
  export HF_DATASETS_CACHE="${CACHE_DIR}"
fi
if [[ -n "${IMAGE_CACHE_DIR}" ]]; then
  mkdir -p "${IMAGE_CACHE_DIR}"
fi

# Pre-download model once to avoid Hub 429 errors during multi-process launch.
if [[ "${HF_PRELOAD_MODEL}" == "1" ]] && [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[INFO] Preloading model to local HF cache: ${MODEL_NAME_OR_PATH}"
  MODEL_REPO_ID="${MODEL_NAME_OR_PATH}" "${PYTHON_BIN}" - <<'PY'
from huggingface_hub import snapshot_download
import os
repo_id = os.environ["MODEL_REPO_ID"]
snapshot_download(repo_id=repo_id, resume_download=True)
print(f"[INFO] Model cached: {repo_id}")
PY
fi

if [[ "${FORCE_TRANSFORMERS_OFFLINE}" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  echo "[INFO] TRANSFORMERS_OFFLINE=1 (model loads from local cache)"
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ -n "${SELECTED_LAYERS}" ]]; then
  LAYER_TAG="selected_$(tr ',' '-' <<< "${SELECTED_LAYERS}")"
else
  LAYER_TAG="last_${USE_LAST_N_LAYERS}"
fi
RUN_DIR="${OUTPUT_ROOT}/${LAYER_TAG}"
RUN_WANDB_NAME="${WANDB_RUN_NAME:-${WANDB_RUN_NAME_PREFIX}_${LAYER_TAG}}"
mkdir -p "${RUN_DIR}"

TRAIN_ARGS=(
  --dataset_type visdoc
  --train_data "${TRAIN_DATA}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --image_cache_dir "${IMAGE_CACHE_DIR}"
  --output_dir "${RUN_DIR}"
  --anchor_dim 2048
  --residual_dim 2048
  --temperature 0.03
  --lambda_anchor "${LAMBDA_ANCHOR}"
  --lambda_residual "${LAMBDA_RESIDUAL}"
  --hybrid_loss_weight "${HYBRID_LOSS_WEIGHT}"
  --residual_loss_weight_start "${RESIDUAL_LOSS_WEIGHT_START}"
  --residual_loss_weight_end "${RESIDUAL_LOSS_WEIGHT_END}"
  --num_hard_negatives 0
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps 1
  --learning_rate 1e-3
  --weight_decay 0.01
  --warmup_ratio 0.03
  --num_epochs 2
  --val_ratio 0.02
  --max_val_samples "${MAX_VAL_SAMPLES}"
  --max_length "${MAX_LENGTH}"
  --min_pixels "${MIN_PIXELS}"
  --max_pixels "${MAX_PIXELS}"
  --attn_implementation "${ATTN_IMPL}"
  --show_progress_bar
  --wandb_project "${WANDB_PROJECT}"
  --wandb_mode "${WANDB_MODE}"
  --wandb_run_name "${RUN_WANDB_NAME}"
  --num_workers "${NUM_WORKERS}"
  --log_every 50
  --trace_every "${TRACE_EVERY}"
  --slow_step_seconds "${SLOW_STEP_SECONDS}"
  --host_memory_trim_every "${HOST_MEMORY_TRIM_EVERY}"
  --save_every_steps "${SAVE_EVERY_STEPS}"
  # --debug_token_filter_max_tokens "${DEBUG_TOKEN_FILTER_MAX_TOKENS}"
)

if [[ "${PIN_MEMORY}" == "1" ]]; then
  TRAIN_ARGS+=(--pin_memory)
fi
if [[ "${PERSISTENT_WORKERS}" == "1" ]] && [[ "${NUM_WORKERS}" -gt 0 ]]; then
  TRAIN_ARGS+=(--persistent_workers)
fi

if [[ -n "${CACHE_DIR}" ]]; then
  TRAIN_ARGS+=(--cache_dir "${CACHE_DIR}")
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  TRAIN_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${QUERY_ENCODE_BATCH_SIZE}" ]]; then
  TRAIN_ARGS+=(--query_encode_batch_size "${QUERY_ENCODE_BATCH_SIZE}")
fi
if [[ -n "${DOC_ENCODE_BATCH_SIZE}" ]]; then
  TRAIN_ARGS+=(--doc_encode_batch_size "${DOC_ENCODE_BATCH_SIZE}")
fi

if [[ -n "${SELECTED_LAYERS}" ]]; then
  TRAIN_ARGS+=(--selected_layers "${SELECTED_LAYERS}")
else
  TRAIN_ARGS+=(--use_last_n_layers "${USE_LAST_N_LAYERS}")
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  TRAIN_ARGS+=(--use_wandb)
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  TRAIN_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ "${SAVE_EVERY_EPOCH}" == "1" ]]; then
  TRAIN_ARGS+=(--save_every_epoch)
fi
if [[ "${DEBUG_PRINT_BATCH}" == "1" ]]; then
  TRAIN_ARGS+=(--debug_print_batch)
fi
if [[ "${DEBUG_PRINT_BATCH_EXIT}" == "1" ]]; then
  TRAIN_ARGS+=(--debug_print_batch_exit)
fi
if (( DEBUG_DUMP_NUM_BATCHES > 0 )); then
  TRAIN_ARGS+=(
    --debug_dump_num_batches "${DEBUG_DUMP_NUM_BATCHES}"
    --debug_dump_samples "${DEBUG_DUMP_SAMPLES}"
    --debug_dump_dir "${RUN_DIR}/debug_dump"
  )
  if [[ "${DEBUG_DUMP_INCLUDE_NEGATIVES}" == "1" ]]; then
    TRAIN_ARGS+=(--debug_dump_include_negatives)
  fi
fi

"${ACCELERATE_BIN}" launch \
  --mixed_precision "${MIXED_PRECISION}" \
  --num_processes "${NUM_PROCESSES}" \
  train.py \
  "${TRAIN_ARGS[@]}"
