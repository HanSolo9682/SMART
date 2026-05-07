#!/usr/bin/env bash
# Adapter late-interaction MMEB-V2 evaluation launcher: linear/residual adapter.

echo "==> Environment"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-2278}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

echo "==> Distributed Evaluation Configuration"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
else
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "Using $GPU_COUNT GPUs per node: $CUDA_VISIBLE_DEVICES"
echo "Total GPUs across all nodes: $((GPU_COUNT * WORLD_SIZE))"
echo ""

if [ -z "$1" ]; then
    echo "Error: Model path is required"
    echo "Usage: $0 <model_path>"
    echo "Example: $0 Qwen/Qwen3-VL-Embedding-2B"
    echo ""
    echo "Adapter late-interaction environment variables:"
    echo "  ADAPTER_TYPE               default residual"
    echo "  ADAPTER_CHECKPOINT_DIR     directory containing adapter.pt, default residual best_adapter"
    echo "  SELECTED_LAYER             hidden-state layer for adapter input, default -1"
    echo "  ENABLE_LATE_INTERACTION    true/false, default true"
    echo "  LAMBDA_ANCHOR              anchor score weight, default 1.0"
    echo "  LAMBDA_LATE                adapted MaxSim score weight, default 1.0"
    echo "  LATE_QUERY_CHUNK_SIZE      queries per MaxSim chunk, default 8"
    echo "  LATE_CANDIDATE_CHUNK_SIZE  candidates per MaxSim chunk, default 64"
    echo "  SANITY_CHECK               true/false, default false"
    echo "  OUTPUT_DIR                 full output directory override"
    exit 1
fi

MODEL_NAME="$1"
MODEL_BASENAME=$(basename "$MODEL_NAME")

BATCH_SIZE="${BATCH_SIZE:-16}"
MODALITIES=("video")
DATA_BASEDIR="${DATA_BASEDIR:-/XXX/MMEB-V2}"
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-/XXX/Qwen_3_vl_weight_all_hidden_only_visdoc/outputs/eval_adapter_late_visdoc_8b_epoch1}"

ADAPTER_TYPE="${ADAPTER_TYPE:-residual}"
ADAPTER_CHECKPOINT_DIR="${ADAPTER_CHECKPOINT_DIR:-/XXX/Qwen_3_vl_weight_all_hidden_only_visdoc/outputs/qwen3_vl_visdoc_8b_only_residual/last_1/epoch_1}"
SELECTED_LAYER="${SELECTED_LAYER:--1}"

ENABLE_LATE_INTERACTION="${ENABLE_LATE_INTERACTION:-true}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}"
LAMBDA_LATE="${LAMBDA_LATE:-1.0}"
LATE_QUERY_CHUNK_SIZE="${LATE_QUERY_CHUNK_SIZE:-16}"
LATE_CANDIDATE_CHUNK_SIZE="${LATE_CANDIDATE_CHUNK_SIZE:-64}"
SANITY_CHECK="${SANITY_CHECK:-false}"
SANITY_CHECK_NUM_EXAMPLES="${SANITY_CHECK_NUM_EXAMPLES:-2}"

LAYER_TAG=$(echo "$SELECTED_LAYER" | sed -e 's/-/n/g')
RUN_NAME="${RUN_NAME:-selected_${LAYER_TAG}_anchor_${LAMBDA_ANCHOR}_late_${LAMBDA_LATE}}"
BASE_OUTPUT_PATH="${OUTPUT_DIR:-$OUTPUT_BASEDIR/$ADAPTER_TYPE/$MODEL_BASENAME/$RUN_NAME}"

echo "================================================="
echo "Processing Model: $MODEL_NAME"
echo "Adapter Type: $ADAPTER_TYPE"
echo "Adapter Checkpoint Dir: $ADAPTER_CHECKPOINT_DIR"
echo "Selected Layer: $SELECTED_LAYER"
echo "Run Name: $RUN_NAME"
echo "Output Base: $BASE_OUTPUT_PATH"
echo "Late Interaction: $ENABLE_LATE_INTERACTION"
echo "lambda_anchor: $LAMBDA_ANCHOR"
echo "lambda_late: $LAMBDA_LATE"
echo "late_query_chunk_size: $LATE_QUERY_CHUNK_SIZE"
echo "late_candidate_chunk_size: $LATE_CANDIDATE_CHUNK_SIZE"
echo "sanity_check: $SANITY_CHECK"
echo "================================================="
echo ""

for MODALITY in "${MODALITIES[@]}"; do
    DATA_CONFIG_PATH="scripts/evaluation/mmeb_v2/${MODALITY}.yaml"
    OUTPUT_PATH="$BASE_OUTPUT_PATH/$MODALITY/"

    echo "-------------------------------------------------"
    echo "  - Modality: $MODALITY"
    echo "  - Output Path: $OUTPUT_PATH"

    if [ "$RANK" -eq 0 ]; then
        mkdir -p "$OUTPUT_PATH"
    fi

    sleep 2

    cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nproc_per_node=$GPU_COUNT \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --max_restarts=0 \
        -m src.evaluation.mmeb_v2.eval_embedding_adapter_late_interaction \
        --normalize true \
        --per_device_eval_batch_size $BATCH_SIZE \
        --model_name_or_path \"$MODEL_NAME\" \
        --dataset_config \"$DATA_CONFIG_PATH\" \
        --output_dir \"$OUTPUT_PATH\" \
        --encode_output_path \"$OUTPUT_PATH\" \
        --data_basedir \"$DATA_BASEDIR\" \
        --enable_late_interaction $ENABLE_LATE_INTERACTION \
        --lambda_anchor $LAMBDA_ANCHOR \
        --lambda_late $LAMBDA_LATE \
        --late_query_chunk_size $LATE_QUERY_CHUNK_SIZE \
        --late_candidate_chunk_size $LATE_CANDIDATE_CHUNK_SIZE \
        --adapter_type \"$ADAPTER_TYPE\" \
        --adapter_checkpoint_dir \"$ADAPTER_CHECKPOINT_DIR\" \
        --selected_layer $SELECTED_LAYER \
        --sanity_check $SANITY_CHECK \
        --sanity_check_num_examples $SANITY_CHECK_NUM_EXAMPLES"

    echo "  - Executing command on node $RANK..."
    eval "$cmd"

    if [ $? -eq 0 ]; then
        echo "  - Done on node $RANK."
    else
        echo "  - Failed on node $RANK."
        exit 1
    fi
    echo "-------------------------------------------------"
    echo ""
done

if [ "$SANITY_CHECK" = "true" ]; then
    echo "Sanity check completed; skipping result gathering."
    exit 0
fi

if [ "$RANK" -eq 0 ]; then
    echo "All jobs completed on master node."
    echo ""
    echo "================================================="
    echo "Gathering evaluation results..."
    echo "================================================="

    python -m src.evaluation.mmeb_v2.gather_results \
        "$BASE_OUTPUT_PATH" \
        --output_dir "$BASE_OUTPUT_PATH"

    if [ $? -eq 0 ]; then
        echo "Results gathered successfully."
    else
        echo "Failed to gather results."
        exit 1
    fi
else
    echo "All jobs completed on worker node $RANK."
fi
