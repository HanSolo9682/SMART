#!/usr/bin/env bash
# Late-interaction MMEB-V2 embedding evaluation launcher.
# This is a separate path from eval_embedding.sh and does not modify the original evaluator.

echo "==> Environment"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# ==============================================================================
# Multi-node Configuration
# ==============================================================================
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-2277}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"

echo "==> Distributed Evaluation Configuration"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""

# ==============================================================================
# GPU Configuration
# ==============================================================================
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
else
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "Using $GPU_COUNT GPUs per node: $CUDA_VISIBLE_DEVICES"
echo "Total GPUs across all nodes: $((GPU_COUNT * WORLD_SIZE))"
echo ""

# ==============================================================================
# Model and Late-Interaction Configuration
# ==============================================================================
if [ -z "$1" ]; then
    echo "Error: Model path is required"
    echo "Usage: $0 <model_path>"
    echo "Example: $0 Qwen/Qwen3-VL-Embedding-2B"
    echo ""
    echo "Late-interaction environment variables:"
    echo "  ENABLE_LATE_INTERACTION   true/false, default true"
    echo "  LAMBDA_ANCHOR             anchor score weight, default 1.0"
    echo "  LAMBDA_LATE               MaxSim score weight, default 1.0"
    echo "  LATE_QUERY_CHUNK_SIZE     queries per MaxSim chunk, default 4"
    echo "  LATE_CANDIDATE_CHUNK_SIZE candidates per MaxSim chunk, default 64"
    echo ""
    echo "Example:"
    echo "  ENABLE_LATE_INTERACTION=true LAMBDA_ANCHOR=1.0 LAMBDA_LATE=0.2 $0 Qwen/Qwen3-VL-Embedding-2B"
    exit 1
fi

MODEL_NAME="$1"
MODEL_BASENAME=$(basename "$MODEL_NAME")

BATCH_SIZE="${BATCH_SIZE:-32}"
MODALITIES=("vis")
DATA_BASEDIR=/XXX/MMEB-V2
OUTPUT_BASEDIR=/XXX/results_8b_baseline_vis

ENABLE_LATE_INTERACTION="${ENABLE_LATE_INTERACTION:-false}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}"
LAMBDA_LATE="${LAMBDA_LATE:-0.0}"
LATE_QUERY_CHUNK_SIZE="${LATE_QUERY_CHUNK_SIZE:-32}"
LATE_CANDIDATE_CHUNK_SIZE="${LATE_CANDIDATE_CHUNK_SIZE:-128}"

BASE_OUTPUT_PATH="$OUTPUT_BASEDIR/$MODEL_BASENAME"

echo "================================================="
echo "Processing Model: $MODEL_NAME"
echo "Output Base: $BASE_OUTPUT_PATH"
echo "Late Interaction: $ENABLE_LATE_INTERACTION"
echo "lambda_anchor: $LAMBDA_ANCHOR"
echo "lambda_late: $LAMBDA_LATE"
echo "late_query_chunk_size: $LATE_QUERY_CHUNK_SIZE"
echo "late_candidate_chunk_size: $LATE_CANDIDATE_CHUNK_SIZE"
echo "================================================="
echo ""

# ==============================================================================
# Main Execution Loop
# ==============================================================================
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
        -m src.evaluation.mmeb_v2.eval_embedding_late_interaction \
        --normalize true \
        --per_device_eval_batch_size $BATCH_SIZE \
        --model_name_or_path \"$MODEL_NAME\" \
        --dataset_config \"$DATA_CONFIG_PATH\" \
        --encode_output_path \"$OUTPUT_PATH\" \
        --data_basedir \"$DATA_BASEDIR\" \
        --enable_late_interaction $ENABLE_LATE_INTERACTION \
        --lambda_anchor $LAMBDA_ANCHOR \
        --lambda_late $LAMBDA_LATE \
        --late_query_chunk_size $LATE_QUERY_CHUNK_SIZE \
        --late_candidate_chunk_size $LATE_CANDIDATE_CHUNK_SIZE"

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
