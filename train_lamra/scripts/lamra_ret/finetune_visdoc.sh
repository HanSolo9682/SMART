NNODES=1                   # Set to 2 for multi-node training
NODE_RANK=0                # 0 on the first node, 1 on the second node
MASTER_ADDR=127.0.0.1      # For multi-node: set to IP of node 0
MASTER_PORT=29509

NUM_GPUS=1                # Number of GPUs per node
NPROC_PER_NODE=$NUM_GPUS

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=qwen3-vl-2b
QUERY_DATA_PATH=/XXX/colpali_extracted_data/query.jsonl
CAND_POOL_PATH=/XXX/colpali_extracted_data/cand_pool.jsonl
INSTRUCTIONS_PATH=query_instructions.tsv
MODEL_LOCAL_PATH=Qwen/Qwen3-VL-2B-Instruct
IMAGE_PATH_PREFIX=/XXX/colpali_extracted_data/images/

TRAIN_VISION_ENCODER=False                              
USE_VISION_LORA=False                                  
TRAIN_VISION_PROJECTOR=False

LATE_METHOD=hybrid   # should be disabled, late_only, or hybrid

USE_LORA=True                                           
Q_LORA=False                                           
LORA_R=128                                                
LORA_ALPHA=256 #32                                           
RUN_ID=${MODEL_ID}_visdoc_baseline_convert_1e-4

DS_STAGE=zero2                                          
PER_DEVICE_BATCH_SIZE=64                            
GRAD_ACCUM=1                                            
NUM_EPOCHS=1                                         

LR=1e-4                                         
MODEL_MAX_LEN=1024

# Construct distributed args
DISTRIBUTED_ARGS="
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}
"


torchrun $DISTRIBUTED_ARGS train/train_mbeir.py \
    --run_name $RUN_ID \
    --model_id $MODEL_ID \
    --query_data_path $QUERY_DATA_PATH \
    --cand_pool_path $CAND_POOL_PATH \
    --instructions_path $INSTRUCTIONS_PATH \
    --output_dir ./checkpoints/$RUN_ID \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "steps" \
    --save_steps 116 \
    --save_total_limit 1000 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --image_path_prefix $IMAGE_PATH_PREFIX \
    --model_local_path $MODEL_LOCAL_PATH \
    --late_method $LATE_METHOD
