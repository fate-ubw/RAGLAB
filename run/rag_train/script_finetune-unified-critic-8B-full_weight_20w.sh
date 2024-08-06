export CUDA_VISIBLE_DEVICES=3,4,5,6
export NCCL_P2P_LEVEL=NVL

MODEL_SIZE=8B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch\
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ./raglab/rag/train_alg/stage3_no_offloading_accelerate.conf \
    ./raglab/rag/train_alg/finetune.py \
    --model_name_or_path ./model/Meta-Llama-3-8B\
    --use_flash_attn \
    --tokenizer_name ./model/Meta-Llama-3-8B \
    --use_slow_tokenizer \
    --train_file ./data/collected_data/train_20w.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./model/output_models/unified-Critic-${MODEL_SIZE}-baseline_20w/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens     
