export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_LEVEL=NVL

MODEL_SIZE=8B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    ./raglab/rag/train_alg/finetune_qlora.py \
    --model_name_or_path ./model/Meta-Llama-3-8B\
    --use_flash_attn \
    --tokenizer_name ./model/Meta-Llama-3-8B \
    --use_slow_tokenizer \
    --train_file ./data/train_data/full_output_1005-remove_special_tokens.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 1 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./model/output_models/Llama3-${MODEL_SIZE}-baseline-adapter/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
    --use_lora \
    --quantization 4bit