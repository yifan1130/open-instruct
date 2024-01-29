#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/yyu429/ENTER/envs/s5/lib
MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=16
lr=5e-5
seq_len=1024
max_value=0.4
max_value_final=0.05
num_token=8
init_warmup=500
final_warmup=1000
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training

#accelerate launch \
/home/yyu429/ENTER/envs/s5/bin/python3 -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --train_file data/processed/tulu_v1/gpt4_alpaca_subset/gpt4_alpaca_data.jsonl \
    --max_seq_length $seq_len \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 1000 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $lr \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/llama2-7b_lr${lr}_seq_len${seq_len}_bsz${TOTAL_BATCH_SIZE}_initwp${init_warmup}_finalwp${final_warmup}_maxvalue${max_value}_final${max_value_final} \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --max_value $max_value \
    --num_token $num_token \
    --max_value_final $max_value_final \
    --init_warmup $init_warmup \
    --final_warmup $final_warmup

#python open_instruct/merge_lora.py \
#    --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#    --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_lora/ \
#    --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#    --save_tokenizer
