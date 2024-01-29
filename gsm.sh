#!/usr/bin/env bash
# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/yyu429/ENTER/envs/s5/lib
/home/yyu429/ENTER/envs/s5/bin/python3 -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/llama-7B-cot-8shot \
    --model meta-llama/Llama-2-7b-hf \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --n_shot 8 \
    --use_slow_tokenizer \
    --max_num_examples 200