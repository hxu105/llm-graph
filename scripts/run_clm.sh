#!/bin/bash

dataset_name=${1:-"la"}

# CUDA_VISIBLE_DEVICES=0 torchrun --nnode 1 --nproc_per_node 1 -m train.run_clm \
CUDA_VISIBLE_DEVICES=1 python -m train.run_clm \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_dir ./data/$dataset_name \
    --multimodal_projector_module sgformer \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --dtype bfloat16 \
    --remove_unused_columns False \
    --output_dir ./outputs/llama3.1-8b-$dataset_name/