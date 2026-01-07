#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -m eval.chat_completion \
    --model_name ./outputs/llama3.1-8b-la \
    --dataset_dir ./data/la \
    --quantization bfloat16 