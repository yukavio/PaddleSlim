#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python3.7 test_var.py \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=120 \
    --lr_strategy="cosine_decay" \
    --model_path="./lll_models"
