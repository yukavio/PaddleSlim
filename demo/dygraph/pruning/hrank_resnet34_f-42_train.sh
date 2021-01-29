#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python3.7 -m paddle.distributed.launch \
--gpus="0,1,2,3" \
--log_dir="hrank_resnet34_f-42_025_train_log" \
new_train.py \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=120 \
    --lr_strategy="cosine_decay" \
    --model_path="./hrank_resnet34_025_120_models" &
