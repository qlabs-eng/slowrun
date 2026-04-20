#!/bin/bash

# Phase 1: NCA Pre-Pre-Training
WANDB_MODE=online torchrun --standalone --nproc_per_node=8 pre_pre_train.py \
    --tokens-per-epoch 50000000 --num-eval-tokens 16000000 \
    --num-epochs 6 \
    --regen-data \
    --grid 12 --patch 2 --num-colors 10 \
    --temperature 0.0001 --dT 1 --init-rollout-steps 10 \
    --filter-rules --filter-threshold 0.5 --filter-upper-bound 1.0 \
    --n-layer 30 --n-head 20 --n-embd 2560 \
    --seq-len 1024 --window-pattern SSSL \
    --device-batch-size 8 --total-batch-size 65536 \
    --lr 6e-4 --weight-decay 0.3 --dropout 0.1 \
    --warmdown-ratio 0.15 \
    --save-dir nca_ckpts/ppt_v8 \
    --run nca-ppt-v8

echo "NCA PPT done"
sleep 5

# Phase 2: Language Training
WANDB_MODE=online torchrun --standalone --nproc_per_node=8 train.py \
    --run lang-pre-train-v11 \
    --pretrained-ckpt nca_ckpts/ppt_v8/nca_pretrained_best.pt \
    --n_embd 2560 --n_head 20 --n_layer 30 \
    --num-epochs 25 \
    --total-batch-size 524288 \
    --lr_multiplier 0.25 \
    --weight-decay 1.3 \
    --dropout 0.1 \
    --nca-load-mode attn \
    --nca-warmup-steps 250 --nca-rampup-steps 200 \
    --warmdown-ratio 0.2 \
    --dupe-start-epoch 12 \
    --logit-avg 7 \
    --swa-last-epochs 6

echo "Language training done"

# w/o NCA training 
WANDB_MODE=online torchrun --standalone --nproc_per_node=8 train.py \
    --run lang-pre-train-v11-no-nca \
    --n_embd 2560 --n_head 20 --n_layer 30 \
    --num-epochs 25 \
    --total-batch-size 524288 \
    --lr_multiplier 0.25 \
    --weight-decay 1.3 \
    --dropout 0.1 \
    --warmdown-ratio 0.2 \
    --dupe-start-epoch 12 \
    --logit-avg 7 \
    --swa-last-epochs 6