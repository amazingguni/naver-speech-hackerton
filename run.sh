#!/bin/sh

export CUDA_VISIBLE_DEVICES="2"

BATCH_SIZE=8
WORKER_SIZE=8
MAX_EPOCHS=200
D_MODEL=128
N_HEAD=4
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
DIM_FEEDFORWARD=2048
DROPOUT=0.1

python ./main.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --max_epochs $MAX_EPOCHS --batch_size $BATCH_SIZE --workers $WORKER_SIZE --max_epochs $MAX_EPOCHS --d_model $D_MODEL --n_head $N_HEAD --num_encoder_layers $NUM_ENCODER_LAYERS --num_decoder_layers $NUM_DECODER_LAYERS --dim_feedforward $DIM_FEEDFORWARD --dropout $DROPOUT
