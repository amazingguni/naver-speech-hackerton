#!/bin/sh

BATCH_SIZE=8
WORKER_SIZE=8
GPU_SIZE=1
CPU_SIZE=4
DATASET="sr-hack-2019-dataset"
MAX_EPOCHS=200
D_MODEL=128
N_HEAD=4
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
DIM_FEEDFORWARD=2048
DROPOUT=0.1


#nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS"
nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --max_epochs $MAX_EPOCHS --d_model $D_MODEL --n_head $N_HEAD --num_encoder_layers $NUM_ENCODER_LAYERS --num_decoder_layers $NUM_DECODER_LAYERS --dim_feedforward $DIM_FEEDFORWARD --dropout $DROPOUT"
