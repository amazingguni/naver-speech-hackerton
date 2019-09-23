#!/bin/sh

BATCH_SIZE=64
WORKER_SIZE=32
GPU_SIZE=2
CPU_SIZE=4
DATASET="sr-hack-2019-dataset"
MAX_EPOCHS=100
#MAX_EPOCHS=1
#LAYER_SIZE=1
LAYER_SIZE=5

#nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS"
nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS --layer_size $LAYER_SIZE --word"
