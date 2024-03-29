#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

BATCH_SIZE=16
WORKER_SIZE=8
MAX_EPOCHS=50

python ./main.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS --transformer
