#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed> <target threshold>
SEED=${1:-1}
TARGET=${2:-0.19588}
DATASET_DIR='../coco'
CHECKPOINT='./pretrained/resnet34-ssd1200.pth'
time stdbuf -o 0 \
  python3 infer.py --seed $SEED --threshold $TARGET --data ${DATASET_DIR} --device 1 --checkpoint $CHECKPOINT --no-cuda | tee run.log.$SEED  
