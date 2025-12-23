#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 \
    python main.py --dataset sampled-streaming-100b 2>&1 | tee /home/$USER/dlrmv3-inference-benchmark.log
