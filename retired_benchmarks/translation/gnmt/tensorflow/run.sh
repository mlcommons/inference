#!/bin/bash

data_dir=${DATA_DIR:-./nmt/data}
model_dir=${MODEL_DIR:-./ende_gnmt_model_4_layer}
output_dir=${MODEL_DIR:-./nmt/data}
task=${TASK:-performance}

time python run_task.py \
    --run=$task \
    --batch_size=$batch_size \
    --dataset_path=$data_dir \
    --model_path=$model_dir \
    --output_path=$output_dir
