#!/bin/bash

data_dir=${DATA_DIR:-./nmt/data}
model_dir=${MODEL_DIR:-./ende_gnmt_model_4_layer}
output_dir=${OUTPUT_DIR:-./nmt/data}
mode=${mode:-performance}
output_dir_mode=$output_dir/$mode

if [ ! -d "${output_dir_mode}" ]; then
    mkdir -p $output_dir_mode
fi

time python run_task.py \
    --run=$mode \
    --batch_size=$batch_size \
    --dataset_path=$data_dir \
    --model_path=$model_dir \
    --output_path=$output_dir_mode
