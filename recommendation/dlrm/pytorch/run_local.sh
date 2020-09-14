#!/bin/bash

source ./run_common.sh

common_opt="--mlperf_conf ../../../mlperf.conf"
OUTPUT_DIR=`pwd`/output/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

set -x # echo the next command
python python/main.py --profile $profile $common_opt --model $model --model-path $model_path \
       --dataset $dataset --dataset-path $DATA_DIR \
       --output $OUTPUT_DIR $EXTRA_OPS $@
