#!/bin/bash

source ./run_common.sh

common_opt="--mlperf_conf ../../mlperf.conf"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=${OUTPUT_DIR:-`pwd`/output/$name}
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

pattern=" |'"
while [ -n "$1" ]; do
    if [[ $1 =~ $pattern ]]; then
        ARGS=$ARGS' "'$1'"'
    else
        ARGS="$ARGS $1"
    fi
    shift
done

cmd="python3 python/main.py --profile $profile $common_opt --model \"$model_path\" $dataset \
    --output \"$OUTPUT_DIR\" $EXTRA_OPS ${ARGS}"

if [[ $EXTRA_OPS == *"tpu"* ]]; then
    cmd="sudo $cmd"
fi

echo $cmd
eval $cmd
