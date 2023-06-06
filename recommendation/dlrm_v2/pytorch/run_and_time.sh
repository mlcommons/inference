#!/bin/bash

source run_common.sh

dockercmd=docker
if [ $device == "gpu" ]; then
    runtime="--runtime=nvidia"
fi

# copy the config to cwd so the docker contrainer has access
cp ../../../mlperf.conf .

OUTPUT_DIR=`pwd`/output/$name
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

image=mlperf-infer-recommendation-$device
docker build  -t $image -f Dockerfile.$device .
opts="--config ./mlperf.conf --profile $profile $common_opt --model $model \
    --model-path $model_path --dataset $dataset --dataset-path $DATA_DIR \
    --output $OUTPUT_DIR $EXTRA_OPS $@"

docker run $runtime -e opts="$opts" \
    -v $DATA_DIR:$DATA_DIR -v $MODEL_DIR:$MODEL_DIR -v `pwd`:/mlperf \
    -v $OUTPUT_DIR:/output -v /proc:/host_proc \
    -t $image:latest /mlperf/run_helper.sh 2>&1 | tee $OUTPUT_DIR/output.txt
