#!/bin/bash

source run_common.sh

dockercmd=docker
if [ $device == "gpu" ]; then
-    dockercmd=nvidia-docker
fi


OUTPUT_DIR=`pwd`/output/$name
mkdir -p $OUTPUT_DIR

image=mlperf-infer-imgclassify-$device
docker build  -t $image -f Dockerfile.$device .
model_path_in_container="/data/$(basename $model_path)"
opts="--profile $profile $common_opt --model $model_path_in_container $dataset \
    --output $OUTPUT_DIR/results.json $extra_args $EXTRA_OPS"

$dockercmd run -e opts="$opts" \
    -v $DATA_DIR:/data -v $MODEL_DIR:/model -v `pwd`:/mlperf \
    -v $OUTPUT_DIR:/output -v /proc:/host_proc \
    -t $image:latest /mlperf/run_helper.sh 2>&1 | tee $OUTPUT_DIR/output.txt
