#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 tf|onnxruntime [cpu|gpu]"
    exit 1
fi
if [ $1 == "tf" ] ; then
    model="$MODEL_DIR/resnet50_v1.pb"
    profile=resnet50-tf
fi
if [ $1 == "onnxruntime" ] ; then
    model="$MODEL_DIR/resnet50_v1.onnx"
    profile=resnet50-onnxruntime
fi
target=cpu
dockercmd=docker
if [ $# -ge 2 ]; then
    target=$2
fi
if [ $target == "gpu" ]; then
    dockercmd=nvidia-docker
fi
if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi

OUTPUT_DIR=`pwd`/output/$profile.$target
mkdir -p $OUTPUT_DIR

image=mlperf-infer-imgclassify-$target
docker build  -t $image -f Dockerfile.$target .
$dockercmd run -e profile=$profile -e EXTRA_OPS="$EXTRA_OPS" \
    -v $DATA_DIR:/data -v $MODEL_DIR:/model -v `pwd`:/mlperf \
    -v $OUTPUT_DIR:/output -v /proc:/host_proc \
    -t $image:latest /mlperf/run_helper.sh 2>&1 | tee $OUTPUT_DIR/output.txt
