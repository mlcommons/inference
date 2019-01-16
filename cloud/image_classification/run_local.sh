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
if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi
shift

target=cpu
if [ $# -ge 1 ]; then
    target=$1
    shift
fi
if [ $target == "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi
common_opt="--count 500 --time 10"
dataset="--dataset-path $DATA_DIR"
OUTPUT_DIR=`pwd`/output/$profile.$target
if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi
python python/main.py --profile $profile $common_opt --model $model $dataset --output $OUTPUT_DIR/results.json $EXTRA_OPS $@
