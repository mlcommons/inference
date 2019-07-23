#!/bin/bash


# CI end to end test for all models

if [ "x$DATA_ROOT" == "x" ]; then
    echo "DATA_ROOT not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi


# gloal options for all runs
gopt="--max-latency 0.05,0.1"

# quick run (you'd set this from the ci system)
# gopt="$gopt --queries-single 1000 --queries-offline 1000 --queries-multi 1000 --time 30 --count 500"

gopt="$gopt $@"

function one_run {
    ./run_local.sh $* --scenario SingleStream 
    ./run_local.sh $* --scenario SingleStream --accuracy
    ./run_local.sh $* --scenario MultiStream
    ./run_local.sh $* --scenario Server
    ./run_local.sh $* --scenario Offline
}

export DATA_DIR=$DATA_ROOT/imagenet2012

#
# resnet50
#
one_run tf gpu resnet50 --qps 200 $gopt
one_run tf cpu resnet50 --qps 20 $gopt
one_run onnxruntime cpu resnet50 --qps 20 $gopt

#
# mobilenet
#
one_run tf gpu mobilenet --qps 250 $gopt
one_run tf cpu mobilenet --qps 20 $gopt
one_run onnxruntime cpu mobilenet --qps 20 $gopt

export DATA_DIR=$DATA_ROOT/coco

#
# ssd-mobilenet
#
one_run tf gpu ssd-mobilenet --qps 50 $gopt
one_run tf cpu ssd-mobilenet --qps 50 $gopt
one_run onnxruntime cpu ssd-mobilenet --qps 50 $gopt

one_run tf gpu ssd-resnet34 --qps 50 $gopt
one_run tf cpu ssd-resnet34 --qps 50 $gopt
one_run onnxruntime cpu ssd-resnet34-tf --qps 1 $gopt
#one_run onnxruntime cpu ssd-resnet34 --qps 1 $gopt
#one_run pytorch cpu ssd-resnet34 --qps 1 $gopt
#one_run pytorch gpu ssd-resnet34 --qps 1 $gopt
