#!/bin/bash

DATABASE_ADDRESS=localhost # the ip of database to publish traces to
DATABASE_NAME=resnet50_caffe2_full_trace # the name of database to publish traces to
MODEL_NAME=ResNet50_Caffe2 # model name
MODEL_VERSION=1.0 # model version
NUM_FILE_PARTS=20 # number of batches to be processed, set to -1 to evalute the entire ImageNet
BATCH_SIZE=64 # batch size
TRACE_LEVEL=FULL_TRACE # trace level
TRACER_ADDRESS=localhost:16686 # the endpoint of tracer

nvidia-docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-gpu-latest predict dataset \
    --verbose \
    --publish=true \
    --publish_predictions=false \
    --fail_on_error=true \
    --num_file_parts=$NUM_FILE_PARTS \
    --gpu=true \
    --batch_size=$BATCH_SIZE \
    --model_name=$MODEL_NAME \
    --model_version=$MODEL_VERSION \
    --database_name=$DATABASE_NAME \
    --database_address=$DATABASE_ADDRESS \
    --tracer_address=$TRACER_ADDRESS

# docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation latency \
#       --batch_size=$BATCH_SIZE \
#       --model_name=$MODEL_NAME \
#       --model_version=$MODEL_VERSION \
#       --database_name=$DATABASE_NAME \
#       --database_address=$DATABASE_ADDRESS \

# docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation accuracy \
#       --batch_size=$BATCH_SIZE \
#       --model_name=$MODEL_NAME \
#       --model_version=$MODEL_VERSION \
#       --database_name=$DATABASE_NAME \
#       --database_address=$DATABASE_ADDRESS
