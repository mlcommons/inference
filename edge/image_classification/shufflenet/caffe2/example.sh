#!/bin/bash

DATABASE_ADDRESS=localhost # the ip of database to publish traces to
DATABASE_NAME=shufflenet_model_trace # the name of database to publish traces to
MODEL_NAME=ShuffleNet_Caffe2 # model name
MODEL_VERSION=1.0 # model version
NUM_FILE_PARTS=100 # number of batches to be processed, set to -1 to evalute the entire ImageNet
BATCH_SIZE=1 # batch size
TRACE_LEVEL=MODEL_TRACE # trace level
TRACER_ADDRESS=localhost # the ip of tracer

docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=true \
      --publish_predictions=false \ 
      --gpu=0 \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS \
      --trace_level=$TRACE_LEVEL \
      --tracer_address-$TRACER_ADDRESS

docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation latency \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS

docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation acurracy \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS