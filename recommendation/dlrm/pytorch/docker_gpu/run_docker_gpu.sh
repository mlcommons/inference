#!/bin/bash

HOST_MLCOMMONS_ROOT_DIR=$HOME/mlcommons/inference	# path to mlcommons/inference
DLRM_DIR=$HOME/mlcommons/dlrm				# path to DLRM	
MODEL_DIR=$HOME/mlcommons/model-kaggle		# path to model folder
DATA_DIR=$HOME/mlcommons/data-kaggle			# path to data folder

docker run --gpus all -it \
-v $DLRM_DIR:/root/dlrm \
-v $MODEL_DIR:/root/model \
-v $DATA_DIR:/root/data \
-v $HOST_MLCOMMONS_ROOT_DIR:/root/mlcommons \
-e DATA_DIR=/root/data \
-e MODEL_DIR=/root/model \
-e DLRM_DIR=/root/dlrm \
-e CUDA_VISIBLE_DEVICES=0 \
dlrm-gpu 

