#!/bin/bash

HOST_MLCOMMONS_ROOT_DIR=$HOME/mlcommons/inference	# path to mlcommons/inference
DLRM_DIR=$HOME/mlcommons/dlrm				# path to DLRM	
MODEL_DIR=$HOME/mlcommons/model-terabyte		# path to model folder
DATA_DIR=$HOME/mlcommons/data-terabyte		# path to data folder

docker run -it \
-v $DLRM_DIR:/root/dlrm \
-v $MODEL_DIR:/root/model \
-v $DATA_DIR:/root/data \
-v $HOST_MLCOMMONS_ROOT_DIR:/root/mlcommons \
-e DATA_DIR=/root/data \
-e MODEL_DIR=/root/model \
-e DLRM_DIR=/root/dlrm \
dlrm-cpu 

