#!/bin/bash

# define env. variables
model_name=resnet
model_dir=vision/classification_and_detection
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)

# work on model directory
cd $work_dir

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=Offline
OLD_MODEL_PATH=./model/resnet/resnet50-19c8e357.pth
MODEL_PATH=./model/resnet/resnet50-19c8e357-pytorch-native.pth
DATASET_PATH=./dataset/imagenet/val
LABEL_PATH=./dataset/imagenet/aux/val.txt
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=50000 # total_len=50,000

python tools/resnet50_v1_to_pytorch-native.py $OLD_MODEL_PATH $MODEL_PATH
python python/main.py --profile=resnet50-pytorch --scenario=$SCENARIO \
                      --model=$MODEL_PATH --dataset-path=$DATASET_PATH --dataset-list=$LABEL_PATH \
                      --output=$LOG_PATH --count=$N_COUNT --accuracy
python tools/accuracy-imagenet.py --imagenet-val-file=$LABEL_PATH \
                                  --mlperf-accuracy-file=$LOG_PATH/mlperf_log_accuracy.json \
                                  --verbose \
                                  &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
