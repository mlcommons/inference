#!/bin/bash

# define env. variables
model_name=retinanet
model_dir=vision/classification_and_detection
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
log_dir=$git_dir/logs
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
MODEL_PATH=$data_dir/models/retinanet/resnext50_32x4d_fpn.pth
DATASET_DIR=$data_dir/dataset/openimages-mlperf
DATASET_PATH=$DATASET_DIR/validation/data
ANNOTATION_PATH=$DATASET_DIR/annotations/openimages-mlperf.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=24781 # total_len=24,781

python python/main.py --profile=retinanet-pytorch --scenario=$SCENARIO \
                      --model=$MODEL_PATH --dataset-path=$DATASET_PATH --dataset-list=$ANNOTATION_PATH \
                      --output=$LOG_PATH --count=$N_COUNT --accuracy
python tools/accuracy-openimages.py --openimages-dir=$DATASET_DIR \
                                    --mlperf-accuracy-file=$LOG_PATH/mlperf_log_accuracy.json \
                                    --output-file=$LOG_PATH/openimages-results.json --verbose \
                                    &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
