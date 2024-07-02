#!/bin/bash

# define env. variables
model_name=gpt-j
model_dir=language/gpt-j
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
MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=13368 # total_len=13,368

LOG_PATH=$LOG_PATH python main.py --scenario=$SCENARIO --model-path=$MODEL_PATH --dataset-path=$DATASET_PATH --max_examples=$N_COUNT --accuracy --gpu
python evaluation.py --mlperf-accuracy-file=$LOG_PATH/mlperf_log_accuracy.json --dataset-file=$DATASET_PATH \
                     &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
