#!/bin/bash

# define env. variables
model_name=bert
model_dir=language/bert
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
MODEL_PATH=./model/model.pytorch
VOCAB_PATH=./model/vocab.txt
DATASET_PATH=./dataset/dev-v1.1.json
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=10833 # total_len = 10,833

LOG_PATH=$LOG_PATH \
ML_MODEL_FILE_WITH_PATH=$MODEL_PATH \
VOCAB_FILE=$VOCAB_PATH \
DATASET_FILE=$DATASET_PATH \
SKIP_VERIFY_ACCURACY=true python run.py --scenario=$SCENARIO --backend=pytorch --max_examples=$N_COUNT --accuracy
python accuracy-squad.py --vocab_file=$VOCAB_PATH --val_data=$DATASET_PATH \
                         --log_file=$LOG_PATH/mlperf_log_accuracy.json --out_file=$LOG_PATH/predictions.json \
                         &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
