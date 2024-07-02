#!/bin/bash

# define env. variables
model_name=rnnt
model_dir=speech_recognition/rnnt
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
MODEL_PATH=$data_dir/models/rnnt/rnnt.pt 
DATASET_PATH=$data_dir/dataset/librispeech/validation
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)

python run.py --scenario $SCENARIO --dataset_dir $DATASET_PATH --pytorch_checkpoint $MODEL_PATH \
              --manifest $DATASET_PATH/dev-clean-wav.json --log_dir $LOG_PATH \
              --accuracy
python3 accuracy_eval.py --log_dir $LOG_PATH --dataset_dir $DATASET_PATH --manifest $DATASET_PATH/dev-clean-wav.json \
                         &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
