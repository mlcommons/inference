#!/bin/bash

# define env. variables
model_name=gpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)

# work on model directory
cd $work_dir

# create and enter conda env.
printf "\n============= STEP-1: Create conda environment and activate =============\n"
conda remove -n $env_name --all -y
rm -rf $conda_base/env/$env_name
conda env create -f eval_environment.yml
set +u
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name
set -u

# build mlperf loadgen
printf "\n============= STEP-2: Build mlperf loadgen =============\n"
pip install pybind11==2.11.1
cd $git_dir/loadgen; python setup.py install
cd -

# pull model and dataset
printf "\n============= STEP-3: Pull dvc data =============\n"
pip install dvc[s3]
dvc pull model --force
dvc pull dataset --force

# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=Offline
MODEL_PATH=./model
DATASET_PATH=./dataset/cnn_eval.json
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=13368 # total_len=13,368

LOG_PATH=$LOG_PATH python main.py --scenario=$SCENARIO --model-path=$MODEL_PATH --dataset-path=$DATASET_PATH --max_examples=$N_COUNT --accuracy --gpu
python evaluation.py --mlperf-accuracy-file=$LOG_PATH/mlperf_log_accuracy.json --dataset-file=$DATASET_PATH \
                     &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n=============End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
