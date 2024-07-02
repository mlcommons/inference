#!/bin/bash

# define env. variables
model_name=bert-squad
model_dir=language/bert
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)

# work on model directory
cd $work_dir
git submodule update --init DeepLearningExamples

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
MODEL_PATH=./model/model.pytorch
VOCAB_PATH=./model/vocab.txt
DATASET_PATH=./dataset/dev-v1.1.json
LOG_PATH=./logs/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=10833 # total_len = 10833

LOG_PATH=$LOG_PATH \
ML_MODEL_FILE_WITH_PATH=$MODEL_PATH \
VOCAB_FILE=$VOCAB_PATH \
DATASET_FILE=$DATASET_PATH \
SKIP_VERIFY_ACCURACY=true python run.py --scenario=$SCENARIO --backend=pytorch --max_examples=$N_COUNT --accuracy
python accuracy-squad.py --vocab_file=$VOCAB_PATH --val_data=$DATASET_PATH \
                         --log_file=$LOG_PATH/mlperf_log_accuracy.json --out_file=$LOG_PATH/predictions.json

printf "\n=============End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
