#!/bin/bash

# define env. variables
model_name=rnnt
model_dir=speech_recognition/rnnt
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)


# work on model directory
cd $work_dir

# create and enter conda env.
printf "\n============= STEP-1: Create conda environment and activate =============\n"
conda remove -n $env_name --all -y
rm -rf $conda_base/envs/$env_name
conda env create -f eval_environment.yml
# conda env create -f environment.yml
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
OLD_MODEL_PATH=./model/rnnt.pt
MODEL_PATH=./model/rnnt.pt 
DATASET_PATH=./dataset
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)

python run.py --scenario $SCENARIO --dataset_dir $DATASET_PATH --pytorch_checkpoint $MODEL_PATH \
              --manifest $DATASET_PATH/dev-clean-wav.json --log_dir $LOG_PATH  --accuracy

printf "\n=============End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
