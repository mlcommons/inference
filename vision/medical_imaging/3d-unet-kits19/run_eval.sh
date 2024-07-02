#!/bin/bash

# define env. variables
model_name=3d-unet
model_dir=vision/medical_imaging/3d-unet-kits19
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
MODEL_PATH=./model/3dunet_kits19_pytorch_checkpoint.pth
PREPROC_DATASET_DIR=./dataset/kits19/preprocessed_data/
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=43 # total_len=43

LOG_PATH=$LOG_PATH \
SKIP_VERIFY_ACCURACY=true \
python run.py --scenario=$SCENARIO --backend=pytorch_checkpoint \
              --model=$MODEL_PATH --preprocessed_data_dir=$PREPROC_DATASET_DIR \
              --performance_count=$N_COUNT --accuracy
python accuracy_kits.py --log_file=$LOG_PATH/mlperf_log_accuracy.json \
                        --preprocessed_data_dir=$PREPROC_DATASET_DIR \
                        --postprocessed_data_dir=$LOG_PATH/predictions

printf "\n=============End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
