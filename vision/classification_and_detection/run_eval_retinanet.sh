#!/bin/bash

# define env. variables
model_name=retinanet
model_dir=vision/classification_and_detection
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
conda env create -f eval_$model_name\_environment.yml
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
dvc pull model/retinanet --force
dvc pull dataset/openimages-mlperf/validation --force
dvc pull dataset/openimages-mlperf/annotations --force

# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=Offline
MODEL_PATH=./model/retinanet/resnext50_32x4d_fpn.pth
DATASET_DIR=./dataset/openimages-mlperf
DATASET_PATH=$DATASET_DIR/validation/data
ANNOTATION_PATH=$(pwd)/$DATASET_DIR/annotations/openimages-mlperf.json
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=24781 # total_len=24,781

python python/main.py --profile=retinanet-pytorch --scenario=$SCENARIO \
                      --model=$MODEL_PATH --dataset-path=$DATASET_PATH --dataset-list=$ANNOTATION_PATH \
                      --output=$LOG_PATH --count=$N_COUNT --accuracy
python tools/accuracy-openimages.py --openimages-dir=$DATASET_DIR \
                                    --mlperf-accuracy-file=$LOG_PATH/mlperf_log_accuracy.json \
                                    --output-file=$LOG_PATH/openimages-results.json --verbose \
                                    &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n=============End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
