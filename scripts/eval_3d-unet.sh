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

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=Offline
MODEL_PATH=./model/3dunet_kits19_pytorch_checkpoint.pth
PREPROC_DATASET_DIR=./dataset/kits19/preprocessed_data/
LOG_PATH=$git_dir/logs/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)

LOG_PATH=$LOG_PATH \
SKIP_VERIFY_ACCURACY=true \
python run.py --scenario=$SCENARIO --backend=pytorch_checkpoint \
              --model=$MODEL_PATH --preprocessed_data_dir=$PREPROC_DATASET_DIR \
              --accuracy
python accuracy_kits.py --log_file=$LOG_PATH/mlperf_log_accuracy.json \
                        --preprocessed_data_dir=$PREPROC_DATASET_DIR \
                        --postprocessed_data_dir=$LOG_PATH/predictions \
                        &> $LOG_PATH/accuracy_result.log

printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
