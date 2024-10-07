#!/bin/bash

#######################################################################
# LLaMA3-70B Load Test Script
#
# This script performs the following steps:
# 1. Sets up environment variables and paths.
# 2. Activates the Conda environment.
# 3. Runs the load test.
# 4. Cleans up and exits.
#
#######################################################################

set -e  # 에러 발생시 bash script 즉시 종료

##############################
# Step 0: DTYPE args 처리
##############################

# 기본 데이터 타입 설정
CONFIG_DTYPE="fp8"

# 인자가 제공되면 데이터 타입 설정
if [ $# -ge 1 ]; then
    CONFIG_DTYPE="$1"
    # 유효한 데이터 타입인지 확인
    if [ "$CONFIG_DTYPE" != "fp8" ] && [ "$CONFIG_DTYPE" != "int8" ]; then
        echo "유효하지 않은 dtype입니다: $CONFIG_DTYPE"
        echo "사용법: ./ci_bert.sh [dtype]"
        echo "  dtype: fp8 (기본값) 또는 int8"
        exit 1
    fi
fi

echo "사용할 데이터 타입: $CONFIG_DTYPE"


##############################
# Step 1: Define Environment Variables and Paths
##############################

# Define model and configuration parameters
MODEL_NAME="llama3-70b"
MODEL_DIR="language/llama2-70b" #TODOS

# Define paths
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
WORK_DIR="$GIT_ROOT_DIR/$MODEL_DIR"
DATA_DIR="/home/home-mcl/shared_data"
MODEL_DATA_DIR="$DATA_DIR/quant/$MODEL_NAME"
LOG_DIR="$GIT_ROOT_DIR/logs"

# Model and quantization paths
MODEL_PATH="$DATA_DIR/models/llama3/Meta-Llama-3.1-70B-Instruct"
QUANT_CONFIG_PATH="$MODEL_DATA_DIR/quant_config_$CONFIG_DTYPE.yaml"

if [ "$CONFIG_DTYPE" == "fp8" ]; then
    QUANT_DATA_PATH="$DATA_DIR/furiosa_llm_models_artifacts/quantized/meta-llama/Meta-Llama-3.1-70B-Instruct/mlperf_submission_slice/W8fA8fKV8f/32L"
elif [ "$CONFIG_DTYPE" == "int8" ]; then
    QUANT_DATA_PATH="$DATA_DIR/furiosa_llm_models_artifacts/quantized/meta-llama/Meta-Llama-3.1-70B-Instruct/mlperf_submission_slice/W8A8KV8/32L"
fi

# work on model directory
cd $WORK_DIR

# enter existing conda env.
export CONDA_EXE="/anaconda/condabin/conda"
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

# eval model
SCENARIO=${SCENARIO:="Offline"}

# quantization args
export CALIBRATE=true

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH

printf "\n============= QLV4 Load TEST =============\n"

python -m ci_file.qllama3_load_test  --model_path=$MODEL_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_data_path=$QUANT_DATA_PATH


printf "\n============= End of Test for llama3.1 =============\n"

# unset exported env. variables
unset SCENARIO
unset LOG_PATH
unset CALIBRATE
unset N_CALIB

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir

