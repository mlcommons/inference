#!/bin/bash

#######################################################################
# QBERT Backward Compatibility Test Script
#
# This script performs the following steps:
# 1. Sets up environment variables and paths.
# 2. Activates the Conda environment.
# 3. Pull DVC.
# 4. Convert the model to quantized model and Checks the equivalence of generated tokens.
# 5. Saves the results and logs.
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
MODEL_NAME="qgpt-j"
MODEL_DIR="language/gpt-j"

# Define paths
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
WORK_DIR="$GIT_ROOT_DIR/$MODEL_DIR"
DATA_DIR="/home/home-mcl/shared_data/"
REF_PATH="$DATA_DIR/quant/gpt-j/ref"
RES_PATH="$DATA_DIR/results"
QUANT_DATA_DIR="$DATA_DIR/quant/gpt-j"
LOG_DIR="$GIT_ROOT_DIR/logs"

# Model and dataset paths
MODEL_PATH="$DATA_DIR/models/gpt-j"
DATASET_PATH="$DATA_DIR/dataset/cnn-daily-mail/validation/cnn_eval.json"

# Scenario and backend settings
SCENARIO=${SCENARIO:="Offline"}

# Quantization configuration
QUANT_CONFIG_PATH="$QUANT_DATA_DIR/quant_config_${CONFIG_DTYPE}.yaml"

# Move to the working directory
cd $WORK_DIR

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S%Z)

N_DATA=${N_DATA:="3"}

##############################
# Step 2: Activate Conda Environment
##############################

echo -e "\n============= STEP 1: Activate Conda Environment ============="

export CONDA_EXE="/anaconda/condabin/conda"
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

# eval model
printf "\n============= STEP-2: pull dvc =============\n"

# TAG=main
# cd $git_dir
# git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
# cd $git_dir/furiosa-llm-models-artifacts
# git checkout $TAG

# quant_data_dvc_dir=/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8fA8fKV8f/28L
# dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml.dvc -r origin --force
# dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy.dvc -r origin --force


# cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml $MODEL_DATA_DIR/quant_format.yaml
# cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy $MODEL_DATA_DIR/quant_param.npy

# rm -rf $git_dir/furiosa-llm-models-artifacts

if [ "$CONFIG_DTYPE" == "fp8" ]; then
    MODEL_DATA_DIR=$DATA_DIR/furiosa_llm_models_artifacts/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8fA8fKV8f/28L
elif [ "$CONFIG_DTYPE" == "int8" ]; then
    MODEL_DATA_DIR=$DATA_DIR/furiosa_llm_models_artifacts/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8A8KV8/28L
fi

QUANT_FORMAT_PATH=$MODEL_DATA_DIR/quant_format.yaml
QUANT_PARAM_PATH=$MODEL_DATA_DIR/quant_param.npy

LOGIT_FOLDER_PATH=$WORK_DIR/ci_file/logit_files
mkdir -p $LOGIT_FOLDER_PATH

##############################
# Step 3: Check Equivalence of Generated Tokens
##############################

printf "\n============= STEP-3: Check the equivalence of generated tokens (현재 MCP에서 과거 qparam, qformat 작동 유무 확인 + 토큰 동치) =============\n"

python -m ci_file.backward_compatibility_test_qgpt_j_forward_test\
                                                --model_path=$MODEL_PATH \
                                                --quant_config_path=$QUANT_CONFIG_PATH \
                                                --submission_quant_format_path=$QUANT_FORMAT_PATH \
                                                --submission_quant_param_path=$QUANT_PARAM_PATH \
                                                --n_data=$N_DATA \
                                                --dataset_path=$DATASET_PATH \
                                                --logit_folder_path=$LOGIT_FOLDER_PATH \
                                                --gpu \
                                                --ref_path=$REF_PATH \
                                                --res_path=$RES_PATH \
                                                --config_dtype=$CONFIG_DTYPE 

##############################
# Step 5: Cleanup and Exit
##############################
unset N_DATA

printf "\n============= End of Forward Test for QGPT-J =============\n"


# Deactivate the Conda environment
conda deactivate

# Return to the git root directory
cd "$GIT_ROOT_DIR"