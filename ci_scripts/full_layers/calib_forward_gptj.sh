#!/bin/bash

#######################################################################
# QGPT-J Forward Test Script
#
# This script performs the following steps:
# 1. Sets up environment variables and paths.
# 2. Activates the Conda environment.
# 3. Runs calibration.
# 4. Checks the equivalence of logits obtained at each generation step.
# 5. Cleans up and exits.
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
CALIB_DATA_PATH="$DATA_DIR/dataset/cnn-daily-mail/calibration/cnn_dailymail_calibration.json"

# Scenario and backend settings
SCENARIO="${SCENARIO:-Offline}"

# Evaluation settings
N_EVAL_DATA="${N_COUNT:-10}"     # Number of evaluation data samples (default: 10)
N_CALIB_DATA="${N_CALIB:-1000}"  # Number of calibration data samples (default: 1000)

# Quantization configuration
QUANT_CONFIG_PATH="$QUANT_DATA_DIR/quant_config_${CONFIG_DTYPE}.yaml"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S%Z)
LOG_PATH="$LOG_DIR/$MODEL_NAME/$SCENARIO/$TIMESTAMP"
CALIBRATION_RANGE_DIR="$LOG_PATH/calibration_range"
mkdir -p "$CALIBRATION_RANGE_DIR"

# Move to the working directory
cd "$WORK_DIR"

##############################
# Step 2: Activate Conda Environment
##############################

echo -e "\n============= STEP 1: Activate Conda Environment ============="

export CONDA_EXE="/anaconda/condabin/conda"
CONDA_BASE=$($CONDA_EXE info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate inference-ci

##############################
# Step 3: Run Calibration (if needed)
##############################

echo -e "\n============= STEP 2: Run Calibration ============="

echo -e "<<EVAL_CONFIG>>"
echo -e "\tSCENARIO: $SCENARIO"
echo -e "\tNUM_EVAL_DATA: $N_EVAL_DATA"

export LOG_PATH

echo -e "\tNUM_CALIB_DATA: $N_CALIB_DATA"
# Update quantization parameter paths to save in the log directory
QUANT_PARAM_PATH="$CALIBRATION_RANGE_DIR/quant_param.npy"
QUANT_FORMAT_PATH="$CALIBRATION_RANGE_DIR/quant_format.yaml"

# Run calibration
python -m quantization.calibrate \
    --model_path="$MODEL_PATH" \
    --quant_config_path="$QUANT_CONFIG_PATH" \
    --quant_param_path="$QUANT_PARAM_PATH" \
    --quant_format_path="$QUANT_FORMAT_PATH" \
    --calib_data_path="$CALIB_DATA_PATH" \
    --n_calib="$N_CALIB_DATA" \
    --gpu

echo "Calibration range saved to $CALIBRATION_RANGE_DIR"

# Paths for golden quantization parameters
GOLDEN_QUANT_PARAM_PATH="$CALIBRATION_RANGE_DIR/quant_param_golden.npy"
GOLDEN_QUANT_FORMAT_PATH="$CALIBRATION_RANGE_DIR/quant_format_golden.yaml"

##############################
# Step 4: Check Equivalence of Logits
##############################

echo -e "\n============= STEP 3: Check Equivalence of Logits ============="

# Number of data samples for the forward test
N_DATA="${N_DATA:-10}"

# Directory for logit files
LOGIT_FOLDER_PATH="$WORK_DIR/ci_file/logit_files"
mkdir -p "$LOGIT_FOLDER_PATH"

# Run forward test to compare logits
python -m ci_file.qgpt_j_forward_test \
    --model_path="$MODEL_PATH" \
    --quant_config_path="$QUANT_CONFIG_PATH" \
    --golden_quant_param_path="$GOLDEN_QUANT_PARAM_PATH" \
    --golden_quant_format_path="$GOLDEN_QUANT_FORMAT_PATH" \
    --submission_quant_param_path="$QUANT_PARAM_PATH" \
    --submission_quant_format_path="$QUANT_FORMAT_PATH" \
    --n_data="$N_DATA" \
    --dataset_path="$DATASET_PATH" \
    --logit_folder_path="$LOGIT_FOLDER_PATH" \
    --gpu \
    --ref_path="$REF_PATH" \
    --res_path="$RES_PATH" \
    --config_dtype="$CONFIG_DTYPE"
    # --update_gen_list  # Uncomment if you need to update the reference answers

##############################
# Step 5: Cleanup and Exit
##############################

# Unset environment variables
unset LOG_PATH
unset N_CALIB_DATA
unset N_DATA

echo -e "\n============= End of Forward Test for QGPT-J ============="

# Deactivate the Conda environment
conda deactivate

# Return to the git root directory
cd "$GIT_ROOT_DIR"
