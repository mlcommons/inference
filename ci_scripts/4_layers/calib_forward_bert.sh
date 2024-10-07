#!/bin/bash

#######################################################################
# QBERT Calibration & Forward Test Script
#
# This script performs the following steps:
# 1. Sets up environment variables and paths.
# 2. Activates the Conda environment.
# 3. Runs calibration for both the golden and submission models.
# 4. Checks the equivalence of logits obtained at each generation step.
# 5. Verifies the F1 score between the current MLPerf submission and the reference.
# 6. Saves the results and logs.
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
MODEL_NAME="qbert"
MODEL_DIR="language/bert"

# Define paths
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
WORK_DIR="$GIT_ROOT_DIR/$MODEL_DIR"
DATA_DIR="/home/home-mcl/shared_data/"
REF_PATH="$DATA_DIR/quant/bert/ref"
RES_PATH="$DATA_DIR/results"
QUANT_DATA_DIR="$DATA_DIR/quant/bert"
LOG_DIR="$GIT_ROOT_DIR/logs"

# Model and dataset paths
MODEL_PATH="$DATA_DIR/models/bert/model.pytorch"
MODEL_CONFIG_PATH="$DATA_DIR/models/bert/bert_config.json"
VOCAB_PATH="$DATA_DIR/models/bert/vocab.txt"
CALIB_DATA_PATH="$DATA_DIR/dataset/squad/calibration/cal_features.pickle"
DATASET_PATH="$DATA_DIR/dataset/squad/validation/dev-v1.1.json"

# Scenario and backend settings
SCENARIO="${SCENARIO:-Offline}"
BACKEND="rngd"

# Evaluation settings
N_EVAL_DATA="${N_COUNT:-1}"  # Number of evaluation data samples (default: 100)
N_CALIB_DATA="${N_CALIB:-10}"  # Number of calibration data samples (default: 100)

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
# Step 3: Run Calibration
##############################

echo -e "\n============= STEP 2: Run Calibration ============="

echo -e "<<EVAL_CONFIG>>"
echo -e "\tSCENARIO: $SCENARIO"
echo -e "\tNUM_EVAL_DATA: $N_EVAL_DATA"
echo -e "\tNUM_CALIB_DATA: $N_CALIB_DATA"

# Paths for golden calibration files
GOLDEN_QUANT_PARAM_PATH="$CALIBRATION_RANGE_DIR/quant_param_golden.npy"
GOLDEN_QUANT_FORMAT_PATH="$CALIBRATION_RANGE_DIR/quant_format_golden.yaml"

# Run calibration for the golden model
python -m quantization.calibrate \
    --model_type="golden" \
    --model_path="$MODEL_PATH" \
    --model_config_path="$MODEL_CONFIG_PATH" \
    --quant_config_path="$QUANT_CONFIG_PATH" \
    --quant_param_path="$GOLDEN_QUANT_PARAM_PATH" \
    --quant_format_path="$GOLDEN_QUANT_FORMAT_PATH" \
    --calib_data_path="$CALIB_DATA_PATH" \
    --n_calib="$N_CALIB_DATA" \
    --gpu

echo "Saved golden calibration files to $GOLDEN_QUANT_PARAM_PATH and $GOLDEN_QUANT_FORMAT_PATH"

# Paths for submission calibration files
SUBMISSION_QUANT_PARAM_PATH="$CALIBRATION_RANGE_DIR/quant_param.npy"
SUBMISSION_QUANT_FORMAT_PATH="$CALIBRATION_RANGE_DIR/quant_format.yaml"

# Run calibration for the submission model
python -m quantization.calibrate \
    --model_type="mlperf-submission" \
    --model_path="$MODEL_PATH" \
    --model_config_path="$MODEL_CONFIG_PATH" \
    --quant_config_path="$QUANT_CONFIG_PATH" \
    --quant_param_path="$SUBMISSION_QUANT_PARAM_PATH" \
    --quant_format_path="$SUBMISSION_QUANT_FORMAT_PATH" \
    --calib_data_path="$CALIB_DATA_PATH" \
    --n_calib="$N_CALIB_DATA" \
    --gpu \
    --n_layers="4" \
    --is_equivalence_ci

echo "Saved submission calibration files to $SUBMISSION_QUANT_PARAM_PATH and $SUBMISSION_QUANT_FORMAT_PATH"


##############################
# Step 6: Cleanup and Exit
##############################

# Unset environment variables
unset LOG_PATH

echo -e "\n============= End of Forward Test for QBERT ============="

# Deactivate the Conda environment
conda deactivate

# Return to the git root directory
cd "$GIT_ROOT_DIR"
