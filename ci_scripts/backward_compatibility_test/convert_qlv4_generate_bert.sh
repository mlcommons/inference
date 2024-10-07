#!/bin/bash

#######################################################################
# QBERT Backward Compatibility Test Script
#
# This script performs the following steps:
# 1. Sets up environment variables and paths.
# 2. Activates the Conda environment.
# 3. Pull DVC.
# 4. Convert the model to quantized model and Checks the equivalence of logits obtained at each generation step.
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
DATASET_PATH="$DATA_DIR/dataset/squad/calibration/cal_features.pickle" # use calibration data for forward test

# Scenario and backend settings
SCENARIO="${SCENARIO:-Offline}"
BACKEND="rngd"

# Quantization configuration
QUANT_CONFIG_PATH="$QUANT_DATA_DIR/quant_config_${CONFIG_DTYPE}.yaml"

# Move to the working directory
cd "$WORK_DIR"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S%Z)
LOG_PATH="$LOG_DIR/$MODEL_NAME/$SCENARIO/$TIMESTAMP"

N_COUNT=${N_COUNT:="100"} # total_len = 10,833
N_DATA=${N_DATA:="1"} # total_len = 10,833

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

# quant_data_dvc_dir=/quantized/furiosa-ai/mlperf-bert-large/mlperf_submission/W8fA8f/24L/
# dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml.dvc -r origin --force
# dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy.dvc -r origin --force

# cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy $MODEL_DATA_DIR/quant_param.npy

# rm -rf $git_dir/furiosa-llm-models-artifacts

if [ "$CONFIG_DTYPE" == "fp8" ]; then
    MODEL_DATA_DIR=$DATA_DIR/furiosa_llm_models_artifacts/quantized/furiosa-ai/mlperf-bert-large/mlperf_submission/W8fA8f/24L
elif [ "$CONFIG_DTYPE" == "int8" ]; then
    MODEL_DATA_DIR=$DATA_DIR/furiosa_llm_models_artifacts/quantized/furiosa-ai/mlperf-bert-large/mlperf_submission/W8A8/24L
fi

QUANT_FORMAT_PATH=$MODEL_DATA_DIR/quant_format.yaml
QUANT_PARAM_PATH=$MODEL_DATA_DIR/quant_param.npy

LOGIT_FOLDER_PATH=$WORK_DIR/ci_file/logit_files
mkdir -p $LOGIT_FOLDER_PATH

##############################
# Step 3: Check Equivalence of Generated Tokens
##############################

printf "\n============= STEP-3: Check the equivalence of generated tokens (현재 MCP에서 과거 qparam, qformat 작동 유무 확인 + 토큰 동치) =============\n"

python -m ci_file.backward_compatibility_test_qbert_forward \
                                    --model_path=$MODEL_PATH \
                                    --model_config_path=$MODEL_CONFIG_PATH \
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
# Step 4: Verify F1 Score Equivalence
##############################

printf "\n============= STEP-4: Check the equivalence of f1 score between current mlperf submission <-> ref =============\n"

# Export necessary environment variables
export ML_MODEL_FILE_WITH_PATH="$MODEL_PATH"
export LOG_PATH="$LOG_PATH"
export DATASET_FILE="$DATA_DIR/dataset/squad/validation/dev-v1.1.json"
export SKIP_VERIFY_ACCURACY=true

# Start timer
SECONDS=0

# Run inference with accuracy check
python -m run --scenario=$SCENARIO \
              --backend=$BACKEND \
              --gpu \
              --quantize \
              --quant_param_path=$QUANT_PARAM_PATH \
              --quant_format_path=$QUANT_FORMAT_PATH \
              --max_examples=$N_COUNT \
              --accuracy

# Record the elapsed time
duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

# Path for the accuracy log file
ACCURACY_LOG_FILE=$LOG_PATH/mlperf_log_accuracy.json

# Run accuracy evaluation
python accuracy-squad.py --vocab_file=$VOCAB_PATH \
                         --val_data=$DATASET_FILE \
                         --log_file=$ACCURACY_LOG_FILE \
                         --out_file=$LOG_PATH/predictions.json \
                         --max_examples=$N_COUNT \
                         > "$LOG_PATH/accuracy_result_$CONFIG_DTYPE.log"

# Extract the current F1 score
CUR_F1_SCORE=$(grep -oP '"f1":\s*\K[0-9.]+' "$LOG_PATH/accuracy_result_$CONFIG_DTYPE.log")

if [ "$CONFIG_DTYPE" == "fp8" ]; then
    REF_F1_SCORE=95.46667  # Reference F1 score for fp8 configuration
elif [ "$CONFIG_DTYPE" == "int8" ]; then
    REF_F1_SCORE=94.86666  # Reference F1 score for fp8 configuration
fi

# Calculate the absolute difference between current and reference F1 scores
DIFF=$(echo "scale=3; $REF_F1_SCORE - $CUR_F1_SCORE" | bc)
DIFF=$(echo "${DIFF#-}")  # Get the absolute value

# Threshold for acceptable difference
THRESHOLD=0.001

# Display the results
echo "==========================================================================="
echo "Current F1 Score: $CUR_F1_SCORE"
echo "Reference F1 Score: $REF_F1_SCORE"
echo "Difference: $DIFF"
echo "==========================================================================="
echo "Checking if the difference exceeds the threshold of $THRESHOLD:"

if (( $(echo "$DIFF > $THRESHOLD" | bc -l) )); then
    echo "FAIL: Difference of $DIFF exceeds the allowed threshold."
else
    echo "PASS: Difference of $DIFF is within the allowed threshold."
fi

echo "==========================================================================="

# Save the results to a JSON file
CURRENT_DATE=$(date '+%Y-%m-%d %H:%M:%S')
RES_FILE_PATH="$RES_PATH/bert_f1_score_${CONFIG_DTYPE}.json"

cat <<EOF > "$RES_FILE_PATH"
{
    "date": "$CURRENT_DATE",
    "count": $N_EVAL_DATA,
    "f1_score": $CUR_F1_SCORE,
    "reference_f1_score": $REF_F1_SCORE
}
EOF

echo "Results saved to $RES_FILE_PATH"
echo "Evaluation log saved to $LOG_PATH"

##############################
# Step 5: Cleanup and Exit
##############################

# Unset environment variables
unset LOG_PATH
unset ML_MODEL_FILE_WITH_PATH
unset DATASET_FILE
unset SKIP_VERIFY_ACCURACY

printf "\n============= End of Forward Test for QBERT =============\n"

# Deactivate the Conda environment
conda deactivate

# Return to the git root directory
cd "$GIT_ROOT_DIR"


