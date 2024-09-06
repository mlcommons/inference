#!/bin/bash

# define env. variables
model_name=qbert
model_dir=language/bert
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=/home/home-mcl/phil/actions-runner/_work/data
REF_PATH=/home/home-mcl/phil/actions-runner/_work/data/quantization/bert/ref
RES_PATH=/home/home-mcl/phil/actions-runner/_work/inference/inference/language/results
quant_data_dir=$data_dir/quantization/bert
log_dir=$git_dir/logs
env_name=mlperf-$model_name
CONFIG_DTYPE=fp8
# work on model directory
cd $work_dir

# enter existing conda env.
export CONDA_EXE="/anaconda/condabin/conda"
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

# eval model
printf "\n============= STEP-1: Run calibration =============\n"
SCENARIO=${SCENARIO:="Offline"}
BACKEND="rngd"

MODEL_PATH=$data_dir/models/bert/model.pytorch
MODEL_CONFIG_PATH=$data_dir/models/bert/bert_config.json
DATASET_PATH=$data_dir/dataset/squad/calibration/cal_features.pickle # use calibration data for forward test
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=${N_COUNT:="100"} # total_len = 10,833

# quantization argsㅜ
CALIB_DATA_PATH=$data_dir/dataset/squad/calibration/cal_features.pickle
QUANT_CONFIG_PATH=$quant_data_dir/quant_config_$CONFIG_DTYPE.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml
N_CALIB=${N_CALIB:=100} # total_len = 100

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

printf "\tNUM_CALIB_DATA: $N_CALIB\n"
GOLDEN_QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param_golden.npy
GOLDEN_QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format_golden.yaml
python -m quantization.calibrate --model_type="golden" \
                                    --model_path=$MODEL_PATH \
                                    --model_config_path=$MODEL_CONFIG_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_param_path=$GOLDEN_QUANT_PARAM_PATH \
                                    --quant_format_path=$GOLDEN_QUANT_FORMAT_PATH \
                                    --calib_data_path=$CALIB_DATA_PATH \
                                    --n_calib=$N_CALIB \
                                    --gpu
printf "Save golden calibration files to $GOLDEN_QUANT_PARAM_PATH and $GOLDEN_QUANT_FORMAT_PATH"

QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format.yaml
python -m quantization.calibrate --model_type="mlperf-submission" \
                                    --model_path=$MODEL_PATH \
                                    --model_config_path=$MODEL_CONFIG_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_param_path=$QUANT_PARAM_PATH \
                                    --quant_format_path=$QUANT_FORMAT_PATH \
                                    --calib_data_path=$CALIB_DATA_PATH \
                                    --n_calib=$N_CALIB \
                                    --gpu \
                                    --is_equivalence_ci
printf "Save submission calibration files to $QUANT_PARAM_PATH and $QUANT_FORMAT_PATH"

N_DATA=1
LOGIT_FOLDER_PATH=$work_dir/ci_file/logit_files
mkdir -p $LOGIT_FOLDER_PATH

printf "\n============= STEP-2: Check the equivalence of logits obtained at each generation step =============\n"

python -m ci_file.qbert_forward_test --model_path=$MODEL_PATH \
                                    --model_config_path=$MODEL_CONFIG_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --golden_quant_param_path=$GOLDEN_QUANT_PARAM_PATH \
                                    --golden_quant_format_path=$GOLDEN_QUANT_FORMAT_PATH \
                                    --submission_quant_format_path=$QUANT_FORMAT_PATH \
                                    --submission_quant_param_path=$QUANT_PARAM_PATH \
                                    --n_data=$N_DATA \
                                    --dataset_path=$DATASET_PATH \
                                    --logit_folder_path=$LOGIT_FOLDER_PATH \
                                    --gpu \
                                    --ref_path=$REF_PATH \
                                    --res_path=$RES_PATH \
                                    --config_dtype=$CONFIG_DTYPE\
                                    --update_gen_list #정답지 업데이트용 argument


printf "\n============= STEP-3: Check the equivalence of f1 score between current mlperf submission <-> ref =============\n"
export ML_MODEL_FILE_WITH_PATH=$MODEL_PATH
export LOG_PATH=$LOG_PATH
VOCAB_PATH=$data_dir/models/bert/vocab.txt
DATASET_PATH=$data_dir/dataset/squad/validation/dev-v1.1.json
export VOCAB_FILE=$VOCAB_PATH
export DATASET_FILE=$DATASET_PATH
export SKIP_VERIFY_ACCURACY=true

SECONDS=0
python -m run --scenario=$SCENARIO \
              --backend=$BACKEND \
              --gpu \
              --quantize \
              --quant_param_path=$QUANT_PARAM_PATH \
              --quant_format_path=$QUANT_FORMAT_PATH \
              --max_examples=$N_COUNT \
              --accuracy
duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

ACCURACY_LOG_FILE=$LOG_PATH/mlperf_log_accuracy.json
python accuracy-squad.py --vocab_file=$VOCAB_PATH \
                         --val_data=$DATASET_PATH \
                         --log_file=$ACCURACY_LOG_FILE \
                         --out_file=$LOG_PATH/predictions.json \
                         --max_examples=$N_COUNT \
                         &> $LOG_PATH/accuracy_result_$CONFIG_DTYPE.log

CUR_F1_SCORE=$(grep -oP '"f1":\s*\K[0-9.]+' "$LOG_PATH/accuracy_result_$CONFIG_DTYPE.log")
REF_F1_SCORE=95.46667

# f1 score 비교: Ref <-> submission model 

DIFF=$(echo "scale=3; $REF_F1_SCORE - $CUR_F1_SCORE" | bc)
# 절대값 계산 (소수점 이하 셋째 자리까지)
DIFF=$(echo "${DIFF#-}" | bc)  # 음수를 제거하여 절대값을 얻음

# 오차가 소수점 셋째 자리까지 차이가 있는지 확인
echo "==========================================================================="
echo "The current F1 score is: $CUR_F1_SCORE"
echo "Reference F1 score is: $REF_F1_SCORE"
echo "==========================================================================="
echo "Check the equivalence of f1 score between current mlperf submission:"
if (( $(echo "$DIFF > 0.001" | bc -l) )); then
    echo "FAIL: Cur F1 Score <-> Ref F1 Score diff : $DIFF, diff 허용수치를 넘었습니다."
else
    echo "PASS: Cur F1 Score <-> Ref F1 Score diff : $DIFF, diff 허용수치 이내 입니다."
fi
echo "==========================================================================="

current_date=$(date '+%Y-%m-%d %H:%M:%S')
res_file_path=$RES_PATH/bert_f1_score_$CONFIG_DTYPE.json
cat <<EOF > "$res_file_path"
{
    "date": "$current_date",
    "count": $N_COUNT,
    "f1_score": $CUR_F1_SCORE,
    "reference_f1_score": $REF_F1_SCORE,
}
EOF

echo "Results saved to $res_file_path"
printf "Save evaluation log to $LOG_PATH"

unset LOG_PATH
unset ML_MODEL_FILE_WITH_PATH
unset VOCAB_FILE
unset DATASET_FILE
unset SKIP_VERIFY_ACCURACY

printf "\n============= End of Forward Test for QBERT =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
