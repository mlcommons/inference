#!/bin/bash

# define env. variables
model_name=qgpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=/home/home-mcl/phil/actions-runner/_work/data
quant_data_dir=$data_dir/quantization/gpt-j
log_dir=$git_dir/logs
env_name=mlperf-$model_name
tag=MLPerf4.1-v3.13.2

printf "\n============= STEP-0: Build libs =============\n"
if ! pip show accelerate > /dev/null 2>&1; then
    echo "accelerate is not installed. Installing..."
    pip install accelerate==0.30.1
fi

if ! pip show pybind11 > /dev/null 2>&1; then
    echo "pybind11 is not installed. Installing..."
    pip install pybind11==2.11.1
fi
cd $git_dir/loadgen; python setup.py install

printf "\n============= STEP-1: Pull dvc data =============\n"
echo "Git-Action: CI는 이미 설치된 dvc /home/home-mcl/phil/actions-runner/_work/data 경로의 data들을 사용합니다."
RELEASED_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
cd $work_dir # work on model directory

printf "\n============= STEP-2: Run calibration =============\n"
SCENARIO=${SCENARIO:="Offline"}
MODEL_TYPE="golden"
MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=${N_COUNT:="13368"} # total_len=13,368

# quantization args
export CALIBRATE=true
N_CALIB=${N_CALIB:=10} # total_len=1,000
CALIB_DATA_PATH=$data_dir/dataset/cnn-daily-mail/calibration/cnn_dailymail_calibration.json
QUANT_CONFIG_PATH=$quant_data_dir/quant_config_int8.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

printf "<<CALIB_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

if [ "$CALIBRATE" = true ]; then
    printf "\tNUM_CALIB_DATA: $N_CALIB\n"
    QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param.npy
    QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format.yaml
    python -m quantization.calibrate --model_path=$MODEL_PATH \
                                     --quant_config_path=$QUANT_CONFIG_PATH \
                                     --quant_param_path=$QUANT_PARAM_PATH \
                                     --quant_format_path=$QUANT_FORMAT_PATH \
                                     --calib_data_path=$CALIB_DATA_PATH \
                                     --n_calib=$N_CALIB \
                                     --gpu \
                                     --save_cache_files
    printf "Save calibration range to $LOG_PATH/calibration_range"
else
    cp $QUANT_PARAM_PATH $LOG_PATH/calibration_range/quant_param.npy
    cp $QUANT_FORMAT_PATH $LOG_PATH/calibration_range/quant_format.yaml
fi

# printf "\n============= STEP-3: Check the equivalence of quantiation parameters =============\n"

# python ci_file/utils/check_qparam_equivalence.py --released_quant_param_path=$RELEASED_PARAM_PATH \
#                                     --created_quant_param_path=$QUANT_PARAM_PATH\



# unset LOG_PATH
# unset CALIBRATE
printf "\n============= End of calibration =============\n"


# get back to git root
cd $git_dir

