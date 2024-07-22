#!/bin/bash

# define env. variables
model_name=qgpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/gpt-j
log_dir=$git_dir/logs
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)
tag=MLPerf4.1-v3.13.2
quant_data_dvc_dir=quantized/GPT-J/mlperf_submission/W8A8KV8/28L

printf "\n============= STEP-1: Pull dvc data =============\n"
cd $git_dir
git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
cd $git_dir/furiosa-llm-models-artifacts
git checkout $tag

dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml.dvc -r origin --force
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy.dvc -r origin --force

mkdir -p $quant_data_dir/calibration_range
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml $quant_data_dir/calibration_range/quant_format.yaml
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy $quant_data_dir/calibration_range/quant_param.npy
rm -rf $git_dir/furiosa-llm-models-artifacts


RELEASED_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy


# work on model directory
cd $work_dir

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
printf "\n============= STEP-2: Run calibration =============\n"
SCENARIO=${SCENARIO:="Offline"}
#BACKEND="rngd"
MODEL_TYPE="golden"
MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=${N_COUNT:="13368"} # total_len=13,368

# quantization args
export CALIBRATE=true
N_CALIB=${N_CALIB:=1000} # total_len=1,000
CALIB_DATA_PATH=$data_dir/dataset/cnn-daily-mail/calibration/cnn_dailymail_calibration.json
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
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





printf "\n============= STEP-3: Check the equivalence of quantiation parameters =============\n"

python ci_file/utils/check_qparam_equivalence.py --released_quant_param_path=$RELEASED_PARAM_PATH \
                                    --created_quant_param_path=$QUANT_PARAM_PATH\

                                            


unset LOG_PATH
unset CALIBRATE

printf "\n============= End of calibration =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
