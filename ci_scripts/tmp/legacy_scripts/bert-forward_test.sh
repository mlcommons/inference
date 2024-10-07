#!/bin/bash

# define env. variables
model_name=qbert
model_dir=language/bert
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/bert
log_dir=$git_dir/logs
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)

# work on model directory
cd $work_dir

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
printf "\n============= STEP-1: Run calibration =============\n"
SCENARIO=${SCENARIO:="Offline"}
#BACKEND="rngd"

MODEL_PATH=$data_dir/models/bert/model.pytorch
MODEL_CONFIG_PATH=$data_dir/models/bert/bert_config.json
DATASET_PATH=$data_dir/dataset/squad/calibration/cal_features.pickle # use calibration data for forward test
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
N_COUNT=${N_COUNT:="10833"} # total_len = 10,833

# quantization args
CALIB_DATA_PATH=$data_dir/dataset/squad/calibration/cal_features.pickle
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml
N_CALIB=2

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
                                    --gpu                                          


unset LOG_PATH

printf "\n============= End of Forward Test for QBERT =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
