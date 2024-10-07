#!/bin/bash

# define env. variables
model_name=qgpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=/home/home-mcl/phil/actions-runner/_work/data
REF_PATH=/home/home-mcl/phil/actions-runner/_work/data/quantization/gpt-j/ref
RES_PATH=/home/home-mcl/phil/actions-runner/_work/inference/inference/language/results
quant_data_dir=$data_dir/quantization/gpt-j
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
#BACKEND="rngd"

MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)

# quantization args
export CALIBRATE=true
export N_CALIB=1000 #test 50 #full 1000
export N_DATA=10
N_COUNT=${N_COUNT:="10"} # total_len=13,368

CALIB_DATA_PATH=$data_dir/dataset/cnn-daily-mail/calibration/cnn_dailymail_calibration.json
QUANT_CONFIG_PATH=$quant_data_dir/quant_config_$CONFIG_DTYPE.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

printf "<<EVAL_CONFIG>>\n"
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
                                     --gpu
    printf "Save calibration range to $LOG_PATH/calibration_range"
else
    cp $QUANT_PARAM_PATH $LOG_PATH/calibration_range/quant_param.npy
    cp $QUANT_FORMAT_PATH $LOG_PATH/calibration_range/quant_format.yaml
fi


GOLDEN_QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param_golden.npy
GOLDEN_QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format_golden.yaml
LOGIT_FOLDER_PATH=ci_file/logit_files
mkdir -p $LOGIT_FOLDER_PATH

printf "\n============= STEP-2: Check the equivalence of logits obtained at each generation step =============\n"

python -m ci_file.qgpt_j_forward_test          --model_path=$MODEL_PATH \
                                                --quant_config_path=$QUANT_CONFIG_PATH \
                                                --golden_quant_param_path=$GOLDEN_QUANT_PARAM_PATH \
                                                --golden_quant_format_path=$GOLDEN_QUANT_FORMAT_PATH \
                                                --submission_quant_format_path=$QUANT_FORMAT_PATH \
                                                --submission_quant_param_path=$QUANT_PARAM_PATH \
                                                --n_data=$N_DATA \
                                                --dataset_path=$DATASET_PATH \
                                                --logit_folder_path=$LOGIT_FOLDER_PATH \
                                                --gpu \
                                                --ref_path=$REF_PATH\
                                                --res_path=$RES_PATH\
                                                --config_dtype=$CONFIG_DTYPE\
                                                # --update_gen_list #정답지 업데이트용 argument

unset LOG_PATH
unset CALIBRATE
unset N_CALIB
unset N_DATA

printf "\n============= End of Forward Test for QGPT-J =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
