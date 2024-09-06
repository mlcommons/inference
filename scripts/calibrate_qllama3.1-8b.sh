#!/bin/bash

# define env. variables
model_name=qllama2-70b
model_dir=language/llama2-70b
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/llama2-70b
log_dir=$git_dir/logs
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)
quant_data_dir=$data_dir/quantization/llama2-70b
tag=MLPerf4.1-v3.13
quant_data_dvc_dir=quantized/LLaMA2-70B/mlperf_submission/W8A8KV8/80L


# work on model directory
cd $work_dir

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
SCENARIO=${SCENARIO:="Offline"}
BACKEND="rngd"
DATA_TYPE=${DATA_TYPE:="float32"}
N_COUNT=${N_COUNT:="24576"} # total_len = 24,576
DEVICE=${DEVICE:="cuda:0"}

if [ $DEVICE = "cpu" ];
    then DATA_TYPE=float32;
fi
# quantization args

#N_CALIB=${N_CALIB:=1000} # total_len=1,000

export N_CALIB=10 #test code
CALIB_DATA_PATH='/home/home-mcl/sunghyuck/inference/mgoin_ultrachat_2k_calibration_128.pkl'
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
printf "\tCALIBRATE: $CALIBRATE\n"
printf "\tDEVICE: $DEVICE\n"
printf "\tNUM_CALIB_DATA: $N_CALIB\n"


CHECKPOINT_PATH=$data_dir/models/llama3/Meta-Llama-3.1-8B-Instruct
DATASET_PATH=$data_dir/dataset/open-orca/validation/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
LOG_PATH=$log_dir/qllama3.1-8b/$SCENARIO/W8A8KV8/$(date +%Y%m%d_%H%M%S%Z)
SUBMISSION_MODEL_SOURCE="mlperf_submission_slice"

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range


printf "\n============= STEP-1: Run calibration =============\n"
# work on model directory
cd $work_dir
QUANT_PARAM_PATH=$LOG_PATH/calibration_range/llama3.1-8B-quant_param.npy
QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/llama3.1-8B-quant_format.yaml

python -m quantization.calibrate_llama3 --model_path=$CHECKPOINT_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_param_path=$QUANT_PARAM_PATH \
                                    --quant_format_path=$QUANT_FORMAT_PATH \
                                    --calib_data_path=$CALIB_DATA_PATH \
                                    --n_calib=$N_CALIB \
                                    --submission_model_source=$SUBMISSION_MODEL_SOURCE \
                                    --gpu \
                                    --save_cache_files

printf "Save calibration range to $LOG_PATH/calibration_range"

printf "\n============= End of calibration =============\n"



# unset exported env. variables
unset SCENARIO
unset DATA_TYPE
unset N_COUNT
unset DEVICE
unset N_CALIB
unset LOG_PATH




# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
