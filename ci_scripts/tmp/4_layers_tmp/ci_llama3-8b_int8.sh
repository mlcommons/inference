#!/bin/bash
# define env. variables
model_name=llama3.1-8b
model_dir=language/llama2-70b
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=/home/home-mcl/phil/actions-runner/_work/data
MODEL_DATA_DIR=/home/home-mcl/phil/actions-runner/_work/data/quantization/llama3-8b/
REF_PATH=$MODEL_DATA_DIR/ref
RES_PATH=/home/home-mcl/phil/actions-runner/_work/inference/inference/language/results
log_dir=$git_dir/logs

CONFIG_DTYPE=int8
# work on model directory
cd $work_dir

# enter existing conda env.
export CONDA_EXE="/anaconda/condabin/conda"
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

printf "\n============= STEP-1: Run calibration =============\n"
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
export CALIBRATE=true
export N_CALIB=1 #test 10 #full 128
export N_DATA=1

CALIB_DATA_PATH=$data_dir/dataset/llama3-1/mgoin_ultrachat_2k_calibration_128.pkl
QUANT_CONFIG_PATH=$MODEL_DATA_DIR/quant_config_$CONFIG_DTYPE.yaml

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_DATA\n"
printf "\tCALIBRATE: $CALIBRATE\n"
printf "\tDEVICE: $DEVICE\n"

CHECKPOINT_PATH=$data_dir/models/llama3/Meta-Llama-3.1-8B-Instruct
DATASET_PATH=$data_dir/dataset/open-orca/validation/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
LOG_PATH=$log_dir/$model_name/$SCENARIO/W8A8KV8/$(date +%Y%m%d_%H%M%S%Z)
SUBMISSION_MODEL_SOURCE="mlperf_submission_slice"

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

if [ "$CALIBRATE" = true ]; then
    QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param.npy
    QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format.yaml
    OUTPUT_PATH=$LOG_PATH/calibration_range/
    printf "\tNUM_CALIB_DATA: $N_CALIB\n"
    python -m quantization.calibrate_llama3\
                                     --model_path=$CHECKPOINT_PATH \
                                     --quant_config_path=$QUANT_CONFIG_PATH \
                                     --quant_param_path=$QUANT_PARAM_PATH \
                                     --quant_format_path=$QUANT_FORMAT_PATH \
                                     --calib_data_path=$CALIB_DATA_PATH \
                                     --n_calib=$N_CALIB \
                                     --submission_model_source=$SUBMISSION_MODEL_SOURCE \
                                     --gpu \
                                     --save_cache_files \
                                     --output_path=$OUTPUT_PATH \
                                     --n_layers=4
fi

GOLDEN_QUANT_PARAM_PATH=$LOG_PATH/calibration_range/quant_param_golden.npy
GOLDEN_QUANT_FORMAT_PATH=$LOG_PATH/calibration_range/quant_format_golden.yaml
LOGIT_FOLDER_PATH=ci_file/logit_files
OUTPUT_FOLDER_PATH=ci_file/output_files
mkdir -p $LOGIT_FOLDER_PATH
mkdir -p $OUTPUT_FOLDER_PATH

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir

