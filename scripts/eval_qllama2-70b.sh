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

# work on model directory
cd $work_dir

# enter existing conda env.
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=${SCENARIO:="Offline"}
BACKEND="rngd"
DATA_TYPE=${DATA_TYPE:="float32"}
N_COUNT=${N_COUNT:="24576"} # total_len = 24,576
DEVICE=${DEVICE:="cuda:0"}

if [ $DEVICE = "cpu" ];
    then DATA_TYPE=float32;
fi
# quantization args
CALIBRATE=${CALIBRATE:=false}
N_CALIB=${N_CALIB:=1000} # total_len=1,000
CALIB_DATA_PATH=$data_dir/dataset/open-orca/calibration/open_orca_gpt4_tokenized_llama.calibration_1000.pkl
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
printf "\tCALIBRATE: $CALIBRATE\n"
printf "\tDEVICE: $DEVICE\n"

CHECKPOINT_PATH=$data_dir/models/llama2/Llama-2-70b-chat-hf
DATASET_PATH=$data_dir/dataset/open-orca/validation/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
LOG_PATH=$log_dir/$model_name/$SCENARIO/W8A8KV8/$(date +%Y%m%d_%H%M%S%Z)

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

if [ "$CALIBRATE" = true ]; then
    printf "\t\tNUM_CALIB_DATA: $N_CALIB\n"
    python -m quantization.calibrate --backend=pytorch \
                                     --model_path=$CHECKPOINT_PATH \
                                     --quant_config_path=$QUANT_CONFIG_PATH \
                                     --quant_param_path=$QUANT_PARAM_PATH \
                                     --quant_format_path=$QUANT_FORMAT_PATH \
                                     --calib_data_path=$CALIB_DATA_PATH \
                                     --n_calib=$N_CALIB \
                                     --model_source $MODEL_SOURCE \
                                     --gpu=True\

fi

SECONDS=0
python -u main.py --scenario $SCENARIO \
                  --backend $BACKEND \
                  --model-path $CHECKPOINT_PATH \
                  --mlperf-conf ../../mlperf.conf \
                  --total-sample-count $N_COUNT \
                  --device $DEVICE \
                  --dataset-path $DATASET_PATH \
                  --dtype $DATA_TYPE \
                  --accuracy \
                  --output-log-dir $LOG_PATH \
                  --quantize \
                  --quant_config_path $QUANT_CONFIG_PATH \
                  --quant_param_path $QUANT_PARAM_PATH \
                  --quant_format_path $QUANT_FORMAT_PATH

duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

ACCURACY_LOG_FILE=$LOG_PATH/mlperf_log_accuracy.json
python evaluate-accuracy.py --checkpoint-path $CHECKPOINT_PATH \
                            --mlperf-accuracy-file $ACCURACY_LOG_FILE \
                            --dataset-file $DATASET_PATH --dtype int64 \
                            &> $LOG_PATH/accuracy_result.log
printf "Save eval log to $LOG_PATH"

printf "\n============= End of eval =============\n"

# unset exported env. variables
unset SCENARIO
unset DATA_TYPE
unset N_COUNT
unset DEVICE

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
