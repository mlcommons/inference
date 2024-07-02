#!/bin/bash

# define env. variables
model_name=llama2-70b
model_dir=language/llama2-70b
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
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
DATA_TYPE=${DATA_TYPE:="float32"}
N_COUNT=${N_COUNT:="24576"} # total_len = 24,576
DEVICE=${DEVICE:="cuda:0"}

if [ $DEVICE = "cpu" ];
    then DATA_TYPE=float32;
fi

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tDATA_TYPE: $DATA_TYPE\n"
printf "\tNUM_DATA: $N_COUNT\n"
printf "\tDEVICE: $DEVICE\n"

if ((${N_COUNT} < 2000)); 
    then USER_CONF=$git_dir/internal_test.conf;
else
    USER_CONF=user.conf;
fi

CHECKPOINT_PATH=$data_dir/models/llama2/Llama-2-70b-chat-hf
DATASET_PATH=$data_dir/dataset/open-orca/validation/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
LOG_PATH=$log_dir/$model_name/$SCENARIO/$DATA_TYPE/$(date +%Y%m%d_%H%M%S%Z)

SECONDS=0
python -u main.py --scenario Offline \
                  --model-path $CHECKPOINT_PATH \
                  --mlperf-conf ../../mlperf.conf \
                  --user-conf $USER_CONF \
                  --total-sample-count $N_COUNT \
                  --device $DEVICE \
                  --dataset-path $DATASET_PATH \
                  --dtype $DATA_TYPE \
                  --accuracy \
                  --output-log-dir $LOG_PATH
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
