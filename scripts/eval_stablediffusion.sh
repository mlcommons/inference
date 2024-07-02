#!/bin/bash

# define env. variables
model_name=stablediffusion
model_dir=text_to_image
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
DATA_TYPE=${DATA_TYPE:="fp32"}
N_COUNT=${N_COUNT:="5000"} # total_len = 5,000
DEVICE=${DEVICE:="cuda"}

if [ $DEVICE = "cpu" ];
    then DATA_TYPE=fp32;
fi

if ((${N_COUNT} < 5000)); 
    then USER_CONF=$git_dir/internal_test.conf;
else
    USER_CONF=user.conf;
fi

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tDATA_TYPE: $DATA_TYPE\n"
printf "\tNUM_DATA: $N_COUNT\n"
printf "\tDEVICE: $DEVICE\n"

MODEL_PATH=$data_dir/models/stablediffusion/fp32
DATASET_PATH=$data_dir/dataset/coco2014/validation
LOG_PATH=$log_dir/$model_name/$SCENARIO/$DATA_TYPE/$(date +%Y%m%d_%H%M%S%Z)

SECONDS=0
python main.py --scenario $SCENARIO \
               --dataset "coco-1024" --profile stable-diffusion-xl-pytorch \
               --dataset-path $DATASET_PATH --model-path $MODEL_PATH --dtype $DATA_TYPE \
               --device $DEVICE \
               --mlperf_conf ../mlperf.conf \
               --user_conf $USER_CONF \
               --count $N_COUNT \
               --accuracy \
               --output $LOG_PATH
duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

python -m json.tool $LOG_PATH/results.json &> $LOG_PATH/accuracy_result.log
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
