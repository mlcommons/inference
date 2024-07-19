#!/bin/bash

# Define environment variables
model_name=qbert
model_dir=language/bert
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=$git_dir/data
quant_data_dir=$data_dir/quantization/bert
log_dir=$git_dir/logs
env_name=mlperf-$model_name
conda_base=$($CONDA_EXE info --base)

# Enter existing conda environment
source "$conda_base/etc/profile.d/conda.sh"
conda activate $env_name

# Eval model
printf "\n============= STEP-4: Run eval =============\n"
SCENARIO=${SCENARIO:=Offline}
BACKEND="rngd"
MODEL_PATH=$data_dir/models/bert/model.pytorch
MODEL_CONFIG_PATH=$data_dir/models/bert/bert_config.json
VOCAB_PATH=$data_dir/models/bert/vocab.txt
DATASET_PATH=$data_dir/dataset/squad/validation/dev-v1.1.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)
MLPERF_CONF=$git_dir/mlperf.conf
DUMP_PATH=none
SKIP_VERIFY_ACCURACY=true
N_COUNT=${N_COUNT:="10833"} # total_len = 10,833

# Quantization args
QUANT_CONFIG_PATH=$quant_data_dir/quant_config.yaml
QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param.npy
QUANT_FORMAT_PATH=$quant_data_dir/calibration_range/quant_format.yaml

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tNUM_EVAL_DATA: $N_COUNT\n"
if [ "$DO_DUMP" = true ]; then
printf "\tDO_DUMP: true\n"
fi

export LOG_PATH=$LOG_PATH
export ML_MODEL_FILE_WITH_PATH=$MODEL_PATH
export VOCAB_FILE=$VOCAB_PATH
export DATASET_FILE=$DATASET_PATH
export SKIP_VERIFY_ACCURACY=$SKIP_VERIFY_ACCURACY

if [ "$DO_DUMP" = true ]; then
    DUMP_PATH="$LOG_PATH/encoder_dump_n$N_COUNT.json"
fi

mkdir -p $LOG_PATH

SECONDS=0
python $work_dir/run.py --scenario=$SCENARIO \
              --backend=$BACKEND \
              --mlperf_conf=$MLPERF_CONF \
              --gpu \
              --quantize \
              --quant_param_path=$QUANT_PARAM_PATH \
              --quant_format_path=$QUANT_FORMAT_PATH \
              --max_examples=$N_COUNT \
              --accuracy \
              --dump_path=$DUMP_PATH
duration=$SECONDS
printf "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed." &> $LOG_PATH/elapsed_time.log

ACCURACY_LOG_FILE=$LOG_PATH/mlperf_log_accuracy.json
python $work_dir/accuracy-squad.py --vocab_file=$VOCAB_PATH \
                         --val_data=$DATASET_PATH \
                         --log_file=$ACCURACY_LOG_FILE \
                         --out_file=$LOG_PATH/predictions.json \
                         --max_examples=$N_COUNT \
                         &> $LOG_PATH/accuracy_result.log

printf "Save evaluation log to $LOG_PATH"

unset LOG_PATH
unset ML_MODEL_FILE_WITH_PATH
unset VOCAB_FILE
unset DATASET_FILE

printf "\n============= End of eval =============\n"

# Exit from conda environment
conda deactivate
