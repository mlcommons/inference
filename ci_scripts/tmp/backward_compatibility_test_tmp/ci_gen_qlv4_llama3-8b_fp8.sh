#!/bin/bash
model_dir=language/llama2-70b
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=/home/home-mcl/phil/actions-runner/_work/data
MODEL_DATA_DIR=/home/home-mcl/phil/actions-runner/_work/data/quantization/llama3-8b/
log_dir=$git_dir/logs
CONFIG_DTYPE=fp8
QUANT_CONFIG_PATH=$MODEL_DATA_DIR/quant_config_$CONFIG_DTYPE.yaml
QUANT_DATA_PATH=$data_dir/furiosa_llm_modles_artifacts/quantized/meta-llama/Meta-Llama-3.1-8B-Instruct/mlperf_submission_slice/W8fA8fKV8f/32L
MODEL_PATH=$data_dir/models/llama3/Meta-Llama-3.1-8B-Instruct
# work on model directory
cd $work_dir

# enter existing conda env.
export CONDA_EXE="/anaconda/condabin/conda"
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

# eval model
SCENARIO=${SCENARIO:="Offline"}

# quantization args
export CALIBRATE=true

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH

printf "\n============= QLV4 Load TEST =============\n"

python -m ci_file.qllama3_load_test  --model_path=$MODEL_PATH \
                                    --quant_config_path=$QUANT_CONFIG_PATH \
                                    --quant_data_path=$QUANT_DATA_PATH


printf "\n============= End of Test for llama3.1 =============\n"

# unset exported env. variables
unset SCENARIO
unset LOG_PATH
unset CALIBRATE
unset N_CALIB

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir

