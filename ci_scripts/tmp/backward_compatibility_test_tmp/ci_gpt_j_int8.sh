#!/bin/bash

# define env. variables
model_name=qgpt-j
model_dir=language/gpt-j
git_dir=$(git rev-parse --show-toplevel)
work_dir=$git_dir/$model_dir
data_dir=/home/home-mcl/phil/actions-runner/_work/data
REF_PATH=/home/home-mcl/phil/actions-runner/_work/data/quantization/gpt-j/ref
RES_PATH=/home/home-mcl/phil/actions-runner/_work/inference/inference/language/results
MODEL_DATA_DIR=$data_dir/furiosa_llm_modles_artifacts/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8A8KV8/28L
quant_data_dir=$data_dir/quantization/gpt-j
log_dir=$git_dir/logs
env_name=mlperf-$model_name
CONFIG_DTYPE=int8
QUANT_CONFIG_PATH=$quant_data_dir/quant_config_$CONFIG_DTYPE.yaml
# work on model directory
cd $work_dir

# enter existing conda env.
export CONDA_EXE="/anaconda/condabin/conda"
conda_base=$($CONDA_EXE info --base)
source "$conda_base/etc/profile.d/conda.sh"
conda activate inference-ci

# eval model
SCENARIO=${SCENARIO:="Offline"}
#BACKEND="rngd"

MODEL_PATH=$data_dir/models/gpt-j
DATASET_PATH=$data_dir/dataset/cnn-daily-mail/validation/cnn_eval.json
LOG_PATH=$log_dir/$model_name/$SCENARIO/$(date +%Y%m%d_%H%M%S%Z)

# quantization args
export CALIBRATE=true
export N_DATA=3

printf "<<EVAL_CONFIG>>\n"
printf "\tSCENARIO: $SCENARIO\n"
printf "\tCALIBRATE: $CALIBRATE\n"

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

printf "\n============= STEP-1: pull dvc =============\n"
mkdir -p $LOGIT_FOLDER_PATH

# TAG=main
# cd $git_dir
# git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
# cd $git_dir/furiosa-llm-models-artifacts
# git checkout $TAG

# quant_data_dvc_dir=/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8fA8fKV8f/28L
# dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml.dvc -r origin --force
# dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy.dvc -r origin --force


# cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml $MODEL_DATA_DIR/quant_format.yaml
# cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy $MODEL_DATA_DIR/quant_param.npy

# rm -rf $git_dir/furiosa-llm-models-artifacts

QUANT_FORMAT_PATH=$MODEL_DATA_DIR/quant_format.yaml
QUANT_PARAM_PATH=$MODEL_DATA_DIR/quant_param.npy
LOGIT_FOLDER_PATH=$work_dir/ci_file/logit_files

printf "\n============= STEP-2: Check the equivalence of logits obtained at each generation step =============\n"

python -m ci_file.backward_compatibility_test_qgpt_j_forward_test\
                                                --model_path=$MODEL_PATH \
                                                --quant_config_path=$QUANT_CONFIG_PATH \
                                                --submission_quant_format_path=$QUANT_FORMAT_PATH \
                                                --submission_quant_param_path=$QUANT_PARAM_PATH \
                                                --n_data=$N_DATA \
                                                --dataset_path=$DATASET_PATH \
                                                --logit_folder_path=$LOGIT_FOLDER_PATH \
                                                --gpu \
                                                --ref_path=$REF_PATH \
                                                --res_path=$RES_PATH \
                                                --config_dtype=$CONFIG_DTYPE

unset LOG_PATH
unset CALIBRATE
unset N_CALIB
unset N_DATA

printf "\n============= End of Forward Test for QGPT-J =============\n"

# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
