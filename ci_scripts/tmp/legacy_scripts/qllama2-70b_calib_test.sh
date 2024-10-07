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
export CALIBRATE=true
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
printf "\tNUM_CALIB_DATA: $N_CALIB\n"


CHECKPOINT_PATH=$data_dir/models/llama2/Llama-2-70b-chat-hf
DATASET_PATH=$data_dir/dataset/open-orca/validation/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
LOG_PATH=$log_dir/$model_name/$SCENARIO/W8A8KV8/$(date +%Y%m%d_%H%M%S%Z)

export LOG_PATH

mkdir -p $LOG_PATH/calibration_range

printf "\n============= STEP-1: Pull dvc data =============\n"
pip install dvc[s3]
dvc pull $data_dir/quantization/llama2-70b.dvc --force
cd $git_dir
git clone https://github.com/furiosa-ai/furiosa-llm-models-artifacts.git
cd $git_dir/furiosa-llm-models-artifacts
git checkout $tag

dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml.dvc -r origin --force
dvc pull $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy.dvc -r origin --force

mkdir -p $quant_data_dir/calibration_range


cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qparam.npy $quant_data_dir/calibration_range/quant_param_from_dvc.npy
cp $git_dir/furiosa-llm-models-artifacts/$quant_data_dvc_dir/qformat.yaml $quant_data_dir/calibration_range/quant_format_from_dvc.yaml
rm -rf $git_dir/furiosa-llm-models-artifacts

RELEASED_QUANT_PARAM_PATH=$quant_data_dir/calibration_range/quant_param_from_dvc.npy
SUBMISSION_MODEL_SOURCE="mlperf_submission_slice"

printf "\n============= STEP-1: Run calibration =============\n"
cd $work_dir    
if [ "$CALIBRATE" = true ]; then
    python -m quantization.calibrate --model_path=$CHECKPOINT_PATH \
                                     --quant_config_path=$QUANT_CONFIG_PATH \
                                     --quant_param_path=$QUANT_PARAM_PATH \
                                     --quant_format_path=$QUANT_FORMAT_PATH \
                                     --calib_data_path=$CALIB_DATA_PATH \
                                     --n_calib=$N_CALIB \
                                     --submission_model_source=$SUBMISSION_MODEL_SOURCE \
                                     --gpu

fi



printf "\n============= STEP-3: Check the equivalence of quantiation parameters =============\n"
printf "Comparing the two quant param files at $RELEASED_QUANT_PARAM_PATH and $QUANT_PARAM_PATH\n"
python ci_file/utils/check_qparam_equivalence.py --released_quant_param_path=$RELEASED_QUANT_PARAM_PATH \
                                    --created_quant_param_path=$QUANT_PARAM_PATH\

# unset exported env. variables

unset CALIBRATE
unset LOG_PATH






# exit from conda env.
conda deactivate

# get back to git root
cd $git_dir
