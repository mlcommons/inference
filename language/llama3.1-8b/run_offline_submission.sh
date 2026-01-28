#!/bin/bash
#Example usage ./run_offline_submission.sh L40S meta-llama/Llama-3.1-8B-Instruct/ cnn_eval.json  TESTING3

GPU=$1 			#Options are L40S/H200/H100
MODEL_DIR=$2
DATASET_PATH=$3
PREFIX=$4   #Specify a descriptive name to describe your submission


USER_CONF="user.conf"
if [[ "${GPU}" == "H100" ]];then 
   USER_CONF="h100_user.conf"
fi

if [[ "${GPU}" == "L40S" ]];then 
   export VLLM_ATTENTION_BACKEND=FLASHINFER
   export TORCH_CUDA_ARCH_LIST=8.9
fi


echo "Using user-conf ${USER_CONF}"


SUBMISSION_DIR="MLPERF-SUBMISSIONS/${GPU}/LLAMA3.1-8B/${PREFIX}/Offline"

#Create the performance ,accuracy and compliance directories

PERF_DIR="${SUBMISSION_DIR}/performance/run_1"
ACCURACY_DIR="${SUBMISSION_DIR}/accuracy/"
COMPLIANCE_DIR="${SUBMISSION_DIR}/compliance/"

mkdir -p ${PERF_DIR}
mkdir -p ${ACCURACY_DIR}
mkdir -p ${COMPLIANCE_DIR}

REQUIRED_CONFIGS=" --max-model-len 131072 "
PERF_CONFIGS=" --max-num-seqs 1024 --kv-cache-dtype fp8 "
PERF_CONFIGS+=" --max-num-batched-tokens 4096 "
HARNESS_CONFIGS=" --batch-size 40104"

AUDIT_CONFIG=" --audit-conf ../../compliance/nvidia/TEST06/audit.config"

echo "Run compliance"
FILENAME="offline_compliance_${GPU}_llama3.18b.log"
python3 SUT_VLLM_SingleReplica.py --model ${MODEL_DIR} --dataset_path ${DATASET_PATH} \
        --user-conf user_conf.conf  --test-mode performance --output-log-dir ${COMPLIANCE_DIR}\
        ${REQUIRED_CONFIGS} ${PERF_CONFIGS} ${HARNESS_CONFIGS}\
        ${AUDIT_CONFIG}\
         >& ${COMPLIANCE_DIR}/${FILENAME}

echo "Checking compliance"
python3 ../../compliance/nvidia/TEST06/run_verification.py -c ${COMPLIANCE_DIR} -o ${COMPLIANCE_DIR} -s Offline

echo "Run accuracy"
FILENAME="offline_accuracy_${GPU}_llama3.18b.log"
python3 SUT_VLLM_SingleReplica.py --model ${MODEL_DIR} --dataset_path ${DATASET_PATH} \
        --user-conf user.conf  --test-mode accuracy --output-log-dir ${ACCURACY_DIR}\
        ${REQUIRED_CONFIGS} ${PERF_CONFIGS} ${HARNESS_CONFIGS}\
         >& ${ACCURACY_DIR}/${FILENAME}
echo "Accuracy run completed"
echo "Evaluating Accuracy "
python3 evaluation.py  --mlperf-accuracy-file ${ACCURACY_DIR}/mlperf_log_accuracy.json  --dataset-file ${DATASET_PATH}  --dtype int32 >& ${ACCURACY_DIR}/accuracy.txt 
sed -i -E 's/np\.int64\(([0-9]+)\)/\1/g' ${ACCURACY_DIR}/accuracy.txt
cat ${ACCURACY_DIR}/accuracy.txt

echo "Run performance"
FILENAME="offline_performance_${GPU}_llama3.18b.log"
python3 SUT_VLLM_SingleReplica.py --model ${MODEL_DIR} --dataset_path ${DATASET_PATH} \
        --user-conf ${USER_CONF}  --test-mode performance --output-log-dir ${PERF_DIR}\
        ${REQUIRED_CONFIGS} ${PERF_CONFIGS} ${HARNESS_CONFIGS}\
         >& ${PERF_DIR}/${FILENAME}
grep "Tokens per second" ${PERF_DIR}/${FILENAME}

