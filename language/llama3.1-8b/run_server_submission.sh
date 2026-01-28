#!/bin/bash

 #bash run_server_submission.sh H100  <MODEL-PATH> <DATASET-PATH>  TEST 38 auto accuracy  --max-model-len 131072 --disable-log-requests --max-num-seqs 1024 --max-num-batched-tokens 4096 
 #you have to run the script multiple times with options of compliance, accuracy and performance

# === GLOBALS ===
VLLM_PID=""
VLLM_PGID=""
GPU=$1
MODEL_DIR=$2
DATASET_PATH=$3
PREFIX=$4
TARGET_QPS=$5
KV_CACHE_DTYPE=$6
CHECK=$7
shift 7
CMD="$*"
echo ${CMD}



if [[ "${GPU}" == "H100" ]];then 
   export TORCH_CUDA_ARCH_LIST="9.0"
fi
if [[ "${GPU}" == "L40S" ]];then 
   export TORCH_CUDA_ARCH_LIST="8.9"
   if [[ "${KV_CACHE_DTYPE}" == "fp8" ]];then
        export TORCH_CUDA_ARCH_LIST="8.9"
        export VLLM_ATTENTION_BACKEND=FLASHINFER
   fi
fi
# === Function: Launch vLLM server ===

start_vllm() {
    local PREFIX=$1
    local GPU=$2
    local MODEL_DIR=$3
    shift 3
    local CMD="$@"

    echo "🔹 Starting vLLM run for [$PREFIX] on GPU $GPU"
    echo "🔹 Starting vLLM run for [$PREFIX] cmd $CMD"

    export CUDA_VISIBLE_DEVICES="$GPU"

    # Launch in background
    vllm serve ${MODEL_DIR} $CMD > "${PREFIX}_vllm.log" 2>&1 &
    VLLM_PID=$!
    sleep 1

    # Get process group ID
    VLLM_PGID=$(ps -o pgid= $VLLM_PID | grep -o '[0-9]*')

    echo "vLLM PID: $VLLM_PID"
    echo "PGID: $VLLM_PGID"
}

# === Function: Kill vLLM server ===
stop_vllm() {
    if [[ -n "$VLLM_PGID" ]]; then
        echo "Stopping vLLM PGID: $VLLM_PGID"
        kill -9 -"$VLLM_PGID"
        wait "$VLLM_PID" 2>/dev/null
        echo "vLLM stopped."
    else
        echo " No PGID available to stop."
    fi
}

SUBMISSION_DIR="MLPERF-SUBMISSIONS/${GPU}/LLAMA3.1-8B/${PREFIX}/Server"

#Create the performance ,accuracy and compliance directories

PERF_DIR="${SUBMISSION_DIR}/performance/run_1"
ACCURACY_DIR="${SUBMISSION_DIR}/accuracy/"
COMPLIANCE_DIR="${SUBMISSION_DIR}/compliance/"

mkdir -p ${PERF_DIR}
mkdir -p ${ACCURACY_DIR}
mkdir -p ${COMPLIANCE_DIR}
#Make the submission directories


if [[ "${CHECK}" == "compliance" ]];then 
start_vllm compliance 0 ${MODEL_DIR} ${CMD}  
sleep  50
AUDIT_CONFIG=" --audit-conf ../../compliance/nvidia/TEST06/audit.config"

echo "Run compliance"
FILENAME="offline_compliance_${GPU}_llama3.18b.log"
python3 SUT_VLLM_SingleReplica_Server.py --model-name ${MODEL_DIR} --dataset-path ${DATASET_PATH} \
        --user-conf user.conf  --test-mode performance --target-qps ${TARGET_QPS} --output-log-dir ${COMPLIANCE_DIR}\
        --api-server-url http://localhost:8000\
        ${AUDIT_CONFIG}\
         >& ${COMPLIANCE_DIR}/${FILENAME}

echo "Checking compliance"
python3 ../../compliance/nvidia/TEST06/run_verification.py -c ${COMPLIANCE_DIR} -o ${COMPLIANCE_DIR} -s Server

stop_vllm
sleep 30
fi

if [[ "${CHECK}" == "accuracy" ]];then 
start_vllm accuracy 0 ${MODEL_DIR} ${CMD}  
sleep  50
echo "Run accuracy"
FILENAME="offline_accuracy_${GPU}_llama3.18b.log"
python3 SUT_VLLM_SingleReplica_Server.py --model-name ${MODEL_DIR} --dataset-path ${DATASET_PATH} \
        --user-conf user.conf  --test-mode accuracy --target-qps ${TARGET_QPS} --output-log-dir ${ACCURACY_DIR}\
        --api-server-url http://localhost:8000\
        >& ${ACCURACY_DIR}/${FILENAME}
echo "Accuracy run completed"
echo "Evaluating Accuracy "
python3 evaluation.py  --mlperf-accuracy-file ${ACCURACY_DIR}/mlperf_log_accuracy.json  --dataset-file ${DATASET_PATH}  --dtype int32 >& ${ACCURACY_DIR}/accuracy.txt 
sed -i -E 's/np\.int64\(([0-9]+)\)/\1/g' ${ACCURACY_DIR}/accuracy.txt
cat ${ACCURACY_DIR}/accuracy.txt
stop_vllm
sleep 30
fi

if [[ "${CHECK}" == "performance" ]];then 
start_vllm performance 0 ${MODEL_DIR} ${CMD}  
sleep  50
echo "Run performance"
FILENAME="offline_performance_${GPU}_llama3.18b.log"
python3 SUT_VLLM_SingleReplica_Server.py --model-name ${MODEL_DIR} --dataset-path ${DATASET_PATH} \
        --user-conf user.conf  --test-mode performance --target-qps ${TARGET_QPS} --output-log-dir ${PERF_DIR}/ \
        --api-server-url http://localhost:8000\
        >& ${PERF_DIR}/${FILENAME}
echo "Performance run completed"
stop_vllm
sleep 30
fi
