#!/bin/bash

model_name="openai/gpt-oss-120b"
data_file="/home/mlperf_inference_storage/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl"
output_file="responses.jsonl"
max_samples=4388
max_token_osl=20000
max_concurrency=256
tep_size=8
mlperf_storage="/home/mlperf_inference_storage"

out_folder=output_$SLURM_JOB_ID
mkdir -p $out_folder

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "    --model-name: Set the model name"
            echo "    --data-file: Set the data file"
            echo "    --output-file: Set the output file"
            echo "    --max-samples: Set the maximum number of samples"
            echo "    --max-token-osl: Set the maximum token output length"
            echo "    --max-concurrency: Set the maximum concurrency for requests sent"
            echo "    --tep-size: Set tp, ep size for server"
            echo "    --mlperf-storage: Set the mlperf storage directory"
            echo "    --help|-h: Show this help message"
            exit 0
            ;;
        --model-name)
            model_name=$2
            shift 2
            ;;
        --data-file)
            data_file=$2
            shift 2
            ;;
        --output-file)
            output_file=$2
            shift 2
            ;;
        --max-samples)
            max_samples=$2
            shift 2
            ;;
        --max-tokens)
            max_tokens=$2
            shift 2
            ;;
        --max-concurrency)
            max_concurrency=$2
            shift 2
            ;;
        --tep-size)
            tep_size=$2
            shift 2
            ;;
        --mlperf-storage)
            mlperf_storage=$2
            shift 2
            ;;
    esac
done

set -x; 
srun --nodes=1 \
    --output=$out_folder/server.log \
    --container-image=./images/lmsysorg+sglang+v0.5.3rc1.sqsh \
    --container-mounts=$(pwd):/$(pwd),$mlperf_storage:/home/mlperf_inference_storage \
    --container-name=sglang_server_$SLURM_JOB_ID \
    python3 -m sglang.launch_server \
    --model-path $model_name \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --tp-size=$tep_size \
    --data-parallel-size=1 \
    --cuda-graph-max-bs $max_concurrency \
    --max-running-requests $max_concurrency \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 16384 \
    --ep-size $tep_size \
    --quantization mxfp4  \
    --enable-flashinfer-allreduce-fusion \
    --enable-symm-mem  \
    --disable-radix-cache \
    --attention-backend trtllm_mha \
    --moe-runner-backend flashinfer_trtllm \
    --stream-interval 10 &
set +x;

SERVER_PID=$!
echo "Server launched with PID: $SERVER_PID"

echo "Waiting for server to start on port 30000..."
TIMEOUT=1200  # 20 minutes timeout
ELAPSED=0
while ! srun --nodes=1 --overlap netstat -tulnp 2>/dev/null | grep -q ":30000"; do
    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process has died. Checking server logs..."
        echo "Last 20 lines of server log:"
        tail -20 $out_folder/server.log
        echo "Server launch failed. Exiting."
        exit 1
    fi
    
    # Check for timeout
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Server failed to start within $TIMEOUT seconds. Checking server logs..."
        echo "Last 20 lines of server log:"
        tail -20 $out_folder/server.log
        echo "Timeout reached. Exiting."
        exit 1
    fi
    
    echo "Server not ready yet, waiting... (${ELAPSED}s/${TIMEOUT}s)"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "Server is ready on port 30000!"

srun --nodes=1 --overlap --container-name sglang_client_$SLURM_JOB_ID --output=$out_folder/client.log \
    python3 send_requests.py \
    --model-name $model_name \
    --data-file $data_file \
    --output $out_folder/responses.jsonl \
    --max-samples $max_samples \
    --max-tokens $max_token_osl \
    --max-concurrency $max_concurrency
