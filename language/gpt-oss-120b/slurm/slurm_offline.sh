#!/bin/bash
#SBATCH --job-name=gpt-oss-offline
#SBATCH --partition=1n4gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=4:00:00

WORKDIR=$SLURM_SUBMIT_DIR
cd $WORKDIR

# Initialize and activate conda
eval "$(conda shell.bash hook)"
conda activate bisection

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p logs outputs

echo "=========================================="
echo "GPT-OSS-120B MLPerf Offline Benchmark"
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi -L | wc -l) GPU(s)"
echo "=========================================="

# Start SGLang server
nohup python3 -m sglang.launch_server \
    --model-path /work/hps0/home/ea0020/other-code/inference/language/gpt-oss-120b/download/gpt-oss-model/gpt-oss-120b \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 \
    --max-running-requests 512 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --enable-metrics \
    --stream-interval 500 \
    > logs/server_$(date +%Y%m%d_%H%M%S).log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to be ready..."
for i in $(seq 1 300); do
    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:30000/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "Server ready after ${i}s (HTTP $HTTP_CODE)"
        break
    fi
    sleep 2
done

# Wait for full warmup (FlashInfer autotune, etc.)
echo "Waiting 60s for server warmup..."
sleep 60

# Run workers
python3 run_mlperf.py \
    --scenario offline \
    --input-file /work/hps0/home/ea0020/other-code/inference/language/gpt-oss-120b/download/gpt-oss-dataset/acc/acc_eval_ref.parquet \
    --backend sglang \
    --server-url http://localhost:30000 \
    --output-dir outputs/results_$(date +%Y%m%d_%H%M%S) \
    --max-new-tokens 32768 \
    --mlperf-conf mlperf.conf \
    --user-conf user.conf || true

# Kill server and its worker processes immediately after benchmark
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Shutting down server (PID $SERVER_PID)..."
    pkill -P $SERVER_PID 2>/dev/null || true
    kill $SERVER_PID 2>/dev/null || true
    sleep 3
    kill -9 $SERVER_PID 2>/dev/null || true
fi
echo "Benchmark complete"
