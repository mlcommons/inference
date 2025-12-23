#!/bin/bash
# Run Offline Benchmark for GPT-OSS-120B
# Assumes SGLang server is already running on localhost:30000

set -e

cd /mnt/mlperf_gpt_oss/language/gpt-oss-120b

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check dataset
DATASET_PATH="/mnt/mlperf_gpt_oss/dataset/v4/perf/perf_eval_ref.parquet"
if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Please download the v4 dataset first."
    exit 1
fi

# Check if server is running
echo "Checking if SGLang server is running..."
if ! curl -s http://localhost:30000/health > /dev/null 2>&1; then
    echo "ERROR: SGLang server not responding on http://localhost:30000"
    echo "Please start the server first: ./sglang/start_server.sh"
    exit 1
fi

echo "✓ Server is running"
echo ""
echo "Starting Offline Benchmark..."
echo "Configuration:"
echo "  - Dataset: $DATASET_PATH"
echo "  - Scenario: Offline"
echo "  - Backend: SGLang"
echo "  - Max Concurrency: 256"
echo "  - Expected Queries: 6396 (with repeats_per_sample=1)"
echo ""

# Run benchmark
python3 run_mlperf.py \
  --scenario offline \
  --input-file "$DATASET_PATH" \
  --backend sglang \
  --server-url http://localhost:30000 \
  --max-concurrency 256 \
  --output-dir mlperf_results

echo ""
echo "✓ Benchmark complete!"
echo "Results saved to: mlperf_results/offline/performance/"
