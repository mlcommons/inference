#!/bin/bash
# SGLang Server Startup for GPT-OSS-120B Offline Benchmark
# System: 2x NVIDIA L40S (46GB each)
# Model: GPT-OSS-120B MXFP4 (~61GB, requires TP=2)

set -e

cd /mnt/mlperf_gpt_oss/language/gpt-oss-120b

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if SGLang is installed
if ! python3 -c "import sglang" 2>/dev/null; then
    echo "ERROR: SGLang not installed in virtual environment"
    echo "Please run: source .venv/bin/activate && pip install 'sglang[all]'"
    exit 1
fi

# echo "Starting SGLang server..."
# echo "Configuration:"
# echo "  - Model: /mnt/models/gpt-oss-120b"
# echo "  - Tensor Parallelism: 2 GPUs"
# echo "  - Max Running Requests: 512"
# echo "  - Memory Fraction: 0.85"
# echo "  - Chunked Prefill Size: 16384"
# echo "  - Port: 30000"
# echo ""

# Start SGLang server
# python3 -m sglang.launch_server \
#   --model-path /mnt/models/gpt-oss-120b \
#   --host 0.0.0.0 \
#   --port 30000 \
#   --tp 2 \
#   --max-running-requests 512 \
#   --mem-fraction-static 0.85 \
#   --chunked-prefill-size 16384 \
#   --enable-metrics

python3 -m sglang.launch_server \
  --model-path /mnt/models/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 2 \
  --max-running-requests 1024 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 32768 \
  --enable-metrics
