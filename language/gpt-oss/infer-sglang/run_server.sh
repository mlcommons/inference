#!/bin/bash

dp=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --dp)
            dp=$2
            shift 2
            ;;
    esac
done

set -x;
python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size=1 \
    --data-parallel-size=$dp \
    --max-running-requests $((dp * 512)) \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --ep-size=1 \
    --quantization mxfp4 \
    --stream-interval 50
