#!/bin/bash

pip install -r requirements.txt

dp=1
model_path=openai/gpt-oss-120b
eagle_path=""
stream_interval=500
extra_args=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dp)
            dp=$2
            shift 2
            ;;
        --model_path)
            model_path=$2
            shift 2
            ;;
        --eagle_path)
            eagle_path=$2
            shift 2
            ;;
        --stream_interval)
            stream_interval=$2
            shift 2
            ;;
        *)
            extra_args="$extra_args $1"
            shift 1
            ;;
    esac
done

args=" --model-path $model_path \
    --host 0.0.0.0 \
    --data-parallel-size=$dp \
    --max-running-requests $((dp * 512)) \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --ep-size=1 \
    --enable-metrics \
    --stream-interval $stream_interval "

if [ -n "$eagle_path" ]; then
    args="$args --speculative-draft-model-path $eagle_path \
        --speculative-algorithm EAGLE3"
fi

# --speculative-num-steps 1 \
# --speculative-eagle-topk 1 \
# --speculative-num-draft-tokens 3 \


set -x;
python3 -m sglang.launch_server $args $extra_args
