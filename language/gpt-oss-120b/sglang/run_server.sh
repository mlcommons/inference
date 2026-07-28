#!/bin/bash

pip install -r requirements.txt

dp=1
model_path=openai/gpt-oss-120b
eagle_path=""
stream_interval=500
extra_args=""

# Interactive scenario: EAGLE3 speculative decoding with the long-context head.
# num_steps=3 and topk=1 are fixed per the Interactive scenario policy.
enable_speculative_decode=false
SPECULATIVE_DRAFT_MODEL_PATH="nvidia/gpt-oss-120b-Eagle3-long-context"
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_TOPK=1

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
	--enable_speculative_decode)
		enable_speculative_decode=true
		shift 1
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

# Explicit --eagle_path wins over the --enable_speculative_decode defaults so
# existing callers keep working unchanged.
if [ -n "$eagle_path" ]; then
	args="$args --speculative-draft-model-path $eagle_path \
        --speculative-algorithm EAGLE3"
elif [ "$enable_speculative_decode" = "true" ]; then
	args="$args --speculative-draft-model-path $SPECULATIVE_DRAFT_MODEL_PATH \
        --speculative-algorithm EAGLE3 \
        --speculative-num-steps $SPECULATIVE_NUM_STEPS \
        --speculative-eagle-topk $SPECULATIVE_TOPK"
fi

set -x
python3 -m sglang.launch_server "$args" "$extra_args"
