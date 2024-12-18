

CHECKPOINT_PATH="${CHECKPOINT_PATH:Meta-Llama-3.1-405B-Instruct}"
DATASET_PATH="${DATASET_PATH:mlperf_llama3.1_405b_dataset_8318.pkl}"

python -u main.py --scenario Server \
	--model-path ${CHECKPOINT_PATH} \
	--batch-size 16 \
	--dtype float16 \
	--user-conf user.conf \
	--total-sample-count 8313 \
	--dataset-path ${DATASET_PATH} \
	--output-log-dir output \
	--tensor-parallel-size ${GPU_COUNT} \
	--vllm 2>&1 | tee server.log
