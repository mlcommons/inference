

CHECKPOINT_PATH="${CHECKPOINT_PATH:meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:cnn_eval.json}"

python -u main.py --scenario Server \
	--model-path ${CHECKPOINT_PATH} \
	--batch-size 16 \
	--dtype float16 \
	--user-conf user.conf \
	--total-sample-count 13368 \
	--dataset-path ${DATASET_PATH} \
	--output-log-dir output \
	--tensor-parallel-size ${GPU_COUNT} \
	--vllm 2>&1 | tee server.log
