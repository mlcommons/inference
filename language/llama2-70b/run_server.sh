

CHECKPOINT_PATH="${CHECKPOINT_PATH:-meta-llama/Llama-2-70b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-open-orca-val-set.pkl}"

python -u main.py --scenario Server \
		--model-path ${CHECKPOINT_PATH} \
		--mlperf-conf mlperf.conf \
		--user-conf user.conf \
		--total-sample-count 24576 \
		--dataset-path ${DATASET_PATH} \
		--device cpu 2>&1 | tee server_log.log
