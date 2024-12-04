CHECKPOINT_PATH="${CHECKPOINT_PATH:Meta-Llama-3.1-405B-Instruct}"
DATASET_PATH="${DATASET_PATH:-open-orca-val-set.pkl}"

python -u main.py --scenario Offline \
		--model-path ${CHECKPOINT_PATH} \
		--mlperf-conf mlperf.conf \
		--user-conf user.conf \
		--total-sample-count 8312 \
		--dataset-path ${DATASET_PATH} \
		--device cpu 2>&1 | tee server_log.log
