

CHECKPOINT_PATH="${CHECKPOINT_PATH:mistralai/Mixtral-8x7B-Instruct-v0.1}"
DATASET_PATH="${DATASET_PATH:dataset/2024_06_06_mixtral_15k_v4.pkl}"

python -u main.py --scenario Server \
		--model-path ${CHECKPOINT_PATH} \
		--mlperf-conf mlperf.conf \
		--user-conf user.conf \
		--total-sample-count 15000 \
		--dataset-path ${DATASET_PATH} \
		--device cpu 2>&1 | tee server_log.log
