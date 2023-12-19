CHECKPOINT_PATH="${CHECKPOINT_PATH:-/raid/data/mlperf-llm/Llama-2-70b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-/raid/data/alicheng/OpenOrca/llama/first_token_hack/open_orca_gpt4_tokenized_llama.sampled_24576.pkl}"

mkdir -p "run_outputs"

python3 -u main.py --scenario Offline \
		--model-path ${CHECKPOINT_PATH} \
        --accuracy \
		--mlperf-conf mlperf.conf \
		--user-conf user.conf \
		--total-sample-count 24576 \
		--dataset-path ${DATASET_PATH} \
        --output-log-dir offline_accuracy_loadgen_logs \
        --dtype float32 \
		--device cuda:0 2>&1 | tee offline_accuracy_log.log

python3 evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
        --mlperf-accuracy-file offline_accuracy_loadgen_logs/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int32
