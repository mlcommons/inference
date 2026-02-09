CHECKPOINT_PATH="${CHECKPOINT_PATH:-meta-llama/Llama-2-70b-chat-hf}"
DATASET_PATH="${DATASET_PATH:-open-orca-val-set.pkl}"

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

python3 consolidate_results.py --dataset-path ${DATASET_PATH} --model-dir ${CHECKPOINT_PATH}
