CHECKPOINT_PATH="${CHECKPOINT_PATH:meta-llama/Meta-Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:cnn_eval.json}"

mkdir -p "run_outputs"

python3 -u main.py --scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --batch-size 16 \
        --accuracy \
        --mlperf-conf mlperf.conf \
        --user-conf user.conf \
        --total-sample-count 13368 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir offline_accuracy_loadgen_logs \
        --dtype float16 \
        --vllm | tee offline_accuracy_log.log

python3 evaluation.py \
        --mlperf-accuracy-file offline_accuracy_loadgen_logs/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int32
