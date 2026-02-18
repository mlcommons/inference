CHECKPOINT_PATH=/share/mlperf_sets/model/llama-2-70b-chat-hf.uri
DATASET_PATH=/share/mlperf_sets/data/validation/llama-2-70b-open-orca-dataset.uri/open_orca_gpt4_tokenized_llama.sampled_24576.pkl

mkdir -p "run_outputs"

python3 -u main.py --scenario Offline --vllm\
        --model-path ${CHECKPOINT_PATH} \
        --accuracy \
        --user-conf user.conf \
        --total-sample-count 24576 \
        --dataset-path ${DATASET_PATH} \
        --num-workers 4 \
        --output-log-dir offline_accuracy_loadgen_logs \
        --dtype float32 \
        --api-server http://127.0.0.1:8000 \
        --api-model-name ${CHECKPOINT_PATH}
        --device cuda:0 2>&1 | tee offline_accuracy_log.log

python3 evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
        --mlperf-accuracy-file offline_accuracy_loadgen_logs/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} \
        --dtype int32

python3 consolidate_results.py --dataset-path ${DATASET_PATH} --model-dir ${CHECKPOINT_PATH}
