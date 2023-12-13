# Reference Implementation for Llama-v2-70B

**Basic implementation for Llama-v2-70B. Few noteworthy items:**

+ Processing of Validation dataset is not finalized yet. Decision on input token lengths is pending
+ Streamer for communicating with loadgen has quite some overhead. This is only meant to provide functional implementation


## Prepare environment
```
conda create -n llama2-70b python=3.9
conda activate llama2-70b

# Install packages
conda install pybind11==2.10.4 -c conda-forge -y
python -m pip install torch==2.2.0.dev20231006+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers==4.31.0 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0

export CUR_DIR=${PWD}
cd <inference-repo-root>/loadgen

# Need to fetch Pablo's changes
git fetch origin pull/1523/head:llm-server
git merge llm-server

python -m pip install .
```


## Get Model
+ For now, MLCommons is not hosting the checkpoing, so you must first go to [llama2-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request, sign in to huggingface (if you don't have account, you'd need to create one). **Please note your authentication credentials** as you may be required to provide them when cloninng below
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=${PWD}/Llama-2-70b-chat-hf
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf ${CHECKPOINT_PATH}

```

## Get Dataset

```
# First get the `open-orca` parquet from huggingface
export OPENORCA_DATASET=${PWD}/open-orca
git clone https://huggingface.co/datasets/Open-Orca/OpenOrca ${OPENORCA_DATASET}

export OPENORCA_PARQUET=${OPENORCA_DATASET}/1M-GPT4-Augmented.parquet
EXPORT_DIR=${PWD}/processed-openorca
export DATASET_PATH=${PWD}/processed-data.pkl

# Process the dataset according the Taskforce's agreed criteria
python3 processorca.py --dataset_pq_path=${OPENORCA_PARQUET} --model_dir=${CHECKPOINT_PATH} --seqlen_limit=2048 --export_dir=${EXPORT_DIR} --num_total_samples=24576

mv ${EXPORT_DIR}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATASET_PATH}
```


## Run Performance Benchmarks

### Offline
```
python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 24576 \
                --device cpu \
		--dataset-path ${DATASET_PATH} \
                --output-log-dir offline-logs

```

### Server
```
python -u main.py --scenario Server \
                --model-path ${CHECKPOINT_PATH} \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 24576 \
                --device cpu \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir server-logs
```


## Run Accuracy Benchmarks

### Offline
```
OUTPUT_LOG_DIR=offline-accuracy-logs

python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
                --accuracy \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 24576 \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir ${OUTPUT_LOG_DIR} \
                --device cpu


ACCURACY_LOG_FILE=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json
if [ -e ${ACCURACY_LOG_FILE} ]; then
        python evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
                --mlperf-accuracy-file ${ACCURACY_LOG_FILE} --dataset-file ${DATASET_PATH} --dtype int32
fi
```

### Server
```
OUTPUT_LOG_DIR=server-accuracy-logs

python -u main.py --scenario Server \
                --model-path ${CHECKPOINT_PATH} \
                --accuracy \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 24576 \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir ${OUTPUT_LOG_DIR} \
                --device cpu


ACCURACY_LOG_FILE=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json
if [ -e ${ACCURACY_LOG_FILE} ]; then
        python evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
                --mlperf-accuracy-file ${ACCURACY_LOG_FILE} --dataset-file ${DATASET_PATH} --dtype int32
fi
```

