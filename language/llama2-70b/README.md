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
cd <inference-repo-root>

git submodule update --init --recursive third_party/pybind/


cd loadgen
python -m pip install .
```

## Get Dataset

```
export DATASET_PATH=</path/to/save/processed-data.pkl>

# Process the dataset per Taskforce agreed criteria
python3 processorca.py --dataset_pq_path=1M-GPT4-Augmented.parquet --model_dir=Llama-2-70b-chat-hf --seqlen_limit=2048 --export_dir=llama/filtered --num_total_samples=24576
```


## Get Model
+ For now, MLCommons is not hosting the checkpoing, so you must first go to [llama2-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request, sign in to huggingface (if you don't have account, you'd need to create one). **Please note your authentication credentials** as you may be required to provide them when cloninng below
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=${PWD}/Llama-2-70b-chat-hf
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf ${CHECKPOINT_PATH}

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

TODO

