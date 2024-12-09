# Reference Implementation for llama2-70b

**Basic implementation for llama2-70b. Few noteworthy items:**

+ Streamer for communicating with loadgen has quite some overhead. This is only meant to provide functional implementation
+ For custom/optimized implementations of this benchmark it is important to include the :
        - For server scenario, it is necessary to call `lg.FirstTokenComplete(response)` for each query. This way the first token will be reported and it's latency will be measured.
        - For all scenarios, when calling `lg.QuerySamplesComplete(response)`, it is necessary that each of the elements in response is a `lg.QuerySampleResponse` that contains the number of tokens (can be create this way: `lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)`). The number of tokens reported should match with the number of tokens on your answer and this will be checked in [TEST06](../../compliance/nvidia/TEST06/)

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/language/llama2-70b) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

 
## Prepare environment

Copy the mlperf.conf file to this folder.
```
cp ../../mlperf.conf .
```

For a CPU-only run:

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

For a GPU-based run:

A dockerfile is provided, along with scripts to help launch it. First, add any docker volume mounts you want in
`launch.sh`. There is a section at the top of the file that looks like:
```
# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH
)
```

For example if you have a raid space located at `/raid/data` on your local machine, you can add it to the same path in the container like so:
```
# Add any volume mounts here with the following syntax
# /path/to/src:/path/to/dir/in/container
MOUNTS=(
    $MLCOMMONS_REPO_PATH:$MLCOMMONS_REPO_PATH
    /raid/data:/raid/data
)
```
Once you have added all your mounts, launch the container with `bash launch.sh`.

Inside the container, set up the environment with `bash build.sh`. This will install all the dependencies from the
CPU-only setup, as well as any GPU versions for applicable libraries like PyTorch.


## Get Model
### MLCommons Members Download
MLCommons hosts the model and preprocessed dataset for download **exclusively by MLCommons Members**. You must first agree to the [confidentiality notice](https://llama2.mlcommons.org) using your organizational email address, then you will receive a link to a directory containing Rclone download instructions. _If you cannot access the form but you are part of a MLCommons Member organization, submit the [MLCommons subscription form](https://mlcommons.org/community/subscribe/) with your organizational email address and [associate a Google account](https://accounts.google.com/SignUpWithoutGmail) with your organizational email address._


### External Download
+ First go to [llama2-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). **Please note your authentication credentials** as you may be required to provide them when cloning below.
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=${PWD}/Llama-2-70b-chat-hf
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf ${CHECKPOINT_PATH}

```

## Get Dataset

### Preprocessed

You can use Rclone to download the preprocessed dataset from a Cloudflare R2 bucket.

To run Rclone on Windows, you can download the executable [here](https://rclone.org/install/#windows).
To install Rclone on Linux/macOS/BSD systems, run:
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
Once Rclone is installed, run the following command to authenticate with the bucket:
```
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```
You can then navigate in the terminal to your desired download directory and run the following command to download the dataset:

```
rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P
```

### Unprocessed

You can also download and process the dataset yourself following the command below:

```
# First get the `open-orca` parquet from huggingface
export OPENORCA_DATASET=${PWD}/open-orca
git clone https://huggingface.co/datasets/Open-Orca/OpenOrca ${OPENORCA_DATASET}

export OPENORCA_PARQUET=${OPENORCA_DATASET}/1M-GPT4-Augmented.parquet
EXPORT_DIR=${PWD}/processed-openorca
export DATASET_PATH=${PWD}/processed-data.pkl

# Process the dataset according the Taskforce's agreed criteria
python3 processorca.py --dataset_pq_path=${OPENORCA_PARQUET} --model_dir=${CHECKPOINT_PATH} --seqlen_limit=1024 --export_dir=${EXPORT_DIR} --num_total_samples=24576

mv ${EXPORT_DIR}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATASET_PATH}
```

The script will perform the following steps on the original open_orca GPT4 dataset:
- filter out all queries with non-ascii characters, except for normal unicode quotes and hyphens.
- filter out all queries with out-of-bound input/output sequence lengths
- filter out all queries with expected answers shorter than 2 words (known to cause issues for Llama2)
- filter out all queries with prompts that generate bad output texts using Llama2 models
- sample equally from the sub-dataset (i.e. COT, NIV, FLAN, T0) and form the final dataset.

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

For a GPU-based run:
```
python3 -u main.py --scenario Offline \
        --model-path ${CHECKPOINT_PATH} \
        --mlperf-conf mlperf.conf \
        --user-conf user.conf \
        --total-sample-count 24576 \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir offline-logs \
        --dtype float32 \
        --device cuda:0 2>&1 | tee offline_performance_log.log
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

The ServerSUT was not tested for GPU runs.


## Run Accuracy Benchmarks

### Offline
```
OUTPUT_LOG_DIR=offline-accuracy-logs

mkdir -p "run_outputs"  # The script will dump all the outputs to 'run_outputs'.

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

# Optional: Create a pickled pandas DataFrame that is the original dataset with extra columns with output data from the
# accuracy run. The following columns will be added:
# - "gen_output_tok_id": A list of ints representing the tokenized output sequence.
# - "gen_output_text": A str representing the untokenized output sequence.
# - "gen_output_tok_len": An int representing the number of output tokens.
# - "rouge1": The rouge1 score for this sample
# - "rouge2": The rouge2 score for this sample
# - "rougeL": The rougeL score for this sample
# This file will by default be saved to 'full_output.pkl'. You can modify this with --output-pkl-path.
python consolidate_results.py --dataset-path ${DATASET_PATH} --model-dir ${CHECKPOINT_PATH}
```

For the GPU run - The above steps have been automated in `run_accuracy.sh`. You can also modify this script to use
`--device cpu` to adapt it to a CPU-only run.


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

The ServerSUT was not tested for GPU runs.


## Accuracy Target
Running the GPU implementation in FP16 precision resulted in the following FP16 accuracy targets (normalized to a 0-100
scale from a 0.0-1.0 scale):
- Rouge1: 44.4312
- Rouge2: 22.0352
- RougeL: 28.6162
- Tokens per sample: 294.45

This was run on a DGX-H100 node. Total runtime was ~4.5 days.
