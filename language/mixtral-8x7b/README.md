# Reference Implementation for Mixtral-8x7B-instruct-v0.1

**Basic implementation for Mixtral-8x7B-instruct-v0.1. Few noteworthy items:**

+ Dataset was constructed by randomly sampling from the validation split of 3 datasets, open_orca_gpt4, GSM8k and MBXP. 5K samples from each one.
+ Streamer for communicating with loadgen has quite some overhead. This is only meant to provide functional implementation
+ For custom/optimized implementations of this benchmark it is important to include the :
        - For server scenario, it is necessary to call `lg.FirstTokenComplete(response)` for each query. This way the first token will be reported and it's latency will be measured.
        - For all scenarios, when calling `lg.QuerySamplesComplete(response)`, it is necessary that each of the elements in response is a `lg.QuerySampleResponse` that contains the number of tokens (can be create this way: `lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)`). The number of tokens reported should match with the number of tokens on your answer and this will be checked in [TEST06](../../compliance/nvidia/TEST06/)

## Automated command to run the benchmark via MLCommons CM 

TODO
 
## Prepare environment

Copy the mlperf.conf file to this folder.
```
cp ../../mlperf.conf .
```

For a CPU-only run:

```
conda create -n Mixtral-8x7B python=3.9
conda activate Mixtral-8x7B

# Install packages
conda install pybind11==2.10.4 -c conda-forge -y
python -m pip install torch==2.2.0.dev20231006+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers==4.31.0 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0
pip install git+https://github.com/amazon-science/mxeval.git@e09974f990eeaf0c0e8f2b5eaff4be66effb2c86

export CUR_DIR=${PWD}
cd <inference-repo-root>/loadgen

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

TODO: Create MLCommons get fixed link.
For now it can be downloaded from [Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main)

## Get Dataset

TODO: Create scripts and procedure to download all of the parts of the dataset

### Preprocessed

#### Using Rclone
We make many of the MLPerf infernce models and datasets available using Rclone. In order to keep compatibility, you can use Rclone to get the preprocessed dataset:

To run Rclone on Windows, you can download the executable [here](https://rclone.org/install/#windows).
To install Rclone on Linux/macOS/BSD systems, run:
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
Once Rclone is installed, cd into the folder where you want to place the dataset and run:
```bash
rclone copyurl https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_v4.pkl ./ -a -P
```
#### Using wget

Alternatively, you can simply cd into the folder where you want to place the dataset and run
```bash
wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_v4.pkl
```

### Calibration dataset

#### Using Rclone
Rclone is installed, cd into the folder where you want to place the dataset and run:
```bash
rclone copyurl https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl ./ -a -P
```

#### Using wget

Alternatively, you can simply cd into the folder where you want to place the dataset and run
```bash
wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl
```
### Unprocessed

TODO: Share instructions and scripts

## Run Performance Benchmarks

### Offline
```
python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 15000 \
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
        --total-sample-count 15000 \
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
                --total-sample-count 15000 \
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
                --total-sample-count 15000 \
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
                --total-sample-count 15000 \
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

### Evaluation
Recreating the enviroment for evaluating the quality metrics can be quite tedious. Therefore we provide a dockerfile and recommend using docker for this task.
1. Build the evaluation container
```bash
docker build . -f Dockerfile.eval -t evaluation
```
2. Run the docker in interactive mode and with 
```bash
sudo docker run -it -v $(pwd):/eval -t evaluation
```
3. 
```bash
cd eval
huggingface-cli login --token [huggingface_token]
python -u evaluate-accuracy.py --checkpoint-path mistralai/Mixtral-8x7B-instruct-v0.1 \
                --mlperf-accuracy-file [path_to_mlperf_accuracy_file] \
                --dataset-file [path_to_dataset] \
                --n_workers 8
```


## Accuracy Target

Reference scores:
Open Orca:
```json
{'rouge1': 45.4911, 'rouge2': 23.2829, 'rougeL': 30.3615, 'rougeLsum': 42.4333}
```
GSM8K:
```json
{'gsm8k_accuracy': 73.78}
```
MBXP:
```json
{'mbxp_accuracy': 60.16}
```
For official submissions, 99% of each reference score is enforced. Additionally, 90%-110% of the generated tokens_per_samples:
```json
{'tokens_per_sample': 145.9}
```