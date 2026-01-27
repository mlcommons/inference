# Reference Implementation for Mixtral-8x7B-instruct-v0.1

**Basic implementation for Mixtral-8x7B-instruct-v0.1. Few noteworthy items:**

+ Dataset was constructed by randomly sampling from the validation split of 3 datasets, open_orca_gpt4, GSM8k and MBXP. 5K samples from each one.
+ Streamer for communicating with loadgen has quite some overhead. This is only meant to provide functional implementation
+ For custom/optimized implementations of this benchmark it is important to include the :
        - For server scenario, it is necessary to call `lg.FirstTokenComplete(response)` for each query. This way the first token will be reported and it's latency will be measured.
        - For all scenarios, when calling `lg.QuerySamplesComplete(response)`, it is necessary that each of the elements in response is a `lg.QuerySampleResponse` that contains the number of tokens (can be create this way: `lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)`). The number of tokens reported should match with the number of tokens on your answer and this will be checked in [TEST06](../../compliance/TEST06/)


## Automated command to run the benchmark via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/language/mixtral-8x7b/) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Prepare environment

For a CPU-only run:

```
conda create -n Mixtral-8x7B python=3.9
conda activate Mixtral-8x7B

# Install packages
conda install pybind11==2.10.4 -c conda-forge -y
python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
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


## Model

| model | accuracy | model source | precision |
| ---- | ---- | ---- | ---- |
| Mixtral-8x7B-Instruct-v0.1 | [Accuracy target](#accuracy-target) | [Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | fp16 |

**Important Note:** Files and configurations of the model have changed, and might change in the future. If you are going to get the model from Hugging Face or any external source, use a version of the model that exactly matches the one in this [commit](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/commit/a60832cb6c88d5cb6e507680d0e9996fbad77050). We strongly recommend to get the model following the steps in the next section:

### Download model through MLCFlow Automation

```
mlcr get,ml-model,mixtral,_r2-downloader,_mlc --outdirname=<path_to_download> -j
```

### Get Checkpoint

#### Using the MLCommons R2 Downloader

(More information about the MLC R2 Downloader, including how to run it on Windows, can be found [here](https://inference.mlcommons-storage.org))

Navigate in the terminal to your desired download directory and run the following command to download the model checkpoint:

``` bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/mixtral-8x7b-model-checkpoint.uri
```

## Get Dataset

### Download Preprocessed dataset through MLCFlow Automation

**Validation**

```
mlcr get,dataset-mixtral,openorca-mbxp-gsm8k-combined,_r2-downloader,_validation --outdirname=<path to download> -j
```

**Calibration**

```
mlcr get,dataset-mixtral,openorca-mbxp-gsm8k-combined,_calibration,_r2-downloader --outdirname=<path to download> -j
```

### Preprocessed

#### Using the MLCommons R2 Downloader
We make many of the MLPerf infernce models and datasets available using the MLC R2 Downloader (more information about the MLC R2 Downloader, including how to run it on Windows, can be found [here](https://inference.mlcommons-storage.org)). In order to keep compatibility, you can use the MLC R2 Downloader to get the preprocessed dataset:

`cd` into the folder where you want to place the dataset and run:
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/mixtral-8x7b-validation-dataset.uri
```
#### Using wget

Alternatively, you can simply cd into the folder where you want to place the dataset and run


```bash
wget https://inference.mlcommons-storage.org/mixtral_8x7b/09292024_mixtral_15k_mintoken2_v1.pkl
```

### Calibration dataset

#### Using the MLCommons R2 Downloader

`cd` into the folder where you want to place the dataset and run:
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/mixtral-8x7b-calibration-dataset.uri
```

#### Using wget

Alternatively, you can simply cd into the folder where you want to place the dataset and run
```bash
wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl
```

## Run Performance Benchmarks

### Offline
```
python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
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

## Accuracy Evaluation

### Evaluate the accuracy through MLCFlow Automation
```bash
mlcr process,mlperf,accuracy,_openorca-gsm8k-mbxp-combined --result_dir=<Path to directory where files are generated after the benchmark run>
```

Please click [here](https://github.com/mlcommons/inference/blob/master/language/mixtral-8x7b/evaluate-accuracy.py) to view the Python script for evaluating accuracy for the Waymo dataset.

### Evaluation
Recreating the enviroment for evaluating the quality metrics can be quite tedious. Therefore we provide a dockerfile and recommend using docker for this task.
1. Build the evaluation container
```bash
docker build . -f Dockerfile.eval -t evaluation
```
2. Run the docker in interactive mode and with
```bash
docker run -it --rm --net=host --runtime=nvidia --ipc=host -v $PWD:$PWD -w $PWD evaluation
```
3.
```bash
cd eval
python -u evaluate-accuracy.py --checkpoint-path [path_to_model_checkpoint] \
                --mlperf-accuracy-file [path_to_mlperf_accuracy_file] \
                --dataset-file [path_to_dataset] \
                --n_workers 8
```


## Accuracy Target

**WARNING:** The full accuracy target was only verified with the standalone script. The reference implementation matches in a subset of the dataset, but hasn't been fully confirm.

Reference scores:
Open Orca:
```json
{'rouge1': 45.5989, 'rouge2': 23.3526, 'rougeL': 30.4608}
```
GSM8K:
```json
{'gsm8k': 73.66}
```
MBXP:
```json
{'mbxp': 60.16}
```
For official submissions, 99% of each reference score is enforced. Additionally, 90%-110% of the generated tokens_per_samples (counting all the non-EOS tokens):
```json
{'tokens_per_sample': 144.84}
```

## Automated command for submission generation via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/submission/) for an automated way to generate submission through MLCFlow. 
