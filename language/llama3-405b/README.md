# Reference Implementation for llama3-405b

**Basic implementation for llama3-405b. Few noteworthy items:**

+ Streamer for communicating with loadgen has quite some overhead. This is only meant to provide functional implementation
+ For custom/optimized implementations of this benchmark it is important to include the :
        - For server scenario, it is necessary to call `lg.FirstTokenComplete(response)` for each query. This way the first token will be reported and it's latency will be measured.
        - For all scenarios, when calling `lg.QuerySamplesComplete(response)`, it is necessary that each of the elements in response is a `lg.QuerySampleResponse` that contains the number of tokens (can be create this way: `lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)`). The number of tokens reported should match with the number of tokens on your answer and this will be checked in [TEST06](../../compliance/nvidia/TEST06/)

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/language/llama3-405b) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.


## Prepare environment

### Local Environment Run

The following steps were tested in Ubuntu 22.04 with python 3.10

- **Prerrequisite for GPU runs:** Install Nvidia Driver and cuda 12.1.

The following links contain the commands for installing the [NVIDIA Driver](https://developer.nvidia.com/datacenter-driver-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) and [Cuda](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

- **Prerrequisite:** Install conda.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init
```

- Set the following helper variables
```bash
export ROOT=$PWD/inference
export LLAMA_FOLDER=$PWD/inference/language/llama3-405b
export LOADGEN_FOLDER=$PWD/inference/loadgen
export DATASET_FOLDER=$PWD/inference/language/llama3-405b/dataset
```

- Clone the inference repository:
```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git \
 --depth 1
```

- Create a conda environment:
```bash
conda create -y -n llama3-405b python=3.10
conda activate llama3-405b
conda install -y -c conda-forge libstdcxx-ng=12
```

- Install requirements and loadgen:
```bash
cd $LLAMA_FOLDER
# Install packages
pip install -r requirements.txt
```

```bash
cd $LOADGEN_FOLDER
pip install -e .
```

### Docker Run

A dockerfile is provided, along with scripts to help launch it. First, add any docker volume mounts you want in
`launch_docker.sh`. There is a section at the top of the file that looks like:
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
Once you have added all your mounts, build and launch the container with `bash launch.sh`.

Now install all the dependencies:
```
pip install -r requirements.txt
pip install -e ../../loadgen
```


## Get Model
### MLCommons Members Download

TODO: Host model and grant access to submitters


### External Download
+ First go to [llama3-request-link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). **Please note your authentication credentials** as you may be required to provide them when cloning below.
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=Meta-Llama-3.1-405B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout be673f326cab4cd22ccfef76109faf68e41aa5f1
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
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3_405b/mlperf_llama3.1_405b_dataset_8313_processed_fp16_eval.pkl ./ -P
```

You can also download the calibration dataset from the Cloudflare R2 bucket by running the following command:

```
rclone copy mlc-inference:mlcommons-inference-wg-public/llama3_405b/mlperf_llama3.1_405b_calibration_dataset_512_processed_fp16_eval.pkl ./ -P
```

## Run Performance Benchmarks

### Offline
```
python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
                --batch-size 16 \
                --dtype float16 \
                --user-conf user.conf \
                --total-sample-count 8313 \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir output \
                --tensor-parallel-size ${GPU_COUNT} \
                --vllm

```

### Server
```
python -u main.py --scenario Server \
                --model-path ${CHECKPOINT_PATH} \
                --batch-size 16 \
                --dtype float16 \
                --user-conf user.conf \
                --total-sample-count 8313 \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir output \
                --tensor-parallel-size ${GPU_COUNT} \
                --vllm
```

The ServerSUT was not tested for GPU runs.


## Run Accuracy Benchmarks

### Offline
```
OUTPUT_LOG_DIR=offline-accuracy-logs

mkdir -p "run_outputs"  # The script will dump all the outputs to 'run_outputs'.

python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
                --batch-size 16 \
                --accuracy \
                --dtype float16 \
                --user-conf user.conf \
                --total-sample-count 8313 \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir output \
                --tensor-parallel-size ${GPU_COUNT} \
                --vllm


ACCURACY_LOG_FILE=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json
if [ -e ${ACCURACY_LOG_FILE} ]; then
        python evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
                --mlperf-accuracy-file ${ACCURACY_LOG_FILE} --dataset-file ${DATASET_PATH} --dtype int32
fi
```

For the GPU run - The above steps have been automated in `run_accuracy.sh`. You can also modify this script to use
`--device cpu` to adapt it to a CPU-only run.


### Server
```
OUTPUT_LOG_DIR=server-accuracy-logs

python -u main.py --scenario Server \
                --model-path ${CHECKPOINT_PATH} \
                --batch-size 16 \
                --accuracy \
                --dtype float16 \
                --user-conf user.conf \
                --total-sample-count 8313 \
                --dataset-path ${DATASET_PATH} \
                --output-log-dir output \
                --tensor-parallel-size ${GPU_COUNT} \
                --vllm


ACCURACY_LOG_FILE=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json
if [ -e ${ACCURACY_LOG_FILE} ]; then
        python evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
                --mlperf-accuracy-file ${ACCURACY_LOG_FILE} --dataset-file ${DATASET_PATH} --dtype int32
fi
```

The ServerSUT was not tested for GPU runs.


## Accuracy Target
Running the GPU implementation in FP16 precision resulted in the following FP16 accuracy targets:
```
{
        'rougeL': 21.6666,
        'exact_match': 90.1335,
        'tokens_per_sample': 684.68,
}
```
The accuracy target is 99% for rougeL and exact_match, and 90% for tokens_per_sample
