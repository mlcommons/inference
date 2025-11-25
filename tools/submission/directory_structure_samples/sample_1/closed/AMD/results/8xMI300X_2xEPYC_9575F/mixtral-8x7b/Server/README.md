# MLPerf Inference 5.1

## Setup

### Model and Dataset

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_model_and_dataset_env.sh
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_model_and_dataset_env.sh
```

Inside the docker, download the model with

```bash
# Generate an access token on huggingface and set it here
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>" python download_model.py
```

Inside the docker, download the dataset with

```bash
bash download_mixtral_8x7b.sh
```

Inside the docker, quantize the model with

```bash
bash quantize_mixtral_8x7b.sh
```

Exit the docker image, because a different image is needed for inference

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

```bash
export MLPERF_IMAGE_NAME=rocm/mlperf-inference:submission_5.1-mixtral_8x7b
```

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_submission_mixtral_8x7b.sh $MLPERF_IMAGE_NAME
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_submission_env.sh $MLPERF_IMAGE_NAME
```

### Running the benchmark

Run the following commands inside the docker container

``` bash
## Performance
python /lab-mlperf-inference/code/mixtral-8x7b/main.py \
   --config-path /lab-mlperf-inference/code/mixtral-8x7b/harness_llm/models/mixtral-8x7b/ \
   --config-name server_mi300x \
   test_mode=performance \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/mixtral-8x7b/user_mi300x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/mixtral-8x7b/Server/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/mixtral-8x7b/main.py \
   --config-path /lab-mlperf-inference/code/mixtral-8x7b/harness_llm/models/mixtral-8x7b/ \
   --config-name server_mi300x \
   test_mode=accuracy \
   harness_config.device_count=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/mixtral-8x7b/user_mi300x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/mixtral-8x7b/Server/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/mixtral-8x7b/scripts/setup_mixtral_accuracy_env.sh
bash /lab-mlperf-inference/code/mixtral-8x7b/scripts/check_mixtral_accuracy_scores.sh \
   /lab-mlperf-inference/results/mixtral-8x7b/Server/accuracy/mlperf_log_accuracy.json
```
