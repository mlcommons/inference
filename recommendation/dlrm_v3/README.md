# MLPerf Inference reference implementation for DLRMv3

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site(WIP)]() for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Install dependencies and build loadgen

The reference implementation has been tested on a single host, with x86_64 CPUs and 8 NVIDIA H100/B200 GPUs. Dependencies can be installed below,
```
sh setup.sh
```

## Dataset download

DLRMv3 uses a synthetic dataset specifically designed to match the model and system characteristics of large-scale sequential recommendation (large item set and long average sequence length for each request). To generate the dataset used for both training and inference, run
```
python streaming_synthetic_data.py
```
The generated dataset has 2TB size, and contains 5 million users interacting with a billion items over 100 timestamps.

Only 1% of the dataset is used in the inference benchmark. The sampled DLRMv3 dataset and trained checkpoint are available at https://inference.mlcommons-storage.org/.

### Download dataset through MLCFlow Automation

```
mlcr get-dataset-mlperf-inference-dlrmv3-synthetic-streaming,_mlc,_r2-downloader --outdirname=<path_to_download> -j
```

### Download dataset through Native method

Script to download the sampled dataset used in inference benchmark:
```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/dlrm-v3-dataset.uri
```

## Model download

### Download model through MLCFlow Automation

```
mlcr get-ml-model-dlrm-v3,_mlc,_r2-downloader --outdirname=<path_to_download> -j
```

### Download model through Native method

Script to download the 1TB trained checkpoint:
```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/dlrm-v3-checkpoint.uri
```

## Inference benchmark

```
WORLD_SIZE=8 python main.py --dataset sampled-streaming-100b
```

`WORLD_SIZE` is the number of GPUs used in the inference benchmark.

```
usage: main.py [-h] [--dataset {streaming-100b,sampled-streaming-100b}] [--model-path MODEL_PATH] [--scenario-name {Server,Offline}] [--batchsize BATCHSIZE]
               [--output-trace OUTPUT_TRACE] [--data-producer-threads DATA_PRODUCER_THREADS] [--compute-eval COMPUTE_EVAL] [--find-peak-performance FIND_PEAK_PERFORMANCE]
               [--dataset-path-prefix DATASET_PATH_PREFIX] [--warmup-ratio WARMUP_RATIO] [--num-queries NUM_QUERIES] [--target-qps TARGET_QPS] [--numpy-rand-seed NUMPY_RAND_SEED]
               [--sparse-quant SPARSE_QUANT] [--dataset-percentage DATASET_PERCENTAGE]

options:
  -h, --help            show this help message and exit
  --dataset {streaming-100b,sampled-streaming-100b}
                        name of the dataset
  --model-path MODEL_PATH
                        path to the model checkpoint. Example: /home/username/ckpts/streaming_100b/89/
  --scenario-name {Server,Offline}
                        inference benchmark scenario
  --batchsize BATCHSIZE
                        batch size used in the benchmark
  --output-trace OUTPUT_TRACE
                        Whether to output trace
  --data-producer-threads DATA_PRODUCER_THREADS
                        Number of threads used in data producer
  --compute-eval COMPUTE_EVAL
                        If true, will run AccuracyOnly mode and outputs both predictions and labels for accuracy calcuations
  --find-peak-performance FIND_PEAK_PERFORMANCE
                        Whether to find peak performance in the benchmark
  --dataset-path-prefix DATASET_PATH_PREFIX
                        Prefix to the dataset path. Example: /home/username/
  --warmup-ratio WARMUP_RATIO
                        The ratio of the dataset used to warmup SUT
  --num-queries NUM_QUERIES
                        Number of queries to run in the benchmark
  --target-qps TARGET_QPS
                        Benchmark target QPS. Needs to be tuned for different implementations to balance latency and throughput
  --numpy-rand-seed NUMPY_RAND_SEED
                        Numpy random seed
  --sparse-quant SPARSE_QUANT
                        Whether to quantize sparse arch
  --dataset-percentage DATASET_PERCENTAGE
                        Percentage of the dataset to run in the benchmark
```

## Accuracy test

Set `run.compute_eval` will run the accuracy test and dump prediction outputs in
`mlperf_log_accuracy.json`. To check the accuracy, run

```
python accuracy.py --path path/to/mlperf_log_accuracy.json
```
We use normalized entropy (NE), accuracy, and AUC as the metrics to evaluate the model quality. For accepted submissions, all three metrics (NE, Accuracy, AUC) must be within 99.9% of the reference implementation values. The accuracy for the reference implementation evaluated on 34,996 requests across 10 inference timestamps are listed below:
```
NE: 86.687%
Accuracy: 69.651%
AUC: 78.663%
```
