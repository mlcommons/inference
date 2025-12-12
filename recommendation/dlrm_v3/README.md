# MLPerf Inference reference implementation for DLRMv3

## Install dependencies and build loadgen

```
sh setup.sh
```

## Dataset download

TODO: pending MLPerf system setup

## Inference benchmark

```
WORLD_SIZE=8 python main.py --dataset sampled-streaming-100b
```

`WORLD_SIZE` is the number of GPUs used in the inference benchmark.

```
usage: main.py [-h] [--dataset {streaming-100b,sampled-streaming-100b}] [--model-path MODEL_PATH] [--scenario-name {SingleStream,MultiStream,Server,Offline}] [--batchsize BATCHSIZE]
               [--output-trace OUTPUT_TRACE] [--data-producer-threads DATA_PRODUCER_THREADS] [--compute-eval COMPUTE_EVAL] [--find-peak-performance FIND_PEAK_PERFORMANCE]
               [--dataset-path-prefix DATASET_PATH_PREFIX] [--warmup-ratio WARMUP_RATIO] [--num-queries NUM_QUERIES] [--target-qps TARGET_QPS] [--numpy-rand-seed NUMPY_RAND_SEED]
               [--sparse-quant SPARSE_QUANT] [--dataset-percentage DATASET_PERCENTAGE]

options:
  -h, --help            show this help message and exit
  --dataset {streaming-100b,sampled-streaming-100b}
                        name of the dataset
  --model-path MODEL_PATH
                        path to the model checkpoint. Example: /home/username/ckpts/streaming_100b/89/
  --scenario-name {SingleStream,MultiStream,Server,Offline}
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
