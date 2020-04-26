# MLPerf Inference Benchmarks for Recommendation Task

This is the reference implementation for MLPerf Inference benchmarks.

### Supported Models

| model | framework | accuracy | AUC | dataset | trained  | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| dlrm | PyTorch | 78.9% | N/A | [Criteo Kaggle DAC](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)       | N/A                                                                     | fp32 |                          |
| dlrm | PyTorch | 81.07% | N/A | [Criteo Terabyte](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) (0.875) | [weights](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt) | fp32 | --max-ind-range=10000000 --data-sub-sample-rate=0.875 |
| dlrm | PyTorch | N/A | 80.25% | [Criteo Terabyte](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)         | [weights](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt)   | fp32 | --max-ind-range=40000000 |

### Disclaimer
This benchmark app is a reference implementation that is not meant to be the fastest implementation possible.
It is written in python which might make it less suitable for large number of CPU's.

The reference implementation includes all required pre-processing of datasets.
It also includes a ```--accuracy``` option to validate accuracy and AUC as required by mlperf.
If you are not using the reference implementation, a few scripts will help:

## Prerequisites and Installation
We support [PyTorch](http://pytorch.org) and might add TensorFlow backend implementation.

The following steps are **only** needed if you run the benchmark **without Docker**.

Python 3.5, 3.6 or 3.7 is supported and we recommend to use Anaconda (See [Dockerfile](Dockerfile.cpu) for a minimal Anaconda install).

Install the desired backend. For pytoch:
```
pip install torch torchvision
pip install scikit-learn
pip install numpy
pip install pydot
pip install torchviz
pip install protobuf
pip install tqdm
```

### Prepare the code and dataset
1. Download or clone the MLPerf [inference](https://github.com/mlperf/inference) and [trainining](https://github.com/mlperf/training) code
```
cd $HOME
mkdir ./mlperf && cd ./mlperf
git clone --recurse-submodules https://github.com/mlperf/training.git
git clone --recurse-submodules https://github.com/mlperf/inference.git
export DLRM_DIR=$HOME/mlperf/training/recommendation
```
2. Download pre-trained model weights
```
cd $HOME/mlperf/inference/v0.5/recommendation
mkdir ./model && cd ./model
mv <downloaded_file> dlrm_terabyte.pytorch
export MODEL_DIR=./model
```
3. Download Criteo dataset
```
cd $HOME/mlperf/inference/v0.5/recommendation
mkdir ./criteo && cd ./criteo
mv <downloaded_file> ./
export DATA_DIR=./criteo
```
4. Build and install the loadgen
```
cd $HOME/mlperf/inference/loadgen
CFLAGS="-std=c++14" python setup.py develop --user
```

### More information about the datasets
| dataset | download link |
| ---- | ---- |
| Criteo Kaggle DAC | https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/ |
| Criteo Terabyte   | https://labs.criteo.com/2013/12/download-terabyte-click-logs/ |

1. The Criteo Kaggle DAC dataset is composed of 7 days, which are stored in file: `train.txt`. This file is expected by the code.


2. The Criteo Terabyte dataset is stored in several files corresponding to 24 days: `day_0.gz, day_1.gz, ..., day_23.gz`. Please unzip all the files to obtain the text files `day_0, day_1, ..., day_23` expected by the code.
```
gunzip day_{0..23}.gz
```

3. The Criteo fake dataset can be created in place of the real datasets in order to facilitate debugging and testing. We provide a fake (random) data generator that can be used to quickly generate data samples in a format compatible with both original and mlperf binary loaders. Please use the following script in `./tools` to quickly create random samples for the corresponding models, which will be placed into `./fake_criteo` directory.
```
./make_fake_criteo.sh [kaggle|terabyte0875|terabyte]
```

## Running the benchmark

Download and install all the pre-requisites. Both local and docker environment need to set 3 environment variables:
```
export DATA_DIR=YourCriteoTerabyteLocation
export MODEL_DIR=YourModelFileLocation
export DLRM_DIR=YourDLRMSourceLocation
```

### Run local
```
./run_local.sh backend model dataset device [options]

backend is one of [pytorch]
model is one of [dlrm]
dataset is one of [kaggle|terabyte]
device is one of [cpu|gpu]
options are arguments that are passed along
```

For example, to run on CPU you may choose to use:

1. Criteo Kaggle DAC
```
./run_local.sh pytorch dlrm kaggle cpu --accuracy --scenario SingleStream
```

2. Criteo Terabyte (0.875)
```
./run_local.sh pytorch dlrm terabyte cpu --accuracy --scenario SingleStream --max-ind-range=10000000 --data-sub-sample-rate=0.875 [--mlperf-bin-loader]
```
3. Criteo Terabyte
```
./run_local.sh pytorch dlrm terabyte cpu --accuracy --scenario SingleStream  --max-ind-range=40000000 [--mlperf-bin-loader]
```
Note that the code support (i) original and (ii) mlperf binary loader, that have slightly different performance characteristics. The latter loader can be enabled by adding `--mlperf-bin-loader` to the command line.

Note that this script will pre-process the data during the first run and reuse it over sub-sequent runs. The pre-processing of data can take a significant amount of time during the first run.

In order to use GPU(s), select the number of GPUs with the environment variable `CUDA_VISIBLE_DEVICES`, and run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./run_local.sh pytorch dlrm terabyte gpu --accuracy
```

### Run as Docker container
TBD

### Validate accuracy for dlrm benchmark
TBD

### Examples for testing
During development running the full benchmark is unpractical. Here are some options to help:

```--count``` limits the number of items in the dataset used for accuracy pass

```--duration``` limits the time the benchmark runs

```--max-latency``` the latency used for Server mode

```--accuracy``` enables accuracy pass

So if you want to tune for example Server mode, try:
```
./run_local.sh pytorch dlrm terabyte cpu --count 100 --duration 60000 --scenario Server --target-qps 200 --max-latency 0.1
or
./run_local.sh pytorch dlrm terabyte cpu --count 100 --duration 60000 --scenario Server --target-qps 100 --max-latency 0.1

```

If you want run with accuracy pass, try:
```
./run_local.sh pytorch dlrm terabyte cpu --accuracy --duration 60000 --scenario Server --target-qps 100 --max-latency 0.2
```

### Usage
```
usage: main.py [-h]
    [--config ../mlperf.conf]
    [--model MODEL] --model-path MODEL_PATH
    [--dataset {kaggle,terabyte}] --dataset-path DATASET_PATH
    [--profile {defaults,dlrm-kaggle-pytorch,dlrm-terabyte-pytorch}]
    [--scenario SCENARIO]
    [--max-ind-range MAX_IND_RANGE] [--data-sub-sample-rate DATA_SUB_SAMPLE_RATE]
    [--max-batchsize MAX_BATCHSIZE] [--mlperf-bin-loader]
    [--output OUTPUT] [--inputs INPUTS] [--outputs OUTPUTS]
    [--backend BACKEND] [--use-gpu] [--threads THREADS]
    [--duration TIME_IN_MS] [--count COUNT] [--target-qps QPS]
    [--max-latency MAX_LATENCY] [--samples-per-query NUM_SAMPLES] [--cache CACHE]
    [--accuracy] [--find-peak-performance]
```

```--config```
the mlperf config file to use, defaults to v0.5/mlperf.conf

```--model```
model name, i.e. dlrm

```--model-path MODEL_PATH```
path to the file with model weights.

```--dataset```
use the specified dataset. Currently we only support Criteo Terabyte.

```--dataset-path```
path to the dataset.

```--scenario {SingleStream,MultiStream,Server,Offline}```
benchmarking mode to be used.

```--profile {dlrm-kaggle-pytorch,dlrm-terabyte-pytorch}```
this fills in default command line options with the once specified in the profile. Command line options that follow may override the those.

```--backend```
only the PyTorch backedn is currently supported. However, we expect to add TensorFlow backend in the future.

```--max-ind-range```
the maximum number of vectors allowed in an embedding table.

```--data-sub-sample-rate```
the rate of sub-sampling of negative samples, either 0.875 or 0.0.

```--max-batchsize MAX_BATCHSIZE```
maximum batchsize we generate to backend (default: 128). If the query contains a very large number of samples it will be broken up into smaller mini-batches of `MAX_BATCHSIZE` samples before forwarding it to the model.

```--mlperf-bin-loader```
flag that enables mlperf binary loader to be used.

```--output OUTPUT```
location of the JSON output.

```--backend BACKEND```
which backend to use. Currently supported is tensorflow, onnxruntime, PyTorch and tflite.

```--use-gpu```
flag that enables use of GPU. The number of GPUs used is controlled by `CUDA_VISIBLE_DEVICES` environment variable.

```--threads THREADS```
number of worker threads to use (default: the number of processors in the system).

```--duration```
duration of the benchmark run in milliseconds (ms).

```--count COUNT```
number of samples from the dataset we use (default: use all samples in the dataset).

```--target-qps QPS```
target/expected QPS for Server and Offline scenarios.

```--max-latency MAX_LATENCY```
comma separated list of which latencies (in seconds) we try to reach in the 99 percentile (deault: 0.01,0.05,0.100).

```--samples-per-query```
number of samples per query in multi-stream scenario.

```--accuracy```
perform inference on the entire dataset to validate achieved model accuracy/AUC metric.

```--find-peak-performance```
TBD

## License

[Apache License 2.0](LICENSE)
