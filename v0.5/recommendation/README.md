# MLPerf Inference Benchmarks for Recommendation Task

This is the reference implementation for MLPerf Inference benchmarks.

## Supported Models

| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| dlrm | PyTorch | TBD% | Criteo Terabyte (sub-sampled=0.875) | [from AWS S3](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt) | [from Facebook Research Github](https://github.com/facebookresearch/dlrm) | fp32 | --max-ind-range=10000000 |
| dlrm | PyTorch | TBD% | Criteo Terabyte | [from AWS S3](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt) | [from Facebook Research Github](https://github.com/facebookresearch/dlrm) | fp32 | --max-ind-range=40000000 |


## Disclaimer
This benchmark app is a reference implementation that is not meant to be the fastest implementation possible.
It is written in python which might make it less suitable for large number of cpu's.

## Tools for preparing datasets and validating accuracy
The reference implementation includes all required pre-processing of datasets.
It also includes a ```--accuracy``` option to validate accuracy as required by mlperf.
If you are not using the reference implementation, a few scripts will help:

### Prepare the Criteo Terabyte dataset
1. Download [Criteo Terabyte dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
```
cd $HOME/mlperf/inference/v0.5/recommendation
export DATA_DIR=./criteo
```
2. Download [DLRM model weights](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt) 
```
export MODEL_DIR=./model
cd $MODEL_DIR
mv <downloaded_file> dlrm_terabyte.pytorch
cd ..
```
3. Download or clone the DLRM source code from [MLPerf trainining](https://github.com/mlperf/training) 
```
cd ../../../
git clone https://github.com/mlperf/training.git
export DLRM_DIR=$HOME/mlperf/training/recommendation
ls mlperf/training/recommendation
```
4. Select the run parameters, for instance
```
export EXTRA_OPS="--time 10 --max-latency 0.2 --count=100 --scenario SingleStream [--max-ind-range=10000000 --data-sub-sample-rate=0.875] [--mlperf-bin-loader]"
```
or
```
export EXTRA_OPS="--time 10 --max-latency 0.2 --count=100 --scenario SingleStream  --max-ind-range=40000000 [--mlperf-bin-loader]"
```
Note that the code support (i) original and (ii) mlperf binary loader, that have slightly different performance characteristics.

5. Run the following script to perform the inference runs
```
/run_local.sh pytorch dlrm terabyte cpu|gpu --accuracy
```
Note that this script will pre-process the data during the first run and reuse it over sub-sequent runs. The pre-processing of data can take a significant amount of time during the first run.  

Also, if running on GPU then the number of GPUs to be used is controlled by environment variable 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### Validate accuracy for dlrm benchmark
TBD

## Datasets
| dataset | download link |
| ---- | ---- |
| Criteo Terabyte | https://labs.criteo.com/2013/12/download-terabyte-click-logs/ |

Note that in order to facilitate debugging and testing, we provide a fake (random) data generator that can be used to quickly generate data samples in a format compatible with both original and mlperf binary loaders. Please use the following 
```
./tools/make_fake_data.sh [terabyte0875|terabyte]
```
to quickly create random data samples for the corresponding models. 

## Prerequisites and Installation
We support [PyTorch](http://pytorch.org) and expect to add TensorFlow backend implementation.
Support for other backends can be easily added.

The following steps are **only** needed if you run the benchmark **without Docker**.

Python 3.5, 3.6 or 3.7 is supported and we recommend to use Anaconda (See [Dockerfile](Dockerfile.cpu) for a minimal Anaconda install).

Install the desired backend.
For pytoch:
```
pip install torch torchvision
```

Build and install the benchmark:
```
cd ../../loadgen; CFLAGS="-std=c++14" python setup.py develop --user; cd ../v0.5/recommendation

python setup.py develop
```


## Running the benchmark
### One time setup

Download the model and dataset for the model you want to benchmark.

Both local and docker environment need to set 3 environment variables:
```
export DATA_DIR=YourCriteoTerabyteLocation
export MODEL_DIR=YourModelFileLocation
export DLRM_DIR=YourDLRMSourceLocation
```


### Run local
```
./run_local.sh backend model dataset device

backend is one of [pytorch]
model is one of [dlrm]
dataset is one of [kaggle|terabyte]
device is one of [cpu|gpu]


For example:

./run_local.sh pytorch dlrm terabyte gpu
```

### Run as Docker container
```
./run_and_time.sh backend model dataset device

backend is one of [pytorch]
model is one of [dlrm]
dataset is one of [kaggle|terabyte]
device is one of [cpu|gpu]

For example:

./run_and_time.sh pytorch dlrm terabyte gpu
```
This will build and run the benchmark.

### Examples for testing
During development running the full benchmark is unpractical. Some options to help:

```--count``` limits the number of items in the dataset used for accuracy pass

```--time``` limits the time the benchmark runs

```--accuracy``` enables accuracy pass

```--max-latency``` the latency used for Server mode

So if you want to tune for example Server mode, try:
```
./run_local.sh pytorch dlrm terabyte gpu --count 100 --time 60 --scenario Server --qps 200 --max-latency 0.1
or
./run_local.sh pytorch dlrm terabyte gpu --count 100 --time 60 --scenario Server --qps 100 --max-latency 0.1

```

If you want run with accuracy pass, try:
```
./run_local.sh pytorch dlrm terabyte gpu --accuracy --time 60 --scenario Server --qps 100 --max-latency 0.2
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
    [--mlperf-bin-loader] 
    [--max-batchsize MAX_BATCHSIZE]
    [--output OUTPUT] [--inputs INPUTS] [--outputs OUTPUTS] 
    [--backend BACKEND] [--use-gpu]
    [--threads THREADS] [--time TIME] [--count COUNT] [--qps QPS]
    [--max-latency MAX_LATENCY] [--cache CACHE] 
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
TBD

```--profile {dlrm-kaggle-pytorch,dlrm-terabyte-pytorch}```
this fills in default command line options with the once specified in the profile. Command line options that follow may override the those.

```--backend```
only the PyTorch backedn is currently supported. However, we expect to add TensorFlow backend in the future.

```--use-gpu```
flag that enables use of GPU. The number of GPUs used is controlled by CUDA_VISIBLE_DEVICES environment variable.

```--max-ind-range```
the maximum number of vectors allowed in an embedding table.

```--data-sub-sample-rate```
the rate of sub-sampling of negative samples, either 0.875 or 0.0.

```--mlperf-bin-loader```
flag that enables mlperf binary loader to be used.

```--inputs INPUTS```
comma separated input name list in case the model format does not provide the input names. This is needed for tensorflow since the graph does not specify the inputs.

```--outputs OUTPUTS```
comma separated output name list in case the model format does not provide the output names. This is needed for tensorflow since the graph does not specify the outputs.

```--output OUTPUT]```
location of the JSON output.

```--backend BACKEND```
which backend to use. Currently supported is tensorflow, onnxruntime, pytorch and tflite.

```--threads THREADS```
number of worker threads to use (default: the number of processors in the system).

```--count COUNT```
number of images the dataset we use (default: use all images in the dataset).

```--qps QPS```
expected QPS.

```--time```
time to scan in seconds

```--cache```
use cache

```--max-latency MAX_LATENCY```
comma separated list of which latencies (in seconds) we try to reach in the 99 percentile (deault: 0.01,0.05,0.100).

```--max-batchsize MAX_BATCHSIZE```
maximum batchsize we generate to backend (default: 128).


## License

[Apache License 2.0](LICENSE)
