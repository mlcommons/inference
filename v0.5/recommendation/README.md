# MLPerf Inference Benchmarks for Recommendation Task

This is the reference implementation for MLPerf Inference benchmarks.

## Supported Models

| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| dlrm | pytorch | TBD% | Criteo Terabyte (sub-sampled=0.875) | [from AWS S3](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt) | [from Facebook Research Github](https://github.com/facebookresearch/dlrm) | fp32 | --max-ind-range=10000000 |
| dlrm | pytorch | TBD% | Criteo Terabyte | [from AWS S3](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt) | [from Facebook Research Github](https://github.com/facebookresearch/dlrm) | fp32 | --max-ind-range=40000000 |


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
cd mlperf/inference/v0.5/recommendation
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
/run_local.sh pytorch dlrm terabyte cpu --accuracy
```
Note that this script will pre-process the data during the first run and reuse it over sub-sequent runs. The pre-processing of data can take a significant amount of time during the first run.  

### Validate accuracy for dlrm benchmark
TBD

## Datasets
| dataset | download link |
| ---- | ---- |
| Criteo Terabyte | https://labs.criteo.com/2013/12/download-terabyte-click-logs/ |

## Prerequisites and Installation
We support [pytoch](http://pytorch.org) backend's with the same benchmark tool. We expect to add TensorFlow implementation in the future
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
cd ../../loadgen; CFLAGS="-std=c++14" python setup.py develop --user; cd ../v0.5/classification_and_detection

python setup.py develop
```


## Running the benchmark
### One time setup

Download the model and dataset for the model you want to benchmark.

Both local and docker environment need to set 2 environment variables:
```
export MODEL_DIR=YourModelFileLocation
export DATA_DIR=YourImageNetLocation
```


### Run local
```
./run_local.sh backend model device

backend is one of [tf|onnxruntime|pytorch|tflite]
model is one of [resnet50|mobilenet|ssd-mobilenet|ssd-resnet34]
device is one of [cpu|gpu]


For example:

./run_local.sh tf resnet50 gpu
```

### Run as Docker container
```
./run_and_time.sh backend model device

backend is one of [tf|onnxruntime|pytorch|tflite]
model is one of [resnet50|mobilenet|ssd-mobilenet|ssd-resnet34]
device is one of [cpu|gpu]

For example:

./run_and_time.sh tf resnet50 gpu
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
./run_local.sh tf resnet50 gpu --count 100 --time 60 --scenario Server --qps 200 --max-latency 0.1
or
./run_local.sh tf ssd-mobilenet gpu --count 100 --time 60 --scenario Server --qps 100 --max-latency 0.1

```

If you want run with accuracy pass, try:
```
./run_local.sh tf ssd-mobilenet gpu --accuracy --time 60 --scenario Server --qps 100 --max-latency 0.2
```


### Usage
```
usage: main.py [-h]
    [--config ../mlperf.conf]
    [--dataset {imagenet,imagenet_mobilenet,coco,coco-300,coco-1200,coco-1200-onnx,coco-1200-pt,coco-1200-tf}]
    --dataset-path DATASET_PATH [--dataset-list DATASET_LIST]
    [--data-format {NCHW,NHWC}]
    [--profile {defaults,resnet50-tf,resnet50-onnxruntime,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-pytorch,ssd-resnet34-onnxruntime}]
    [--scenario list of SingleStream,MultiStream,Server,Offline]
    [--max-batchsize MAX_BATCHSIZE]
    --model MODEL [--output OUTPUT] [--inputs INPUTS]
    [--outputs OUTPUTS] [--backend BACKEND] [--threads THREADS]
    [--time TIME] [--count COUNT] [--qps QPS]
    [--max-latency MAX_LATENCY] [--cache CACHE] [--accuracy]
```

```--config```
the mlperf config file to use, defaults to v0.5/mlperf.conf

```--dataset```
use the specified dataset. Currently we only support ImageNet.

```--dataset-path```
path to the dataset.

```--data-format {NCHW,NHWC}```
data-format of the model (default: the backends prefered format).

```--scenario {SingleStream,MultiStream,Server,Offline}```
comma separated list of benchmark modes.

```--profile {resnet50-tf,resnet50-onnxruntime,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-onnxruntime}```
this fills in default command line options with the once specified in the profile. Command line options that follow may override the those.

```--model MODEL```
the model file.

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
Number of images the dataset we use (default: use all images in the dataset).

```--qps QPS```
Expected QPS.

```--max-latency MAX_LATENCY```
comma separated list of which latencies (in seconds) we try to reach in the 99 percentile (deault: 0.01,0.05,0.100).

```--max-batchsize MAX_BATCHSIZE```
maximum batchsize we generate to backend (default: 128).


## License

[Apache License 2.0](LICENSE)
