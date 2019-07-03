# MLPerf Inference benchmark for image classification and object detection

This is the reference implementation for MLPerf Inference benchmarks.

You can find a short tutorial how to use this benchmark [here](https://github.com/mlperf/inference/blob/master/cloud/image_classification/GettingStarted.ipynb).

## Supported Models

| model | framework | accuracy | dataset | model link | model source | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | tensorflow | 74.97% (should be 76.47%) | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2535873/files/resnet50_v1.pb) | [mlperf](https://github.com/mlperf/training/tree/master/image_classification), [tensorflow](https://github.com/tensorflow/models/tree/master/official/resnet) | NHWC. More information on resnet50 v1.5 can be found [here](https://github.com/tensorflow/models/tree/master/official/resnet).||
| resnet50-v1.5 | onnx, pytorch | 74.97%, should be 76.47% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2592612/files/resnet50_v1.onnx) | [from zenodo](https://zenodo.org/record/2535873/files/resnet50_v1.pb) converted with [this script](https://github.com/mlperf/inference/blob/master/cloud/image_classification/tools/resnet50-to-onnx.sh) | NCHW, tested on pytorch and onnxruntime |
| mobilenet-v1 | tensorflow | 69.86, should be 70.9% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2271498/files/mobilenet_v1_1.0_224.tgz) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) | NHWC |
| mobilenet-v1 | onnx, pytorch | 69.86, should be 70.9% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) converted with [this script](https://github.com/mlperf/inference/blob/master/cloud/image_classification/tools/mobilenet-to-onnx.sh) | NCHW, tested on pytorch and onnxruntime |
| ssd-mobilenet 300x300 | tensorflow | mAP 0.20 | coco resized to 300x300 | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | NHWC |
| ssd-mobilenet 300x300 | pytorch | mAP 0.20 | coco resized to 300x300 | [from zenodo](https://zenodo.org/record/3239977/files/ssd_mobilenet_v1.pytorch) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | NHWC |
| ssd-mobilenet 300x300 | onnx | mAP 0.20 | coco resized to 300x300 | [from zenodo](https://zenodo.org/record/3163026/files/ssd_mobilenet_v1_coco_2018_01_28.onnx) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) converted with [this script](https://github.com/mlperf/inference/blob/master/cloud/image_classification/tools/ssd-mobilenet-to-onnx.sh) | NHWC, tested on onnxruntime, some runtime warnings |
| ssd-resnet34 1200x1200 | tensorflow | mAP 0.20 | coco resized to 1200x1200| [from zenodo](https://zenodo.org/record/3262269/files/ssd_resnet34_mAP_20.2.pb), [checkpoint](https://zenodo.org/record/3262269) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector/tensorflow) | NHWC |
| ssd-resnet34 1200x1200 | pytorch | mAP 0.20 | coco resized to 1200x1200 | [from zenodo](https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector/pytorch) | NCHW |
| ssd-resnet34 1200x1200 | onnx | mAP 0.20 | coco resized to 1200x1200 | [from zenodo](https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector) converted using the these [instructions](https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch#6-onnx) | Works but needs more testing |

TODO: add instructions to resize coco dataset

## Disclaimer
This is an early version of the benchmark to get feedback from others.
Do expect some changes.

The benchmark is a reference implementation that is not meant to be the fastest implementation possible.
It is written in python which might make it less suitable for lite models like mobilenet or large number of cpu's.
We are thinking to provide a c++ implementation with identical functionality in the near future.

## Datasets
| dataset | download link | 
| ---- | ---- | 
| imagenet2012 (validation) | http://image-net.org/challenges/LSVRC/2012/ | 
| coco (validation) | http://images.cocodataset.org/zips/val2017.zip | 

### Using Collective Knowledge (CK)

Alternatively, you can download the datasets using the [Collective Knowledge](http://cknowledge.org)
framework (CK) for collaborative and reproducible research.

First, install CK and pull its repositories containing dataset packages:
```bash
$ python -m pip install ck --user
$ ck version
V1.9.8.1
$ ck pull repo:ck-env
```

#### ImageNet 2012 validation dataset
Download the original dataset and auxiliaries:
```bash
$ ck install package --tags=image-classification,dataset,imagenet,val,original,full
$ ck install package --tags=image-classification,dataset,imagenet,aux
```
Copy the labels next to the images:
```bash
$ ck locate env --tags=image-classification,dataset,imagenet,val,original,full
/home/dvdt/CK-TOOLS/dataset-imagenet-ilsvrc2012-val
$ ck locate env --tags=image-classification,dataset,imagenet,aux
/home/dvdt/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux
$ cp `ck locate env --tags=aux`/val.txt `ck locate env --tags=val`/val_map.txt
```

#### COCO 2017 validation dataset
```bash
$ ck install package --tags=object-detection,dataset,coco,2017,val,original
$ ck locate env --tags=object-detection,dataset,coco,2017,val,original
/home/dvdt/CK-TOOLS/dataset-coco-2017-val
```

## Prerequisites and Installation
We support [tensorfow+tflite](https://github.com/tensorflow/tensorflow), [onnxruntime](https://github.com/Microsoft/onnxruntime)  and [pytoch](http://pytorch.org) backend's with the same benchmark tool.
Support for other backends can be easily added.

The following steps are **only** needed if you run the benchmark **without Docker**.

Python 3.5, 3.6 or 3.7 is supported and we recommend to use Anaconda (See [Dockerfile](Dockerfile.cpu) for a minimal Anaconda install).

Install the desired backend.
For tensorflow:
```
pip install tensorflow or pip install tensorflow-gpu
```
For onnxruntime:
```
pip install onnxruntime or pip install onnxruntime-gpu
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
./run_local.sh tf resnet50 gpu --count 100 --time 60 --scenario Server --qps 200 --max-latency 0.2
or
./run_local.sh tf ssd-mobilenet gpu --count 100 --time 60 --scenario Server --qps 100 --max-latency 0.2

```

If you want run with accuracy pass, try:
```
./run_local.sh tf ssd-mobilenet gpu --accuracy --count 100 --time 60 --scenario Server --qps 100 --max-latency 0.2
```


### Usage
```
usage: main.py [-h]
    [--dataset {imagenet,imagenet_mobilenet,coco,coco-300,coco-1200,coco-1200-onnx,coco-1200-pt,coco-1200-tf}]
    --dataset-path DATASET_PATH [--dataset-list DATASET_LIST]
    [--data-format {NCHW,NHWC}]
    [--profile {defaults,resnet50-tf,resnet50-onnxruntime,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-pytorch,ssd-resnet34-onnxruntime}]
    [--scenario list of SingleStream,MultiStream,Server,Offline]
    [--queries-single QUERIES_SINGLE]
    [--queries-offline QUERIES_OFFLINE]
    [--queries-multi QUERIES_MULTI] [--max-batchsize MAX_BATCHSIZE]
    --model MODEL [--output OUTPUT] [--inputs INPUTS]
    [--outputs OUTPUTS] [--backend BACKEND] [--threads THREADS]
    [--time TIME] [--count COUNT] [--qps QPS]
    [--max-latency MAX_LATENCY] [--cache CACHE] [--accuracy]
```

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

```--queries-single QUERIES_SINGLE```
queries to use for SingleStream scenario (default: 1024).

```--queries-offline QUERIES_OFFLINE```
queries to use for Offline scenario (default: 24576).

```--queries-multi QUERIES_MULTI```
queries to use for MultiStream scenario (default: 24576).

```--max-batchsize MAX_BATCHSIZE```
maximum batchsize we generate to backend (default: 128).


## License

[Apache License 2.0](LICENSE)
