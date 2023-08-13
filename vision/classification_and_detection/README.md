# MLPerf™ Inference Benchmarks for Image Classification and Object Detection Tasks

This is the reference implementation for MLPerf Inference Classification and Object Detection benchmarks

Please see the below readme files for an automated way to run this benchmark out of the box and do an end-to-end submission with or without docker using the [MLCommons CM](https://github.com/mlcommons/ck/tree/master/cm) language.
1. [Resnet50](README_cm_resnet50.md)
2. [Retinanet](README_cm_retinanet.md)

Alternatively, you can find a short tutorial on how to use this benchmark [here](https://github.com/mlperf/inference/blob/master/vision/classification_and_detection/GettingStarted.ipynb).


## Supported Models

| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | tensorflow | 76.456% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2535873/files/resnet50_v1.pb) | [mlperf](https://github.com/mlperf/training/tree/master/image_classification), [tensorflow](https://github.com/tensorflow/models/tree/master/official/resnet) | fp32 | NHWC. More information on resnet50 v1.5 can be found [here](https://github.com/tensorflow/models/tree/master/official/resnet).||
| resnet50-v1.5 | onnx | 76.456% | imagenet2012 validation | from zenodo: [opset-8](https://zenodo.org/record/2592612/files/resnet50_v1.onnx), [opset-11](https://zenodo.org/record/4735647/files/resnet50_v1.onnx) | [from zenodo](https://zenodo.org/record/2535873/files/resnet50_v1.pb) converted with [this script](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/convert-to-onnx.sh) | fp32 | NCHW, tested on pytorch and onnxruntime |
| resnet50-v1.5 | pytorch | 76.014% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth) | [from TorchVision](https://github.com/pytorch/vision/blob/v0.8.2/torchvision/models/resnet.py) | fp32 | NCHW |
| resnet50-v1.5 | pytorch | 75.790% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/4589637/files/resnet50_INT8bit_quantized.pt) | Edgecortix [quantization script](tools/calibrate_torchvision_model.py) | A: int8, W: uint8 | NCHW |
| resnet50-v1.5 | ncnn | 76.14% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/8073420) | [from TorchVision](https://github.com/pytorch/vision/blob/v0.11.2/torchvision/models/resnet.py) | fp32 | NCHW |
| retinanet 800x800 | pytorch | mAP 0.3755 | OpenImages mlperf validation set resized to 800x800 | [from zenodo](https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth) | from mlperf. [Source Code](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd/model) and [Weights](https://zenodo.org/record/6605272) | fp32 | NCHW |
| retinanet 800x800 | onnx | mAP 0.3757 | OpenImages mlperf validation set resized to 800x800 | [from zenodo](https://zenodo.org/record/6617879/files/resnext50_32x4d_fpn.onnx) | from mlperf converted from the pytorch model. [Source Code](https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd/model) and [Weights](https://zenodo.org/record/6605272) | fp32 | NCHW |

<details>
<summary>Click here to see deprecated models.</summary> 
    
| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| mobilenet-v1 (removed since mlperf-v0.7)| tensorflow | 71.676% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) | fp32 | NHWC |
| mobilenet-v1 quantized (removed since mlperf-v0.7)| tensorflow | 70.694% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224_quant.tgz) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | int8 | NHWC |
| mobilenet-v1 (removed since mlperf-v0.7)| tflite | 71.676% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) | fp32 | NHWC |
| mobilenet-v1 quantized (removed since mlperf-v0.7)| tflite | 70.762% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224_quant.tgz) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) | int8 | NHWC |
| mobilenet-v1 (removed since mlperf-v0.7)| onnx | 71.676% | imagenet2012 validation | from zenodo: [opset-8](https://zenodo.org/record/3157894/files/mobilenet_v1_1.0_224.onnx), [opset-11](https://zenodo.org/record/4735651/files/mobilenet_v1_1.0_224.onnx) | [from tensorflow](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz) converted with [this script](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/convert-to-onnx.sh) | fp32 | NCHW, tested on pytorch and onnxruntime |
| mobilenet-v1 (removed since mlperf-v0.7)| onnx, pytorch | 70.9% | imagenet2012 validation | [from zenodo](https://zenodo.org/record/3353417/files/Quantized%20MobileNet.zip) | ??? | int8 | ??? |
| ssd-mobilenet 300x300 (removed since mlperf-v2.1) | tensorflow | mAP 0.23 | coco resized to 300x300 | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | fp32 | NHWC |
| ssd-mobilenet 300x300 quantized finetuned (removed since mlperf-v2.1) | tensorflow | mAP 0.23594 | coco resized to 300x300 | [from zenodo](https://zenodo.org/record/3252084/files/mobilenet_v1_ssd_8bit_finetuned.tar.gz) | Habana | int8 | ??? |
| ssd-mobilenet 300x300 symmetrically quantized finetuned (removed since mlperf-v2.1)| tensorflow | mAP 0.234 | coco resized to 300x300 | [from zenodo](https://zenodo.org/record/3401714/files/ssd_mobilenet_v1_quant_ft_no_zero_point_frozen_inference_graph.pb) | Habana | int8 | ??? |
| ssd-mobilenet 300x300 (removed since mlperf-v2.1)| pytorch | mAP 0.23 | coco resized to 300x300 | [from zenodo](https://zenodo.org/record/3239977/files/ssd_mobilenet_v1.pytorch) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | fp32 | NHWC |
| ssd-mobilenet 300x300 (removed since mlperf-v2.1)| onnx | mAP 0.23 | coco resized to 300x300 | from zenodo [opset-8](https://zenodo.org/record/3163026/files/ssd_mobilenet_v1_coco_2018_01_28.onnx), [opset-11](https://zenodo.org/record/4735652/files/ssd_mobilenet_v1_coco_2018_01_28.onnx) | [from tensorflow](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) converted using [this script](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/convert-to-onnx.sh) | fp32 | NHWC, tested on onnxruntime, some runtime warnings |
| ssd-mobilenet 300x300 (removed since mlperf-v2.1)| onnx, pytorch | mAP 0.23 | coco resized to 300x300  | [from zenodo](https://zenodo.org/record/3252084/files/mobilenet_v1_ssd_8bit_finetuned.tar.gz) | ??? | int8 | ??? |
| ssd-resnet34 1200x1200 (removed since mlperf-v2.1) | tensorflow | mAP 0.20 | coco resized to 1200x1200| [from zenodo](https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip?download=1) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector/tensorflow), [training model](https://github.com/lji72/inference/tree/tf_ssd_resent34_align_onnx/others/cloud/single_stage_detector/tensorflow) | fp32 | NCHW |
| ssd-resnet34 1200x1200 (removed since mlperf-v2.1)| pytorch | mAP 0.20 | coco resized to 1200x1200 | [from zenodo](https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector/pytorch) | fp32 | NCHW |
| ssd-resnet34 1200x1200 (removed since mlperf-v2.1) | onnx | mAP 0.20 | coco resized to 1200x1200 | from zenodo [opset-8](https://zenodo.org/record/3228411/files/resnet34-ssd1200.onnx) | [from mlperf](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector) converted using the these [instructions](https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch#6-onnx) | fp32 | Converted from pytorch model. |
| ssd-resnet34 1200x1200 (removed since mlperf-v2.1) | onnx | mAP 0.20 | coco resized to 1200x1200 | from zenodo [opset-11](https://zenodo.org/record/4735664/files/ssd_resnet34_mAP_20.2.onnx) | [from zenodo](https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip) converted using [this script](https://github.com/mlcommons/inference/blob/master/vision/classification_and_detection/tools/convert-to-onnx.sh) | fp32 | Converted from the tensorflow model and uses the same interface as the tensorflow model. |
</details>


## Disclaimer
This benchmark app is a reference implementation that is not meant to be the fastest implementation possible.
It is written in python which might make it less suitable for lite models like mobilenet or large number of cpu's.
There is a [C++ implementation](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-cpp) in
[MLCommons CK repository](https://github.com/mlcommons/ck) which currently supports onnxruntime backend.

## Tools for preparing datasets and validating accuracy
The reference implementation includes all required pre-processing of datasets.
It also includes a ```--accuracy``` option to validate accuracy as required by mlperf.
If you are not using the reference implementation, a few scripts will help:
### Prepare the coco dataset
The tool is [here](../../tools/upscale_coco).
You can run it for ssd-mobilenet like:
```
python upscale_coco.py --inputs /data/coco/ --outputs /data/coco-300 --size 300 300 --format png
```
and for ssd-resnet34 like:
```
python upscale_coco.py --inputs /data/coco/ --outputs /data/coco-1200 --size 1200 1200 --format png
```
### Prepare the imagenet dataset
to come.

### Validate accuracy for resnet50 and mobilenet benchmarks
The tool is [here](tools/accuracy-imagenet.py). You can run it like:
```
python tools/accuracy-imagenet.py --mlperf-accuracy-file mlperf_log_accuracy.json --imagenet-val-file /data/imagenet2012/val_map.txt
```

### Validate accuracy for ssd-mobilenet and ssd-resnet34 benchmarks
The tool is [here](tools/accuracy-coco.py). You can run it like:
```
python tools/accuracy-coco.py --mlperf-accuracy-file mlperf_log_accuracy.json --coco-dir /data/coco --use-inv-map
```

## Datasets
| dataset | download link |
| ---- | ---- |
| imagenet2012 (validation) | http://image-net.org/challenges/LSVRC/2012/ |
| openimages | We provide a [script](tools/openimages_mlperf.sh) to download the openimages mlperf validation set. You can download the dataset by going into the tools folder and running `./openimages_mlperf -d <DOWNLOAD_PATH>  -m <MAX_IMAGES>` |
| openimages (calibration) | We also provide a [script](tools/openimages_calibration_mlperf.sh) to download the openimages mlperf validation set. You can download the dataset by going into the tools folder and running `./openimages_calibration_mlperf -d <DOWNLOAD_PATH>`. This requires you to have [calibration list](../../calibration/openimages/openimages_cal_images_list.txt) |
<details>
<summary>Click here to see deprecated models.</summary> 

## Datasets
| dataset | download link |
| ---- | ---- |
| coco (validation) (not needed since mlperf-v2.1) | http://images.cocodataset.org/zips/val2017.zip |
| coco (annotations) (not needed since mlperf-v2.1)| http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
</details>

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
cd ../../loadgen; CFLAGS="-std=c++14" python setup.py develop --user; cd ../vision/classification_and_detection

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

backend is one of [tf|onnxruntime|pytorch|tflite|tvm-onnx|tvm-pytorch]
model is one of [resnet50|retinanet|mobilenet|ssd-mobilenet|ssd-resnet34]
device is one of [cpu|gpu]


For example:

./run_local.sh tf resnet50 gpu
```

### Run as Docker container
```
./run_and_time.sh backend model device

backend is one of [tf|onnxruntime|pytorch|tflite|tvm-onnx|tvm-pytorch]
model is one of [resnet50|retinanet|mobilenet|ssd-mobilenet|ssd-resnet34]
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
./run_local.sh onnxruntime retinanet gpu --count 100 --time 60 --scenario Server --qps 100 --max-latency 0.1

```

If you want run with accuracy pass, try:
```
./run_local.sh onnxruntime retinanet gpu --accuracy --time 60 --scenario Server --qps 100 --max-latency 0.2
```


### Usage
```
usage: main.py [-h]
    [--mlperf_conf ../../mlperf.conf]
    [--user_conf user.conf]
    [--dataset {imagenet,openimages-300-retinanet,openimages-800-retinanet,openimages-1200-retinanet,openimages-800-retinanet-onnx,imagenet_mobilenet,coco,coco-300,coco-1200,coco-1200-onnx,coco-1200-pt,coco-1200-tf}]
    --dataset-path DATASET_PATH [--dataset-list DATASET_LIST]
    [--data-format {NCHW,NHWC}]
    [--profile {defaults,resnet50-tf,resnet50-onnxruntime,retinanet-pytorch,retinanet-onnxruntime,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-pytorch,ssd-resnet34-onnxruntime,resnet50-tvm-onnx,resnet50-tvm-pytorch}]
    [--scenario list of SingleStream,MultiStream,Server,Offline]
    [--max-batchsize MAX_BATCHSIZE]
    --model MODEL [--output OUTPUT] [--inputs INPUTS]
    [--outputs OUTPUTS] [--backend BACKEND] [--threads THREADS]
    [--time TIME] [--count COUNT] [--qps QPS]
    [--max-latency MAX_LATENCY] [--cache CACHE] [--accuracy]
```

```--mlperf_conf```
the mlperf config file to use for rules compliant parameters, defaults to ../../mlperf.conf

```--user_conf```
the user config file to use for user LoadGen settings such as target QPS, defaults to user.conf

```--dataset```
use the specified dataset. Currently we only support ImageNet.

```--dataset-path```
path to the dataset.

```--data-format {NCHW,NHWC}```
data-format of the model (default: the backends prefered format).

```--scenario {SingleStream,MultiStream,Server,Offline}```
comma separated list of benchmark modes.

```--profile {resnet50-tf,resnet50-onnxruntime,retinanet-onnxruntime,retinanet-pytorch,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-onnxruntime,resnet50-tvm-onnx,resnet50-tvm-pytorch}```
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
