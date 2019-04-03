# MobileNet via TensorFlow Lite

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [The anatomy of the benchmark](#anatomy)

<a name="installation"></a>
## Installation instructions

Please follow the common [installation instructions](../README.md#installation) first.

**NB:** See [`ck-tensorflow:program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite) for more details.

### Install TensorFlow Lite (TFLite)

Install TFLite from source:
```
$ ck install package:lib-tflite-1.13.1-src-static [--target_os=android23-arm64]
```

You can also install TFLite from a prebuilt binary package for your target e.g.:
```
$ ck list package:lib-tflite-prebuilt*
lib-tflite-prebuilt-0.1.7-linux-aarch64
lib-tflite-prebuilt-0.1.7-linux-x64
lib-tflite-prebuilt-0.1.7-android-arm64
$ ck install package:lib-tflite-prebuilt-0.1.7-android-arm64 [--target_os=android23-arm64]
```
**NB:** Currently we have no TFLite 1.13.1 prebuilt packages.
Please [let us know](info@dividiti.com) if you would like us to create some.


### Install the MobileNet models for TFLite

To select interactively from one of the non-quantized and quantized MobileNets-v1-1.0-224 models adopted for MLPerf Inference v0.5:
```
$ ck install package --tags=model,tflite,mlperf,mobilenet
```

To install the [non-quantized model](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) directly:
```
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized
```

To install the [quantized model](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) directly:
```
$ ck install package --tags=model,tflite,mlperf,mobilenet,quantized
```

#### Bonus

##### Install the ResNet50 model

To install the MLPerf ResNet50-v1.5 model:
```bash
$ ck install package --tags=model,tflite,mlperf,resnet
More than one package or version found:

 0) model-tflite-resnet50-mlperf-convert-from-tf  Version 1.5  (35e84375ac48dcb1)
 1) model-tflite-resnet50-mlperf  Version 1.5  (d60d4e9a84151271)
```
Option 0 downloads the TF model and converts it to TFLite. Option 1 uses a pre-converted TFLite model.

You can benchmark ResNet exactly in the same way as MobileNet.
Just replace `mobilenet` with `resnet` in the [benchmarking instructions](#benchmarking) below.

##### Install other MobileNets models
You can also install any other MobileNets model compatible with TFLite as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet,tflite
```

### Compile the TFLite image classification client
```
$ ck compile program:image-classification-tflite [--target_os=android23-arm64]
```

### Run the TFLite image classification client once

Run the client (if required, connect an Android device to your host machine via USB):

- with the non-quantized model:
```
$ ck run program:image-classification-tflite [--target_os=android23-arm64]
...
*** Dependency 3 = weights (TensorFlow-Python model and weights):
...
    Resolved. CK environment UID = 4edbb2648a48d94d (version 1_1.0_224_2018_08_02)
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.84 - (65) n01751748 sea snake
0.08 - (58) n01737021 water snake
0.04 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.000860s
All images loaded in 0.007685s
All images classified in 0.173653s
Average classification time: 0.173653s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

- with the quantized model:
```
$ ck run program:image-classification-tflite [--target_os=android23-arm64]
...
*** Dependency 3 = weights (TensorFlow-Python model and weights):
...
    Resolved. CK environment UID = 3f0ca5c4d25b4ea3 (version 1_1.0_224_quant_2018_08_02)
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.80 - (65) n01751748 sea snake
0.09 - (34) n01665541 leatherback turtle, leatherback, leather...
0.05 - (58) n01737021 water snake
0.03 - (54) n01729322 hognose snake, puff adder, sand viper
0.00 - (33) n01664065 loggerhead, loggerhead turtle, Caretta c...
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.000589s
All images loaded in 0.000290s
All images classified in 0.450257s
Average classification time: 0.450257s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance
```
$ ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-performance \
--tags=image-classification,mlperf,mobilenet,tflite,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```
$ ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
...
Processing batches...

Batch 1 of 2

Batch loaded in 0.00802251 s
Batch classified in 0.16831 s

Batch 2 of 2

Batch loaded in 0.00776105 s
Batch classified in 0.0762354 s
...
Summary:
-------------------------------
Graph loaded in 0.000663s
All images loaded in 0.015784s
All images classified in 0.244545s
Average classification time: 0.076235s
Accuracy top 1: 0.5 (1 of 2)
Accuracy top 5: 1.0 (2 of 2)
--------------------------------
```

### Benchmark the accuracy
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-tflite-accuracy \
--tags=image-classification,mlperf,mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="anatomy"></a>
## The anatomy of the benchmark

While the componentized nature of CK workflows streamlines
[installation](#installation) and [benchmarking](#benchmarking), it also makes
it less obvious what the components are and where they are stored. This section
describes the anatomy of the benchmark in terms of its components.

### Model

To view the CK entry of an installed model:
```
$ ck search env --tags=model,tflite,mlperf,mobilenet,quantized
local:env:3f0ca5c4d25b4ea3
```

To view more information about the model's CK entry:
```
$ ck show env --tags=model,tflite,mlperf,mobilenet,quantized
Env UID:         Target OS: Bits: Name:                                                                Version:                   Tags:

3f0ca5c4d25b4ea3   linux-64    64 TensorFlow model and weights (mobilenet-v1-1.0-224-quant-2018_08_02) 1_1.0_224_quant_2018_08_02 2018_08_02,64bits,downloaded,host-os-linux-64,mlperf,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,model,nhwc,python,quantised,quantized,target-os-linux-64,tensorflowmodel,tf,tflite,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.0,v1.1.0.224.0.2018,v1.1.0.224.0.2018.8,v1.1.0.224.0.2018.8.2,weights
```

To view the environment variables set up by the model's CK entry:
```
$ ck cat `ck search env --tags=model,tflite,mlperf,mobilenet,quantized`
#! /bin/bash
#
# --------------------[ TensorFlow model and weights (mobilenet-v1-1.0-224-quant-2018_08_02) ver. 1_1.0_224_quant_2018_08_02, /home/anton/CK_REPOS/local/env/3f0ca5c4d25b4ea3/env.sh ]--------------------
# Tags: 2018_08_02,64bits,downloaded,host-os-linux-64,mlperf,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,model,nhwc,python,quantised,quantized,target-os-linux-64,tensorflowmodel,tf,tflite,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.0,v1.1.0.224.0.2018,v1.1.0.224.0.2018.8,v1.1.0.224.0.2018.8.2,weights
#
# CK generated script

if [ "$1" != "1" ]; then if [ "$CK_ENV_TENSORFLOW_MODEL_SET" == "1" ]; then return; fi; fi

# Soft UOA           = model.tensorflow.py (439b9f1757f27091)  (tensorflowmodel,model,weights,python,tf,tflite,nhwc,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,2018_08_02,quantized,quantised,mlperf,downloaded,host-os-linux-64,target-os-linux-64,64bits,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.0,v1.1.0.224.0.2018,v1.1.0.224.0.2018.8,v1.1.0.224.0.2018.8.2)
# Host OS UOA        = linux-64 (4258b5fe54828a50)
# Target OS UOA      = linux-64 (4258b5fe54828a50)
# Target OS bits     = 64
# Tool version       = 1_1.0_224_quant_2018_08_02
# Tool split version = [1, 1, 0, 224, 0, 2018, 8, 2]

export CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT=224
export CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH=224
export CK_ENV_TENSORFLOW_MODEL_INPUT_LAYER_NAME=input
export CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER=1.0
export CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION=224
export CK_ENV_TENSORFLOW_MODEL_MOBILENET_VERSION=1
export CK_ENV_TENSORFLOW_MODEL_MODULE=/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded/mobilenet-model.py
export CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA=YES
export CK_ENV_TENSORFLOW_MODEL_OUTPUT_LAYER_NAME=MobilenetV1/Predictions/Reshape_1
export CK_ENV_TENSORFLOW_MODEL_ROOT=/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded
export CK_ENV_TENSORFLOW_MODEL_TFLITE_FILENAME=mobilenet_v1_1.0_224_quant.tflite
export CK_ENV_TENSORFLOW_MODEL_TFLITE_FILEPATH=/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded/mobilenet_v1_1.0_224_quant.tflite
export CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILENAME=mobilenet_v1_1.0_224_quant_frozen.pb
export CK_ENV_TENSORFLOW_MODEL_TF_FROZEN_FILEPATH=/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded/mobilenet_v1_1.0_224_quant_frozen.pb
export CK_ENV_TENSORFLOW_MODEL_WEIGHTS=/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded/mobilenet_v1_1.0_224_quant.ckpt
export CK_ENV_TENSORFLOW_MODEL_WEIGHTS_ARE_CHECKPOINTS=YES
export CK_MODEL_DATA_LAYOUT=NHWC

export CK_ENV_TENSORFLOW_MODEL_SET=1
```

To inspect the model's files on disk:
```
$ ck locate env --tags=model,tflite,mlperf,mobilenet,quantized
/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded
$ ls -la `ck locate env --tags=model,tflite,mlperf,mobilenet,quantized`
total 43524
drwxr-xr-x  2 anton dvdt     4096 Mar 25 12:31 .
drwxrwxr-x 18 anton dvdt     4096 Mar 25 12:32 ..
-rw-rw-r--  1 anton dvdt     2240 Mar 25 12:31 ck-install.json
-rw-rw-r--  1 anton dvdt     3477 Mar 25 12:31 mobilenet-model.py
-rw-rw-r--  1 anton dvdt    20309 Mar 25 12:31 mobilenet_v1.py
-rw-r--r--  1 anton dvdt 17020468 Aug  3  2018 mobilenet_v1_1.0_224_quant.ckpt.data-00000-of-00001
-rw-r--r--  1 anton dvdt    14644 Aug  3  2018 mobilenet_v1_1.0_224_quant.ckpt.index
-rw-r--r--  1 anton dvdt  5143394 Aug  3  2018 mobilenet_v1_1.0_224_quant.ckpt.meta
-rw-r--r--  1 anton dvdt  4276352 Aug  3  2018 mobilenet_v1_1.0_224_quant.tflite
-rw-r--r--  1 anton dvdt   885850 Aug  3  2018 mobilenet_v1_1.0_224_quant_eval.pbtxt
-rw-r--r--  1 anton dvdt 17173742 Aug  3  2018 mobilenet_v1_1.0_224_quant_frozen.pb
-rw-r--r--  1 anton dvdt       89 Aug  3  2018 mobilenet_v1_1.0_224_quant_info.txt
```

**NB:** The TFLite weights are in the `mobilenet_v1_1.0_224*.tflite` file. Only
the TFLite weights are different between the `2018_02_22` and `2018_08_02`
MobileNets-v1 packages. We have adopted the latter for MLPerf Inference v0.5.

### **To be continued...**
