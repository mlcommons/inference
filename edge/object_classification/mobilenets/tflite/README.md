# MobileNets via TensorFlow Lite

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
$ ck install package:lib-tflite-0.1.7-src-static [--target_os=android23-arm64]
```

You can also install TFLite from a prebuilt binary package for your target e.g.:
```
$ ck list package:lib-tflite-prebuilt*
lib-tflite-prebuilt-0.1.7-linux-aarch64
lib-tflite-prebuilt-0.1.7-linux-x64
lib-tflite-prebuilt-0.1.7-android-arm64
$ ck install package:lib-tflite-prebuilt-0.1.7-android-arm64 [--target_os=android23-arm64]
```

### Install the MobileNets model for TFLite

Install the MobileNets-v1-1.0-224 model adopted for MLPerf Inference v0.5:
```
$ ck install package --tags=tensorflowmodel,mobilenet,mlperf
```

**NB:** You can also install any other MobileNets model compatible with TFLite as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet,tflite
```

### Compile the TFLite image classification client
```
$ ck compile program:image-classification-tflite [--target_os=android23-arm64]
```

### Run the TFLite image classification client once

Run the client (if required, connect an Android device to your host machine via USB):
```
$ ck run program:image-classification-tflite [--target_os=android23-arm64]
...
*** Dependency 3 = weights (TensorFlow-Python model and weights):

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
Graph loaded in 0.001723s
All images loaded in 0.025555s
All images classified in 0.391207s
Average classification time: 0.391207s
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
--record --record_repo=local --record_uoa=mlperf-mobilenet-v1-1.00-224-tflite-0.1.7-performance \
--tags=mlperf,image-classification,mobilenet-v1-1.0-224,tflite-0.1.7,performance \
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
--record --record_repo=local --record_uoa=mlperf-mobilenet-v1-1.00-224-tflite-0.1.7-accuracy \
--tags=mlperf,image-classification,mobilenet-v1-1.0-224,tflite-0.1.7,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="anatomy"></a>
## The anatomy of the benchmark

While the componentized nature of CK workflows streamlines
[installation](#installation) and [benchmarking](#benchmarking), it also makes
it less obvious what the components are and where they are stored. This section
describes the anatomy of the benchmark in terms of its components.

### Model

To view the CK entry of the installed model:
```
$ ck search env --tags=tensorflowmodel,mobilenet,tflite
local:env:4edbb2648a48d94d
```

To view more information about the CK entry:
```
$ ck show env --tags=tensorflowmodel,mobilenet,tflite
Env UID:         Target OS: Bits: Name:                                                                 Version:             Tags:

4edbb2648a48d94d   linux-64    64 TensorFlow python model and weights (mobilenet-v1-1.0-224-2018_08_02) 1_1.0_224_2018_08_02 2018_08_02,64bits,frozen,host-os-linux-64,mlperf,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,python,target-os-linux-64,tensorflowmodel,tflite,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2,weights
```

To view the environment variables set up by the CK entry:
```
$ ck cat `ck search env --tags=tensorflowmodel,mobilenet,tflite`
#! /bin/bash
#
# --------------------[ TensorFlow python model and weights (mobilenet-v1-1.0-224-2018_08_02) ver. 1_1.0_224_2018_08_02, /home/anton/CK_REPOS/local/env/4edbb2648a48d94d/env.sh ]--------------------
# Tags: 2018_08_02,64bits,frozen,host-os-linux-64,mlperf,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,python,target-os-linux-64,tensorflowmodel,tflite,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2,weights
#
# CK generated script

if [ "$1" != "1" ]; then if [ "$CK_ENV_TENSORFLOW_MODEL_SET" == "1" ]; then return; fi; fi

# Soft UOA           = model.tensorflow.py (439b9f1757f27091)  (tensorflowmodel,weights,python,frozen,tflite,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,2018_08_02,mlperf,host-os-linux-64,target-os-linux-64,64bits,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2)
# Host OS UOA        = linux-64 (4258b5fe54828a50)
# Target OS UOA      = linux-64 (4258b5fe54828a50)
# Target OS bits     = 64
# Tool version       = 1_1.0_224_2018_08_02
# Tool split version = [1, 1, 0, 224, 2018, 8, 2]

export CK_ENV_TENSORFLOW_MODEL_IMAGE_HEIGHT=224
export CK_ENV_TENSORFLOW_MODEL_IMAGE_WIDTH=224
export CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER=1.0
export CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION=224
export CK_ENV_TENSORFLOW_MODEL_MOBILENET_VERSION=1
export CK_ENV_TENSORFLOW_MODEL_MODULE=/home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-2018_08_02-py/mobilenet-model.py
export CK_ENV_TENSORFLOW_MODEL_NORMALIZE_DATA=YES
export CK_ENV_TENSORFLOW_MODEL_WEIGHTS=/home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-2018_08_02-py/mobilenet_v1_1.0_224.ckpt
export CK_ENV_TENSORFLOW_MODEL_WEIGHTS_ARE_CHECKPOINTS=YES

export CK_ENV_TENSORFLOW_MODEL_SET=1
```

To inspect the model's files on disk:
```
$ ls -ls /home/anton/CK_TOOLS/tensorflowmodel-mobilenet-v1-1.0-224-2018_08_02-py
total 102888
drwxr-xr-x  2 anton dvdt     4096 Jan  2 17:57 .
drwxr-xr-x 51 anton dvdt    12288 Jan  2 18:14 ..
-rw-r--r--  1 anton dvdt     2028 Jan  2 17:57 ck-install.json
-rw-r--r--  1 anton dvdt     3477 Jan  2 17:57 mobilenet-model.py
-rw-r--r--  1 anton dvdt 67903136 Aug  3 01:38 mobilenet_v1_1.0_224.ckpt.data-00000-of-00001
-rw-r--r--  1 anton dvdt    19954 Aug  3 01:38 mobilenet_v1_1.0_224.ckpt.index
-rw-r--r--  1 anton dvdt  3386971 Aug  3 01:38 mobilenet_v1_1.0_224.ckpt.meta
-rw-r--r--  1 anton dvdt 17085200 Aug  3 01:38 mobilenet_v1_1.0_224_frozen.pb
-rw-r--r--  1 anton dvdt       83 Aug  3 01:38 mobilenet_v1_1.0_224_info.txt
-rw-r--r--  1 anton dvdt 16901128 Aug  3 01:39 mobilenet_v1_1.0_224.tflite
-rw-r--r--  1 anton dvdt    20309 Jan  2 17:57 mobilenet_v1.py
```

**NB:** The TFLite weights are in the `mobilenet_v1_1.0_224.tflite` file. Only
the TFLite weights are different between the `2018_02_22` and `2018_08_02`
MobileNets-v1 packages. We have adopted the latter for MLPerf Inference v0.5.

### **To be continued...**
