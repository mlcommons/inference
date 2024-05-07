[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Inference - Image Classification

MLPerf Inference v0.5 uses MobileNets-v1-1.0-224 (called MobileNet in what follows) and ResNet50-v1.5 (called ResNet in what follows).

# Table of contents

1. [Installation](#installation)
    - [Install prerequisites](#installation-debian) (Debian-specific)
    - [Install CK workflows](#installation-workflows) (universal)
1. [Benchmarking](#benchmarking)
    - [via TensorFlow Lite](tflite/README.md)
    - [via TensorFlow (C++)](tf-cpp/README.md)
    - [via TensorFlow (Python)](tf-py/README.md)
    - [via ONNX](onnx/README.md)
1. [Understanding the anatomy of a benchmark](#anatomy)
1. [Inspecting experimental results](#results)

<a name="installation"></a>
# Installation

**NB:** If you would like to get a feel of CK workflows, you can skip
installation instructions and try [benchmarking](#benchmarking)
instructions on available Docker images:
- TensorFlow Lite:
    - [Debian 9](https://github.com/ctuning/ck-mlperf/tree/master/docker/image-classification-tflite.debian-9)
    - [CentOS 7](https://github.com/ctuning/ck-mlperf/tree/master/docker/image-classification-tflite.centos-7)
    - [Ubuntu 16.04](https://github.com/ctuning/ck-mlperf/tree/master/docker/image-classification-tflite.ubuntu-16.04)
    - [Ubuntu 18.04](https://github.com/ctuning/ck-mlperf/tree/master/docker/image-classification-tflite.ubuntu-18.04)
    - [Ubuntu 18.04 - Dashboard](https://github.com/ctuning/ck-mlperf/tree/master/docker/image-classification-tflite.dashboard.ubuntu-18.04)
- TensorFlow (C++):
    - [Debian 9](https://github.com/ctuning/ck-mlperf/tree/master/docker/image-classification-tf-cpp.debian-9)
- Arm NN:
    - [Debian 9](https://github.com/ARM-software/armnn-mlperf/tree/master/docker/image-classification-armnn-tflite.debian-9)

Even if you would like to run CK workflows natively (e.g. on an Arm-based
development board or Android phone), you may wish to have a quick look into the
latest Dockerfile's to check for latest updates e.g. system-specific
dependencies.

<a name="installation-debian"></a>
## Debian

- Common tools and libraries.
- [Python](https://www.python.org/), [pip](https://pypi.org/project/pip/), [NumPy](https://numpy.org/), [Collective Knowledge](https://cknowledge.org) (CK).
- (Optional) [Android SDK](https://developer.android.com/studio/), [Android NDK](https://developer.android.com/ndk/).

### Install common tools and libraries
```bash
$ sudo apt install git wget libz-dev curl
$ sudo apt install gcc g++ autoconf autogen libtool
```

### Install Python 3 and the latest pip
```bash
$ sudo apt install python3 python3-pip
$ sudo python3 -m pip install --upgrade pip
```

**NB:** Care must be taken not to mix Python 3 and Python 2 packages.
If your system uses Python 2 by default, we recommend you prefix
all CK commands, for example, with `CK_PYTHON=python3` for CK to run under Python 3:
```
$ python --version
Python 2.7.13
$ ck python_version
2.7.13 (default, Sep 26 2018, 18:42:22)
[GCC 6.3.0 20170516]
$ CK_PYTHON=python3 ck python_version
3.5.3 (default, Sep 27 2018, 17:25:39)
[GCC 6.3.0 20170516]
```
Similarly, if you use multiple Python 3 versions (e.g. 3.5 and 3.6), we recommend
you stick to one of them for consistency:
```
$ CK_PYTHON=python3.5 ck python_version
3.5.2 (default, Nov 12 2018, 13:43:14)
[GCC 5.4.0 20160609]
$ CK_PYTHON=python3.6 ck python_version
3.6.7 (default, Oct 25 2018, 09:16:13)
[GCC 5.4.0 20160609]
```

### Install required Python 3 packages
Choose one of the following installation options:
1. system-wide via pip;
1. user-space via pip;
1. user-space via CK.

With the first two options, packages get installed via pip and get registered
with CK later (typically, on the first run of a program).

With the last option, packages also get installed via pip but get registered
with CK at the same time (so there is less chance of mixing things up).

#### Option 1: system-wide installation via pip (under `/usr`)
```bash
$ sudo python3 -m pip install numpy ck
```
#### Option 2: user-space installation via pip (under `$HOME`)
```bash
$ python3 -m pip install numpy ck --user
```
#### Option 3: User-space installation via CK (under `$HOME` and `$CK_TOOLS`)
Install CK via pip (or [from GitHub](https://github.com/ctuning/ck#installation)):
```bash
$ python3 -m pip install ck --user
$ ck version
V1.10.3
```

Install and register Python packages with CK:
```bash
$ ck pull repo:ck-env
$ ck detect soft:compiler.python --full_path=`which python3`
$ ck install package --tags=lib,python-package,numpy
```

If the above dependencies have been installed on a clean system, you should be
able to inspect the registered CK environments e.g. as follows:
```
$ ck show env --tags=python-package
Env UID:         Target OS: Bits: Name:                     Version: Tags:

4e82bab01c8ee3b7   linux-64    64 Python NumPy library      1.16.2   64bits,host-os-linux-64,lib,needs-python,needs-python-3.5.2,numpy,python-package,target-os-linux-64,v1,v1.16,v1.16.2,vmaster

$ ck cat env --tags=python-package | grep PYTHONPATH
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-numpy-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
```

### [Optional] Install Android SDK and NDK

You can optionally target Android API 23 (v6.0 "Marshmallow") devices using the
`--target_os=android23-arm64` flag
(or [similar](https://source.android.com/setup/start/build-numbers)), when using
the TensorFlow Lite benchmark (recommended) and TensorFlow (C++) benchmark (not recommended).

On Debian Linux, you can install the [Android SDK](https://developer.android.com/studio/) and the [Android NDK](https://developer.android.com/ndk/) as follows:
```
$ sudo apt install android-sdk
$ adb version
Android Debug Bridge version 1.0.36
Revision 1:7.0.0+r33-2
$ sudo apt install google-android-ndk-installer
```
**NB:** On Ubuntu 18.04, NDK r13b gets installed. On Ubuntu 16.04, download [NDK r18b](https://dl.google.com/android/repository/android-ndk-r18b-linux-x86_64.zip) and extract it into e.g. `/usr/local`. NDK r18c only supports LLVM, which currently requires a CK quirk to work properly (removing a dependency on `soft:compiler.gcc.android.ndk` from `soft:compiler.llvm.android.ndk`).

<a name="installation-workflows"></a>
## Install CK workflows for MLPerf

### Pull CK repositories
```bash
$ ck pull repo:ck-mlperf
```
**NB:** Transitive dependencies include [repo:ck-tensorflow](https://github.com/ctuning/ck-tensorflow).

### Install a small dataset (500 images)
```bash
$ ck install package:imagenet-2012-val-min
```
**NB:** ImageNet dataset descriptions are in [repo:ck-env](https://github.com/ctuning/ck-env).

### Install the full dataset (50,000 images)
```bash
$ ck install package:imagenet-2012-val
```

**NB:** If you already have the ImageNet validation dataset downloaded in a directory e.g. `$HOME/ilsvrc2012-val/`, you can simply detect it as follows:
```bash
$ ck detect soft:dataset.imagenet.val --full_path=$HOME/ilsvrc2012-val/ILSVRC2012_val_00000001.JPEG
```

### Preprocess datasets

ImageNet can be preprocessed in many different ways,
which can significantly affect the resulting accuracy.
We currently support 3 different preprocessing methods:
```
$ ck install package --tags=dataset,imagenet,preprocessed

More than one package or version found:

 0) dataset-imagenet-preprocessed-using-tensorflow (fac1d0d5f4e69a85)
 1) dataset-imagenet-preprocessed-using-pillow (a6a4613ba6dfd570)
 2) dataset-imagenet-preprocessed-using-opencv (4932bbdd2ac7a17b)

Please select the package to install [ hit return for "0" ]: 
```

Preprocessing using OpenCV (option 2) is the current official method.
You can perform it directly by adding the `using-opencv` tag as follows:
```
$ ck install package --tags=dataset,imagenet,preprocessed,using-opencv --ask
```

You can locate the preprocessed files on disk using the same tags as follows:
```
$ ck cat env --tags=dataset,imagenet,preprocessed,using-opencv | grep CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR
export CK_ENV_DATASET_IMAGENET_PREPROCESSED_DIR=/datasets/dataset-imagenet-preprocessed-using-opencv
```

CK installs all the dependencies automatically. (More information on recommended choices for dependencies can be provided on demand.)

#### Summary of preprocessing methods

The table below summarizes the available methods.

| Model                | Pillow            | OpenCV universal         | OpenCV for MobileNet         | OpenCV for ResNet         | TensorFlow         |
|-|-|-|-|-|-|
| Matches official?    | No                | No                       | Yes                          | Yes                       | No                 |
| Additional tags      | `using-pillow`    | `using-opencv,universal` | `using-opencv,for-mobilenet` | `using-opencv,for-resnet` | `using-tensorflow` |
| Supported models     | ResNet, MobileNet | ResNet, MobileNet        | MobileNet only               | ResNet only               | ResNet only        |
| Supported platforms  | x86, arm          | x86                      | x86                          | x86                       | x86 (prebuilt TF)  |
| Data format          | rgb8 (int8)       | rgb8 (int8)              | rgbf32 (float32)             | rgbf32 (float32)          | rgbf32 (float32)   |
| Data size            | 7.1G              | 7.1G                     | 29G                          | 29G                       | 29G                |

The official code [preprocesses](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py) images using OpenCV. ResNet and MobileNet require different preprocessing after resizing: ResNet requires [means to be subtracted](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L178); MobileNet requires [normalization to the (-1, 1) range](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L195). In addition, the official ResNet preprocessing uses [area interpolation](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L172), instead of the default [bilinear interpolation](https://github.com/mlperf/inference/blob/master/v0.5/classification_and_detection/python/dataset.py#L154). The same behaviour can be reproduced via CK. Preprocessed images for both MobileNet and ResNet, however, require 58G on disk, as pixels are stored as 32-bit floats.

An alternative, dubbed OpenCV (universal), uses bilinear interpolation and stores pixels as 8-bit integers. At load time, however, minor additional processing may be required depending on the model (e.g. see sample [code for normalization](https://github.com/ctuning/ck-tensorflow/blob/master/program/image-classification-tflite/benchmark.h#L463) and [code for subtracting means](https://github.com/ctuning/ck-tensorflow/blob/master/program/image-classification-tflite/benchmark.h#L469)).


#### Accuracy on the ImageNet 2012 validation set

The table below shows the accuracy on the ImageNet 2012 validation set
(50,000 images) measured [via TensorFlow (C++)](tf-cpp/README.md).

| Model                   | Metric | Pillow  | OpenCV universal | OpenCV for MobileNet | OpenCV for ResNet | TensorFlow |
|-|-|-|-|-|-|-|
| ResNet                  |  Top1  | 0.76170 | 0.76422          | N/A                  | 0.76456           | 0.76522    |
|                         |  Top5  | 0.92866 | 0.93074          | N/A                  | 0.93016           | 0.93066    |
| MobileNet non-quantized |  Top1  | 0.71226 | 0.71676          | 0.71676              | N/A               | N/A        |
|                         |  Top5  | 0.89834 | 0.90118          | 0.90118              | N/A               | N/A        |
| MobileNet quantized     |  Top1  | 0.70348 | 0.70700          | 0.70694              | N/A               | N/A        |
|                         |  Top5  | 0.89376 | 0.89594          | 0.89594              | N/A               | N/A        |

Considering Top1, the universal OpenCV method is slightly less accurate for ResNet, but slightly more accurate for MobileNet quantized than the official code. The TensorFlow method is most accurate for ResNet, but is not suitable for MobileNet. The Pillow method is least accurate, but can be used on Arm platforms. The difference between the universal OpenCV and the TensorFlow methods on ResNet is exactly 0.1% or 50 images.

#### Detect datasets preprocessed on a different machine

**TODO**


<a name="benchmarking"></a>
## Benchmarking

You can benchmark MobileNet using one of the available options:
- [via TensorFlow Lite](tflite/README.md)
- [via TensorFlow (C++)](tf-cpp/README.md)
- [via TensorFlow (Python)](tf-py/README.md)
- [via ONNX](onnx/README.md)

Please come back here if you would like to learn about [the anatomy of a benchmark](#anatomy), or
how to inspect and visualize [experimental results](#results).

<a name="anatomy"></a>
## The anatomy of a benchmark

While the componentized nature of CK workflows streamlines
[installation](#installation) and [benchmarking](#benchmarking), it also makes
it less obvious what the components are and where they are stored. This section
describes the anatomy of a benchmark in terms of its components. We use the
[TFLite MobileNet implementation](tflite/README.md) as a running example.

### Model

To search for the CK entry of an installed model, use `ck search env` with the same tags you used to install it e.g.:
```
$ ck search env --tags=model,tflite,mlperf,mobilenet,quantized
local:env:3f0ca5c4d25b4ea3
```

To view more information about the CK entry, use `ck show env` e.g.:
```
$ ck show env --tags=model,tflite,mlperf,mobilenet,quantized
Env UID:         Target OS: Bits: Name:                                                                Version:                   Tags:

3f0ca5c4d25b4ea3   linux-64    64 TensorFlow model and weights (mobilenet-v1-1.0-224-quant-2018_08_02) 1_1.0_224_quant_2018_08_02 2018_08_02,64bits,downloaded,host-os-linux-64,mlperf,mobilenet,mobilenet-v1,mobilenet-v1-1.0-224,model,nhwc,python,quantised,quantized,target-os-linux-64,tensorflowmodel,tf,tflite,v1,v1.1,v1.1.0,v1.1.0.224,v1.1.0.224.0,v1.1.0.224.0.2018,v1.1.0.224.0.2018.8,v1.1.0.224.0.2018.8.2,weights
```

To view the environment variables set up by the CK entry, use `ck cat env` e.g.:
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

To inspect the model's files on disk, use `ck locate env` e.g.:
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


<a name="results"></a>
## Inspecting and visualizing experimental results

### Inspecting recorded experimental results

If you run the same command several times selecting different models (quantized
or non-quantized) or datasets (500 images or 50,000 images), CK will create
several _experimental points_ in the same repository e.g.:
```bash
$ ck find local:experiment:mlperf-mobilenet-tf-cpp-accuracy
/home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy
$ ck list_points local:experiment:mlperf-mobilenet-tf-cpp-accuracy
78dae6354e471199
918c80bc5d4906b0
```
You can then retrieve various run parameters from such experimental points.

#### Accuracy
You can quickly inspect the accuracy recorded for a particular point as follows:
```bash
$ grep \"run\": -A2 /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/ckp-918c80bc5d4906b0.0001.json
      "run": {
        "accuracy_top1": 0.718,
        "accuracy_top5": 0.9,
$ grep \"run\": -A2 /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/ckp-78dae6354e471199.0001.json
      "run": {
        "accuracy_top1": 0.704,
        "accuracy_top5": 0.898,
```

#### Model
You can quickly inspect the model used for a particular point as follows:
```bash
$ grep RUN_OPT_GRAPH_FILE /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/ckp-918c80bc5d4906b0.0001.json
      "RUN_OPT_GRAPH_FILE": "/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-downloaded/mobilenet_v1_1.0_224_frozen.pb",
$ grep RUN_OPT_GRAPH_FILE /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/ckp-78dae6354e471199.0001.json
      "RUN_OPT_GRAPH_FILE": "/home/anton/CK_TOOLS/model-tf-mlperf-mobilenet-quantized-downloaded/mobilenet_v1_1.0_224_quant_frozen.pb",
```
As expected, the lower accuracy comes from the quantized model.

#### Dataset
Unfortunately, the dataset path is recorded only to `pipeline.json`.
This file gets overwritten on each run of `ck benchmark`, so only
the dataset used in the latest command can be retrieved:
```bash
$ grep \"CK_ENV_DATASET_IMAGENET_VAL\": /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/pipeline.json
          "CK_ENV_DATASET_IMAGENET_VAL": "/home/anton/CK_TOOLS/dataset-imagenet-ilsvrc2012-val-min"
```

#### Batch count
You can, however, check the batch count e.g.:
```bash
$ grep CK_BATCH_COUNT /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/ckp-78dae6354e471199.0001.json
      "CK_BATCH_COUNT": "500",
```

#### Image cropping
By default, input images preprocessed for a client program [get cropped](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-cpp#ck_crop_percent) by 87.5%:
```bash
$ grep CK_CROP_PERCENT /home/anton/CK_REPOS/local/experiment/mlperf-mobilenet-tf-cpp-accuracy/ckp-78dae6354e471199.0001.json
      "CK_CROP_PERCENT": 87.5,
```

This can be changed by passing e.g. `--env.CK_CROP_PERCENT=100` to `ck
benchmark` (but see [here](https://github.com/mlperf/inference/tree/21efaf57d55ccd78e77b87c668bb09d47564bb6a/v0.5/classification_and_detection/optional_harness_ck/classification/tf-cpp#reference-accuracy) how this can make results worse).


### Visualizing experimental results

If you have collected some experimental results on your machine (or transferred to it results collected from elsewhere), you can visualize them as follows:
```
$ ck display dashboard --scenario=mlperf.mobilenets
```

You can also see a demo CK [MobileNets](http://cknowledge.org/dashboard/mlperf.mobilenets) dashboard online.
