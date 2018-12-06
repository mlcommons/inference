[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Edge Inference - Object Detection - SSD-MobileNet-v1

1. [Installation](#installation)
    1. [Install prerequisites](#installation-debian) (Debian-specific)
    1. [Install CK workflows](#installation-workflows) (universal)
1. [Benchmark SSD-MobileNet-v1 via TensorFlow (Python)](tf-py/README.md)

<a name="installation"></a>
# Installation

<a name="installation-debian"></a>
## Debian (last tested with Ubuntu v18.04)

- Common tools and libraries.
- [Python](https://www.python.org/), [pip](https://pypi.org/project/pip/), [SciPy](https://www.scipy.org/), [Collective Knowledge](https://cknowledge.org) (CK).
- (Optional) [Android SDK](https://developer.android.com/studio/), [Android NDK](https://developer.android.com/ndk/).

### Install common tools and libraries
```bash
$ sudo apt install autoconf autogen libtool zlib1g-dev
$ sudo apt install gcc g++ git wget
$ sudo apt install libblas-dev liblapack-dev
```

### Install Python, pip, SciPy and CK
```bash
$ sudo apt install python3 python3-pip python3-tk
$ sudo python3 -m pip install cython
$ sudo python3 -m pip install scipy
$ sudo python3 -m pip install ck
```
**NB:** CK also supports Python 2.

<a name="installation-workflows"></a>
## Install CK workflows

### Pull CK repositories
```bash
$ ck pull repo:ck-tensorflow
```

### Install the COCO dataset
```bash
$ ck install package:dataset-coco-2014
```

### Install the SSD-MobileNet-v1 model
```bash
$ ck install package:tensorflowmodel-object-detection-ssd-mobilenet-v1-coco
$ ck install package:tensorflowmodel-api
```
