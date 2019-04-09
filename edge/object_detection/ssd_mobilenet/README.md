[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Inference - Object Detection - SSD-MobileNet

**NB:** MLPerf Inference v0.5 uses SSD-MobileNet-v1-1.0-224 (called SSD-MobileNet in what follows).

# Table of contents

1. [Installation](#installation)
    - [Install prerequisites](#installation-debian) (Debian-specific)
    - [Install CK workflows](#installation-workflows) (universal)
1. [Benchmarking](#benchmarking)
    - [via TensorFlow (Python)](tf-py/README.md)
    - via TFLite (**coming soon**!)

<a name="installation"></a>
# Installation

<a name="installation-debian"></a>
## Debian (tested with Ubuntu v18.04 and v16.04)

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
$ ck pull repo:ck-mlperf
```
**NB:** Transitive dependencies include [repo:ck-tensorflow](https://github.com/ctuning/ck-tensorflow).

### Install the COCO 2017 validation dataset (5,000 images)
```bash
$ ck install package:dataset-coco-2017-val
```
**NB:** COCO dataset descriptions are in [repo:ck-env](https://github.com/ctuning/ck-env).

**NB:** If you have previously installed the COCO 2017 validation dataset via CK to e.g. `$HOME/coco/`, you can simply detect it as follows:
```bash
$ ck detect soft:dataset.coco.2017.val --full_path=$HOME/coco/val2017/000000000139.jpg
```
(CK also places annotations under `annotations/val2017/`.)

<a name="benchmarking"></a>
## Benchmarking

You can benchmark MobileNet using one of the available options:
- [via TensorFlow (Python)](tf-py/README.md)
- via TensorFlow Lite (**coming soon!**)
