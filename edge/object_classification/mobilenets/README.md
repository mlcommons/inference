[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Inference - Object Classification - MobileNets
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (Howard et al., 2017)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks ](https://arxiv.org/abs/1801.04381) (Sandler et al., 2018)

1. [Installation](#installation)
    1. [Install prerequisites](#installation-debian) (Debian-specific)
    1. [Install CK workflows](#installation-workflows) (universal)
1. [Benchmark MobileNets via TensorFlow Lite](tflite/README.md)
1. [Benchmark MobileNets via TensorFlow (C++)](tf-cpp/README.md)
1. [Benchmark MobileNets via TensorFlow (Python)](tf-py/README.md)

<a name="installation"></a>
# Installation

<a name="installation-debian"></a>
## Debian (last tested with Ubuntu v18.04)

- Common tools and libraries.
- [Python](https://www.python.org/), [pip](https://pypi.org/project/pip/), [SciPy](https://www.scipy.org/), [Collective Knowledge](https://cknowledge.org) (CK).
- (Optional) [Android SDK](https://developer.android.com/studio/), [Android NDK](https://developer.android.com/ndk/).

### Install common tools and libraries
```
$ sudo apt install autoconf autogen libtool zlib1g-dev
$ sudo apt install gcc g++ git wget
$ sudo apt install libblas-dev liblapack-dev
```

### Install Python, pip, SciPy and CK
```
$ sudo apt install python3 python3-pip
$ sudo python3 -m pip install scipy
$ sudo python3 -m pip install ck
```
**NB:** CK also supports Python 2.

### [Optional] Install Android SDK and NDK

You can optionally target Android API 23 (v6.0 "Marshmallow") devices using the
`--target_os=android23-arm64` flag 
(or [similar](https://source.android.com/setup/start/build-numbers)), when using
the TensorFlow Lite benchmark (recommended) and TensorFlow (C++) benchmark (not recommended).

On Debian Linux, you can install the [Android SDK](https://developer.android.com/studio/) and the [Android NDK](https://developer.android.com/ndk/) as follows:
```
$ sudo apt install android-sdk
$ sudo apt install google-android-ndk-installer
$ adb version
Android Debug Bridge version 1.0.36
Revision 1:7.0.0+r33-2
```

<a name="installation-workflows"></a>
## Install CK workflows

### Pull CK repositories
```
$ ck pull repo:ck-tensorflow
```

### Install a small dataset (500 images)
```
$ ck install package:imagenet-2012-val-min 
```
**NB:** ImageNet dataset descriptions are contained in [repo:ck-env](https://github.com/ctuning/ck-env).
