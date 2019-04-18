[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Inference - Object Classification - MobileNet
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) (Howard et al., 2017)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks ](https://arxiv.org/abs/1801.04381) (Sandler et al., 2018)

**NB:** MLPerf Inference v0.5 uses MobileNets-v1-1.0-224 (called MobileNet in what follows).

# Table of contents

1. [Installation](#installation)
    - [Install prerequisites](#installation-debian) (Debian-specific)
    - [Install CK workflows](#installation-workflows) (universal)
1. [Benchmarking](#benchmarking)
    - [via TensorFlow Lite](tflite/README.md)
    - [via TensorFlow (C++)](tf-cpp/README.md)
    - [via TensorFlow (Python)](tf-py/README.md)
    - [via ONNX](onnx/README.md)

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
- system-wide via pip;
- user-space via pip;
- user-space via CK.

With the first two options, packages get installed via pip and get registered
with CK later (typically, on the first run of a program).

With the last option, packages also get installed via pip but get registered
with CK at the same time (so there is less chance of mixing things up).

#### System-wide installation via pip (under `/usr`)
```bash
$ sudo python3 -m pip install scipy ck
```
#### User-space installation via pip (under `$HOME`)
```bash
$ python3 -m pip install scipy ck --user
```
#### User-space installation via CK (under `$HOME` and `$CK_TOOLS`)
```bash
$ python3 -m pip install ck --user
$ ck version
V1.9.7
```
**NB:** CK can also be installed [via Git](https://github.com/ctuning/ck#ubuntu).

```bash
$ ck detect soft:compiler.python --full_path=`which python3`
$ ck install package --tags=lib,python-package,numpy
$ ck install package --tags=lib,python-package,scipy
```

If the above dependencies have been installed on a clean system, you should be
able to inspect the registered CK environments e.g. as follows:
```
$ ck show env --tags=python-package
Env UID:         Target OS: Bits: Name:                     Version: Tags:

4e82bab01c8ee3b7   linux-64    64 Python NumPy library      1.16.2   64bits,host-os-linux-64,lib,needs-python,needs-python-3.5.2,numpy,python-package,target-os-linux-64,v1,v1.16,v1.16.2,vmaster
66642698751a2fcf   linux-64    64 Python SciPy library      1.2.1    64bits,host-os-linux-64,lib,needs-python,needs-python-3.5.2,python-package,scipy,target-os-linux-64,v1,v1.2,v1.2.1,vmaster

$ ck cat env --tags=python-package | grep PYTHONPATH
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-numpy-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-scipy-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
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

<a name="benchmarking"></a>
## Benchmarking

You can benchmark MobileNet using one of the available options:
- [via TensorFlow Lite](tflite/README.md)
- [via TensorFlow (C++)](tf-cpp/README.md)
- [via TensorFlow (Python)](tf-py/README.md)
- [via ONNX](onnx/README.md)
