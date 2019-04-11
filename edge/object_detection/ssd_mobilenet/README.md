[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Inference - Object Detection - SSD-MobileNet

**NB:** MLPerf Inference v0.5 uses SSD-MobileNet-v1-1.0-224 (called SSD-MobileNet in what follows).

# Table of contents

**NB:** This README file provides installation and benchmarking instructions for the TensorFlow versions.
For the PyTorch version, please see the [corresponding README](pytorch/README.md) file.

1. [Installation](#installation)
    - [Install prerequisites](#installation-debian) (Debian-specific)
    - [Install CK workflows](#installation-workflows) (universal)
1. [Benchmarking](#benchmarking)
    - [via TensorFlow (Python)](tf-py/README.md)
    - via TensorFlow Lite (**coming soon!**)

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
**NB:** Python 3 is needed for the [COCO API](https://github.com/cocodataset/cocoapi)
used to evaluate object detection accuracy on the [COCO dataset](http://cocodataset.org).

**NB:** Care must be taken not to mix Python 3 and Python 2 packages.
If your system uses Python 2 by default, we recommend you prefix
all CK commands with `CK_PYTHON=python3`, for example:
```
$ python --version
Python 2.7.13
$ ck python_version
2.7.13 (default, Sep 26 2018, 18:42:22)
[GCC 6.3.0 20170516]
anton@hikey962:~$ CK_PYTHON=python3 ck python_version
3.5.3 (default, Sep 27 2018, 17:25:39)
[GCC 6.3.0 20170516]
```
Similarly, if you use multiple Python 3 versions (e.g. 3.5 and 3.6), we recommend
you stick to using one of them for consistency:
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
with CK later (typically, on the first run of the program).

With the last option, packages also get installed via pip but get registered
with CK at the same time (so there is less chance of mixing things up).

#### System-wide installation via pip (under `/usr`)
```bash
$ sudo python3 -m pip install cython scipy matplotlib pillow ck
```
#### User-space installation via pip (under `$HOME`)
```bash
$ python3 -m pip install cython scipy matplotlib pillow ck --user
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
$ ck install package --tags=lib,python-package,scipy
$ ck install package --tags=lib,python-package,matplotlib
$ ck install package --tags=lib,python-package,pillow
$ python3 -m pip install cython --user
```
**NB:** Cython cannot be currently installed via CK (but we are working on it).

<a name="installation-workflows"></a>
## Install CK workflows

### Pull CK repositories
```bash
$ ck pull repo:ck-mlperf
```
**NB:** Transitive dependencies include [repo:ck-tensorflow](https://github.com/ctuning/ck-tensorflow).

To update all CK repositories (e.g. after a bug fix):
```
$ ck pull repo --all
```

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

You can benchmark SSD-MobileNet using one of the available options:
- [via TensorFlow (Python)](tf-py/README.md)
- via TensorFlow Lite (**coming soon!**)
