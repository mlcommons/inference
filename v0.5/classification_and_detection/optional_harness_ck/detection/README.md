[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# MLPerf Inference - Object Detection

MLPerf Inference v0.5 uses SSD-MobileNet-v1-1.0-224 (called SSD-MobileNet in what follows) and SSD-ResNet34 (called SSD-ResNet in what follows).

# Table of contents

1. [Installation](#installation)
    - [Install prerequisites](#installation-debian) (Debian-specific)
    - [Install CK workflows](#installation-workflows) (universal)
1. [Benchmarking](#benchmarking)
    - [via TensorFlow (Python)](tf-py/README.md)
    - [via TensorFlow Lite](tflite/README.md)

<a name="installation"></a>
# Installation

**NB:** If you would like to get a feel of CK workflows, you can skip
installation instructions and try [benchmarking](#benchmarking)
instructions on available Docker images:
- TensorFlow Lite:
    - [Debian 9](https://github.com/ctuning/ck-mlperf/tree/master/docker/object-detection-tflite.debian-9)
- Arm NN:
    - [Debian 9](https://github.com/ARM-software/armnn-mlperf/tree/master/docker/object-detection-armnn-tflite.debian-9)

Even if you would like to run CK workflows natively (e.g. on an Arm-based
development board or Android phone), you may wish to have a quick look into the
latest Dockerfile's to check for latest updates e.g. system-specific
dependencies.

<a name="installation-debian"></a>
## Debian

- Common tools and libraries.
- [Python](https://www.python.org/), [pip](https://pypi.org/project/pip/), [SciPy](https://www.scipy.org/), [Collective Knowledge](https://cknowledge.org) (CK).
- (Optional) [Android SDK](https://developer.android.com/studio/), [Android NDK](https://developer.android.com/ndk/).

### Install common tools and libraries
```bash
$ sudo apt install git wget libz-dev curl cmake
$ sudo apt install gcc g++ autoconf autogen libtool
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
$ sudo python3 -m pip install cython scipy==1.2.1 matplotlib pillow ck
```
#### Option 2: user-space installation via pip (under `$HOME`)
```bash
$ python3 -m pip install cython scipy==1.2.1 matplotlib pillow ck --user
```
#### Option 3: user-space installation via CK (under `$HOME` and `$CK_TOOLS`)
Install CK via pip (or [from GitHub](https://github.com/ctuning/ck#installation)):
```bash
$ python3 -m pip install ck --user
$ ck version
V1.9.7
```
Install and register Python packages with CK:
```bash
$ ck pull repo:ck-env
$ ck detect soft:compiler.python --full_path=`which python3`
$ ck install package --tags=lib,python-package,numpy
$ ck install package --tags=lib,python-package,scipy --force_version=1.2.1
$ ck install package --tags=lib,python-package,matplotlib
$ ck install package --tags=lib,python-package,pillow
$ ck install package --tags=lib,python-package,cython
```

If the above dependencies have been installed on a clean system, you should be
able to inspect the registered CK environments e.g. as follows:
```
$ ck show env --tags=python-package
Env UID:         Target OS: Bits: Name:                     Version: Tags:

4e82bab01c8ee3b7   linux-64    64 Python NumPy library      1.16.2   64bits,host-os-linux-64,lib,needs-python,needs-python-3.5.2,numpy,python-package,target-os-linux-64,v1,v1.16,v1.16.2,vmaster
66642698751a2fcf   linux-64    64 Python SciPy library      1.2.1    64bits,host-os-linux-64,lib,needs-python,needs-python-3.5.2,python-package,scipy,target-os-linux-64,v1,v1.2,v1.2.1,vmaster
78e8a1bfb4eb052c   linux-64    64 Python Matplotlib library 3.0.3    64bits,host-os-linux-64,lib,matplotlib,needs-python,needs-python-3.5.2,python-package,target-os-linux-64,v3,v3.0,v3.0.3,vmaster
a6f9c25377710f6f   linux-64    64 Python Pillow library     6.0.0    64bits,PIL,host-os-linux-64,lib,needs-python,needs-python-3.5.2,pillow,python-package,target-os-linux-64,v6,v6.0,v6.0.0,vmaster
498dbe464d051b44   linux-64    64 Python Cython library     0.29.9   64bits,cython,host-os-linux-64,lib,needs-python,needs-python-3.5.2,python-package,target-os-linux-64,v0,v0.29,v0.29.9,vmaster

$ ck cat env --tags=python-package | grep PYTHONPATH
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-numpy-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-scipy-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-matplotlib-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-pillow-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
export PYTHONPATH=/home/anton/CK_TOOLS/lib-python-cython-compiler.python-3.5.2-linux-64/build:${PYTHONPATH}
```

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
$ ck install package --tags=object-detection,dataset,coco.2017,val,original,full
```
**NB:** COCO dataset descriptions are in [repo:ck-env](https://github.com/ctuning/ck-env).

**NB:** If you have previously installed the COCO 2017 validation dataset via CK to e.g. `$HOME/coco/`, you can simply detect it as follows:
```bash
$ ck detect soft:dataset.coco.2017.val --full_path=$HOME/coco/val2017/000000000139.jpg
```
(CK also places annotations under `annotations/val2017/`.)

### Preprocess the COCO 2017 validation dataset (first 50 images)
```bash
$ ck install package --tags=object-detection,dataset,coco.2017,preprocessed,first.50
```

### Preprocess the COCO 2017 validation dataset (all 5,000 images)
```bash
$ ck install package --tags=object-detection,dataset,coco.2017,preprocessed,full
```

<a name="benchmarking"></a>
## Benchmarking

You can benchmark SSD-MobileNet using one of the available options:
- [via TensorFlow (Python)](tf-py/README.md)
- [via TensorFlow Lite](tflite/README.md)
