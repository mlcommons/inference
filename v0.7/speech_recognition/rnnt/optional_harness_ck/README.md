# MLPerf Inference - Speech Recognition

Below we give an _essential_ sequence of steps that should result in a successful setup 
of the RNN-T workflow on Linux systems.

The steps are extracted from a [minimalistic Amazon Linux
2](https://github.com/ctuning/ck-mlperf/blob/master/docker/speech-recognition.rnnt/Dockerfile.amazonlinux.min)
Docker image, which is derived from a more verbose [Amazon Linux
2](https://github.com/ctuning/ck-mlperf/blob/master/docker/speech-recognition.rnnt/Dockerfile.amazonlinux)
Docker image by omitting steps that the [Collective Knowledge
framework](https://github.com/ctuning/ck) performs automatically.

For example, installing the preprocessed dataset is explicit in the verbose image:
```
#-----------------------------------------------------------------------------#
# Step 3. Download the official MLPerf Inference RNNT dataset (LibriSpeech
# dev-clean) and preprocess it to wav.
#-----------------------------------------------------------------------------#
RUN ck install package --tags=dataset,speech-recognition,dev-clean,original
# NB: Can ignore the lzma related warning.
RUN ck install package --tags=dataset,speech-recognition,dev-clean,preprocessed
#-----------------------------------------------------------------------------#
```
but is implicit in the minimalistic image:
```
#- #-----------------------------------------------------------------------------#
#- # Step 3. Download the official MLPerf Inference RNNT dataset (LibriSpeech
#- # dev-clean) and preprocess it to wav.
#- #-----------------------------------------------------------------------------#
#- RUN ck install package --tags=dataset,speech-recognition,dev-clean,original
#- # NB: Can ignore the  lzma related warning.
#- RUN ck install package --tags=dataset,speech-recognition,dev-clean,preprocessed
#- #-----------------------------------------------------------------------------#
```
because it's going to be triggered by a test performance run:
```
#+ #-----------------------------------------------------------------------------#
#+ # Step 6. Pull all the implicit dependencies commented out in Steps 1-5.
#+ #-----------------------------------------------------------------------------#
RUN ck run program:speech-recognition-pytorch-loadgen --cmd_key=performance --skip_print_timers
#+ #-----------------------------------------------------------------------------#
```
(Omitted steps are commented out with `#- `. Added steps are commented with `#+ `.)

For other possible variations and workarounds see the [complete
collection](https://github.com/ctuning/ck-mlperf/blob/master/docker/speech-recognition.rnnt/README.md)
of Docker images for this workflow including Ubuntu, Debian and CentOS.

# Table of Contents

1. [Installation](#install)
    1. Install [system-wide prerequisites](#install_system)
        1. [Ubuntu 20.04 or similar](#install_system_ubuntu)
        1. [CentOS 7 or similar](#install_system_centos_7)
        1. [CentOS 8 or similar](#install_system_centos_8)
    1. Install [Collective Knowledge](#install_ck) (CK) and its repositories
    1. Detect [GCC](#detect_gcc)
    1. Detect [Python](#detect_python)
    1. Install [Python dependencies](#install_python_deps)
    1. Install a branch of the [MLPerf Inference](#install_inference_repo) repo

<a name="install"></a>
## Installation

<a name="install_system"></a>
### Install system-wide prerequisites

**NB:** Run the below commands for your Linux system with `sudo` or as superuser.

<a name="install_system_ubuntu"></a>
#### Ubuntu 20.04 or similar
```bash
$ sudo apt update -y
$ sudo apt install -y apt-utils
$ sudo apt upgrade -y
$ sudo apt install -y\
 python3 python3-pip\
 gcc g++\
 make patch vim\
 git wget zip libz-dev\
 libsndfile1-dev
$ sudo apt clean
```

<a name="install_system_centos_7"></a>
#### CentOS 7 or similar
```bash
$ sudo yum upgrade -y
$ sudo yum install -y\
 python3 python3-pip python3-devel\
 gcc gcc-c++\
 make which patch vim\
 git wget zip unzip\
 tar xz\
 libsndfile-devel
$ sudo yum clean all
```

<a name="install_system_centos_8"></a>
#### CentOS 8 or similar
```bash
$ sudo yum upgrade -y
$ sudo yum install -y\
 gcc gcc-c++\
 make which patch vim\
 git wget zip unzip\
 openssl-devel bzip2-devel libffi-devel\
$ sudo yum clean all
$ sudo dnf install -y python3 python3-pip python3-devel
$ sudo dnf --enablerepo=PowerTools install -y libsndfile-devel
```


<a name="install_ck"></a>
### Install [Collective Knowledge](http://cknowledge.org/) (CK) and its repositories

```bash
$ export CK_PYTHON=/usr/bin/python3
$ $CK_PYTHON -m pip install --ignore-installed pip setuptools --user
$ $CK_PYTHON -m pip install ck
$ ck version
V1.15.0
$ ck pull repo:ck-mlperf
$ ck pull repo:ck-pytorch
```

<a name="detect_gcc"></a>
### Detect (system) GCC
```
$ export CK_CC=/usr/bin/gcc
$ ck detect soft:compiler.gcc --full_path=$CK_CC
$ ck show env --tags=compiler,gcc
Env UID:         Target OS: Bits: Name:          Version: Tags:

b8bd7b49f72f9794   linux-64    64 GNU C compiler 7.3.1    64bits,compiler,gcc,host-os-linux-64,lang-c,lang-cpp,target-os-linux-64,v7,v7.3,v7.3.1
```
**NB:** Required to build the FLAC and SoX dependencies of preprocessing. CK can normally detect compilers automatically, but we are playing safe here.

<a name="detect_python"></a>
### Detect (system) Python
```
$ export CK_PYTHON=/usr/bin/python3
$ ck detect soft:compiler.python --full_path=$CK_PYTHON
$ ck show env --tags=compiler,python
Env UID:         Target OS: Bits: Name:  Version: Tags:

633a6b22205eb07f   linux-64    64 python 3.7.6    64bits,compiler,host-os-linux-64,lang-python,python,target-os-linux-64,v3,v3.7,v3.7.6
```
**NB:** CK can normally detect available Python interpreters automatically, but we are playing safe here.

<a name="install_python_deps"></a>
### Install Python dependencies (in userspace)

#### Install implicit dependencies via pip
```bash
$ export CK_PYTHON=/usr/bin/python3
$ $CK_PYTHON -m pip install --user --upgrade \
  tqdm wheel toml unidecode inflect sndfile librosa numba==0.48
...
Successfully installed inflect-4.1.0 librosa-0.7.2 llvmlite-0.31.0 numba-0.48.0 sndfile-0.2.0 unidecode-1.1.1 wheel-0.34.2
```
**NB:** These dependencies are _implicit_, i.e. CK will not try to satisfy them. If they are not installed, however, the workflow will fail.


#### Install explicit dependencies via CK (also via `pip`, but register with CK at the same time)
```bash
$ ck install package --tags=python-package,torch
$ ck install package --tags=python-package,pandas
$ ck install package --tags=python-package,sox
$ ck install package --tags=python-package,absl
```
**NB:** These dependencies are _explicit_, i.e. CK will try to satisfy them automatically. On a machine with multiple versions of Python, things can get messy, so we are playing safe here.

<a name="install_inference_repo"></a>
### Install an MLPerf Inference [branch](https://github.com/dividiti/inference/tree/dvdt-rnnt) with [dividiti](http://dividiti.com)'s tweaks for RNN-T
```bash
$ ck install package --tags=mlperf,inference,source,dividiti.rnnt
```
**NB:** This source will be used for building LoadGen as well.
