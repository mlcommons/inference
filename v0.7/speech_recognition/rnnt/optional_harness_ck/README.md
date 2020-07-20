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
    1. [System-wide prerequisites](#install_system)
        1. [Ubuntu 20.04 or similar](#install_system_ubuntu)
        1. [CentOS 7 or similar](#install_system_centos_7)
        1. [CentOS 8 or similar](#install_system_centos_8)
    1. [Collective Knowledge](#install_ck) (CK) and its repositories

<a name="install"></a>
## Installation

<a name="install_system"></a>
### Install system-wide prerequisites

**NB:** Run the below commands for your Linux system as superuser or with `sudo`.

<a name="install_system_ubuntu"></a>
#### Ubuntu 20.04 or similar
```bash
# DEBIAN_FRONTEND="noninteractive" apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 python3 python3-pip\
 gcc g++\
 make patch vim\
 git wget zip libz-dev\
 libsndfile1-dev\
 && apt clean
```

<a name="install_system_centos_7"></a>
#### CentOS 7 or similar
```bash
# yum upgrade -y\
 && yum install -y\
 python3 python3-pip python3-devel\
 gcc gcc-c++\
 make which patch vim\
 git wget zip unzip\
 tar xz\
 libsndfile-devel\
 && yum clean all
```

<a name="install_system_centos_8"></a>
#### CentOS 8 or similar
```bash
# yum upgrade -y\
 && yum install -y\
 gcc gcc-c++\
 make which patch vim\
 git wget zip unzip\
 openssl-devel bzip2-devel libffi-devel\
 && yum clean all
# dnf install -y python3 python3-pip python3-devel
# dnf --enablerepo=PowerTools install -y libsndfile-devel
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
