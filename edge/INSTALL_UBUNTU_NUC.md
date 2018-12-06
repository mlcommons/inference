# Ubuntu Installation on NUC

The [Intel NUC](http://a.co/d/1WnVbNk) system is the reference edge platform for MLPerf inference.
All base line numbers need to be reported on an Ubuntu 16.04 installation with GCC 5.4 installation and the following compile options

```bash
-O2 -g -Wall -Werror=format-security -Werror=implicit-function-declaration -Wl,-z,defs -Wl,-z,now -fasynchronous-unwind-tables -fexceptions -grecord-gcc-switches
```

Running in single threaded mode and with no vectorization.

### Installing Ubuntu

Installing a vanilla Ubuntu does not work with the NUC system, since it tries to use the 16GB Optane device as a disk and fails to perform the installation.
Instead, one needs to install Ubuntu using the premade [NUC images](https://www.ubuntu.com/download/iot/intel-nuc-desktop).

- Download the NUC Ubuntu [Image](http://people.canonical.com/~platform/images/nuc/pc-dawson-xenial-amd64-m4-20180507-32.iso)
- Create a bootable USB of the image using [unetbootin](https://unetbootin.github.io/), [etcher](https://www.balena.io/etcher/), or any other tool
- Boot the USB and install Ubuntu.

### Installing Docker

[Install Docker](https://docs.docker.com/engine/installation/). An easy way is using

```
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

### Using MLPerf Docker Image

MLModelScope provides pre-made Docker images that follow the MLPerf requirements

- [Base](https://hub.docker.com/r/carml/base)
- [Caffe](https://hub.docker.com/r/carml/caffe)
- [Caffe2](https://hub.docker.com/r/carml/caffe2)
- [MXNet](https://hub.docker.com/r/carml/mxnet)
- [Tensorflow](https://hub.docker.com/r/carml/tensorflow)
- [PyTorch](https://hub.docker.com/r/carml/pytorch)

> The images build the frameworks using the MLPerf compile options, but they rely on the `apt` binaries for the dependent libraries (such as OpenBLAS, libJPEG, ...). Frameworks that are distributed only as binaries such as CTNK and TensorRT are not available as MLPerf Docker images.

These images enable one to evaluate and measure MLPerf models easily without having to build everything from source.
