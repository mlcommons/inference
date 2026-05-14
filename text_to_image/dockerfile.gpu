ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.08-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

# install LDM
COPY . /diffusion
RUN cd /diffusion && \
    pip install --no-cache-dir -r requirements.txt

# install loadgen
RUN cd /tmp && \
    git clone --recursive https://github.com/mlcommons/inference && \
    cd inference/loadgen && \
    pip install pybind11 && \
    CFLAGS="-std=c++14" python setup.py install && \
    rm -rf mlperf