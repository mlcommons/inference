FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /tmp

# Generic python installations
# PyTorch Audio for DeepSpeech: https://github.com/SeanNaren/deepspeech.pytorch/releases
# Development environment installations
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-setuptools \
  sox \
  libsox-dev \
  libsox-fmt-all \
  git \
  cmake \
  tree \
  htop \
  bmon \
  iotop \
  tmux \
  vim \
  apt-utils

# Make pip happy about itself.
# This step does not work well because after upgrade, pip3 get's lost...
#RUN pip3 install --upgrade pip

# Unlike apt-get, upgrading pip does not change which package gets installed,
# (since it checks pypi everytime regardless) so it's okay to cache pip.
# Install pytorch
# http://pytorch.org/
# Note the 3 versions of pytorch choices made available here are for convenience
# in case you need to debug. We only had success with 0.4.0
RUN pip3 install h5py \
                hickle \
                matplotlib \
                tqdm \
                torch==0.4.1 \
                torchvision \
                cffi \
			    onnx \
                python-Levenshtein \
                librosa \
                wget \
                tensorboardX

RUN apt-get update && apt-get install -yes --no-install-recommends cmake sudo

ENV CUDA_HOME "/usr/local/cuda"

# Install warp-ctc
RUN git clone https://github.com/SeanNaren/warp-ctc.git && \
    cd warp-ctc && \
    mkdir -p build && cd build && cmake .. && make && \
    cd ../pytorch_binding && python3 setup.py install

# Install pytorch audio
RUN apt-get install -y sox libsox-dev libsox-fmt-all
RUN git clone https://github.com/pytorch/audio.git
RUN cd audio; python3 setup.py install

# Install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip3 install .

ENV SHELL /bin/bash
