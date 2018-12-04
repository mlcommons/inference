FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /tmp

# Generic python installations
# PyTorch Audio for DeepSpeech: https://github.com/SeanNaren/deepspeech.pytorch/releases
# Development environment installations
RUN apt-get update && apt-get install -y \
  python \
  python-pip \
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
RUN pip install --upgrade pip

# Unlike apt-get, upgrading pip does not change which package gets installed,
# (since it checks pypi everytime regardless) so it's okay to cache pip.
# Install pytorch
# http://pytorch.org/
# Note the 3 versions of pytorch choices made available here are for convenience
# in case you need to debug. We only had success with 0.4.0
RUN pip install h5py \
                hickle \
                matplotlib \
                tqdm \
                http://download.pytorch.org/whl/cu80/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl \
                torchvision \
                cffi \
                python-Levenshtein \
                librosa \
                wget \
                tensorboardX

RUN apt-get update && apt-get install --yes --no-install-recommends cmake sudo

ENV CUDA_HOME "/usr/local/cuda"

# Install warp-ctc
# This wrap-ctc is changed to specifically work with pytorch 0.4.0
# Specifically the changes are in "warp-ctc/pytorch_binding/src/binding.cpp"
RUN git clone https://github.com/ahsueh1996/warp-ctc.git && \
    cd warp-ctc && \
    mkdir -p build && cd build && cmake .. && make VERBOSE=1 && \
    cd ../pytorch_binding && python setup.py install

# Install pytorch audio
# Two options are available: the newest torchaudio and the specific commit
# This solves https://github.com/mlperf/training/issues/41
# As a consequence, we need to... "fix" torchaudio later
RUN apt-get install -y sox libsox-dev libsox-fmt-all
RUN git clone https://github.com/pytorch/audio.git
RUN cd audio; python setup.py install					# This option pulls the newest version
# RUN cd audio; git reset --hard 67564173db19035329f21caa7d2be986c4c23797; python setup.py install # This option to solve dependency issue

# Install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip install .

ENV SHELL /bin/bash
