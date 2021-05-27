# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, Intel Corporation, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim ripgrep tree

RUN python3 -m pip install --upgrade pip

# Install dependencies
RUN python3 -m pip install wrapt --upgrade --ignore-installed
RUN python3 -m pip install onnx==1.9.0 Pillow==8.2.0 tensorflow==2.4.1 numpy>=1.19.2
RUN python3 -m pip install tensorflow-addons https://github.com/onnx/onnx-tensorflow/archive/refs/heads/rel-1.8.0.zip
RUN python3 -m pip install git+https://github.com/NVIDIA/dllogger
RUN python3 -m pip install nibabel==3.2.1 scipy==1.6.3
RUN python3 -m pip install https://github.com/mlcommons/logging/archive/refs/tags/0.7.1.zip
RUN python3 -m pip install test-generator==0.1.1
RUN python3 -m pip install pdbpp

# Install onnxruntime
# GPU release build
# RUN python3 -m pip install onnxruntime-gpu
# CPU release build
RUN python3 -m pip install onnxruntime

# Install LoadGen
# Cloning the LoadGen so that we have clean git repo from within the docker container.
RUN cd /tmp \
 && git clone https://github.com/mlcommons/inference.git \
 && cd inference \
 && git submodule update --init third_party/pybind \
 && cd loadgen \
 && python3 setup.py install \
 && cd /tmp \
 && rm -rf inference

# TF 2.4.1 libcusolver touchup 
RUN cd /usr/local/cuda-11.1/targets/x86_64-linux/lib/ \
 && ln -s libcusolver.so.11 libcusolver.so.10 \
 && cd -
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Add user
ARG GID
ARG UID
ARG GROUP
ARG USER
RUN echo root:root | chpasswd \
 && groupadd -f -g ${GID} ${GROUP} \
 && useradd -G sudo -g ${GID} -u ${UID} -m ${USER} \
 && echo ${USER}:${USER} | chpasswd \
 && echo -e "\nexport PS1=\"(mlperf) \\u@\\h:\\w\\$ \"" | tee -a /home/${USER}/.bashrc \
 && echo -e "\n%sudo ALL=(ALL:ALL) NOPASSWD:ALL\n" | tee -a /etc/sudoers
