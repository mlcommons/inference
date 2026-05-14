# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

ARG BASE_IMAGE=ubuntu
ARG BASE_TAG=24.04
FROM ${BASE_IMAGE}:${BASE_TAG} as dev-base

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    cmake \
                    libblas-dev \
                    liblapack-dev \
                    autoconf \
                    unzip \
                    wget \
                    git \
                    vim \
                    ca-certificates \
                    pkg-config \
                    build-essential \
                    numactl \
                    libnuma-dev \
                    libtcmalloc-minimal4 \
                    sudo \
                    ffmpeg \
                    sox \
                    python3.12 \
                    python3-pip \
                    python3.12-dev \
                    python-is-python3 \
                    gcc-12 g++-12 && \
                    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# pip packages
ARG PYTHON_VERSION=3.12
RUN pip install --break-system-packages torch==2.7.0 torchaudio==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --break-system-packages pandas==2.2.2 toml==0.10.2 unidecode==1.3.8 inflect==7.3.1 librosa==0.10.2 py-libnuma==1.2 numpy==2.0.1 && \
    pip install --break-system-packages sox==1.5.0 && \
    pip install --break-system-packages setuptools-scm && \
    pip install --break-system-packages transformers==4.52.4 && \
    pip install --break-system-packages -U openai-whisper

WORKDIR /workspace
COPY ./ /workspace/

# vllm
RUN mkdir -p /workspace/third_party && cd /workspace/third_party && \
    git clone https://github.com/vllm-project/vllm vllm-cpu && \
    cd vllm-cpu && \
    git checkout b124e1085b1bf977e3dac96d99ffd9d8ddfdb6cc && \
    git log -n1 && \
    pip3 install --break-system-packages -r requirements/cpu.txt && \
    VLLM_TARGET_DEVICE=cpu pip install --break-system-packages . --no-build-isolation

# mlperf loadgen
RUN mkdir -p /workspace/third_party && cd /workspace/third_party && \
    git clone https://github.com/mlcommons/inference.git mlperf-inference && cd mlperf-inference && git checkout 2814499 && cd loadgen && \
    pip install --break-system-packages . && \
    cp /workspace/third_party/mlperf-inference/mlperf.conf /workspace/
