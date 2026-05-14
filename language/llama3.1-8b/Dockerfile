# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


FROM nvcr.io/nvidia/pytorch:24.07-py3
SHELL ["/bin/bash", "-c"]

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN rm -rf /var/lib/apt/lists/* && rm -rf /etc/apt/sources.list.d/* \
 && apt update \
 && apt install -y --no-install-recommends build-essential autoconf \
        libtool git ccache curl wget pkg-config sudo ca-certificates \
        automake libssl-dev bc python3-dev python3-pip google-perftools \
        gdb libglib2.0-dev clang sshfs libre2-dev libboost-dev \
        libnuma-dev numactl sysstat sshpass ntpdate less iputils-ping \
 && apt -y autoremove \
 && apt remove -y cmake \
 && apt install -y --no-install-recommends pkg-config zip g++ zlib1g-dev \
        unzip libarchive-dev
RUN apt install -y --no-install-recommends rsync

# Install setuptools
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --upgrade setuptools wheel virtualenv

# Install conda
WORKDIR /tmp
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
    && bash Miniconda3-* -b -p /opt/miniconda3
ENV PATH="$PATH:/opt/miniconda3/bin"
RUN conda create -n llama3.1-8b python=30
RUN chmod -R 777 /opt/miniconda3

# Set the env variable for vLLM
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
