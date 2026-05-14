# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

# Install LoadGen
# Cloning the LoadGen so that we have clean git repo from within the docker container.
RUN cd /tmp \
 && git clone https://github.com/mlperf/inference.git \
 && cd inference \
 && git submodule update --init third_party/pybind \
 && cd loadgen \
 && python3 setup.py install \
 && cd /tmp \
 && rm -rf inference

# Install dependencies
RUN python3 -m pip install torch==1.4.0 onnx==1.6.0 transformers==2.4.0 \ 
    onnxruntime==1.2.0 numpy==1.18.0 tokenization==1.0.7

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
