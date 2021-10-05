# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020, Intel Corporation, All rights reserved.
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
RUN python3 -m pip install wrapt --upgrade --ignore-installed
RUN python3 -m pip install onnx numpy==1.18.0 Pillow==7.0.0 tensorflow
RUN python3 -m pip install tensorflow-addons https://github.com/onnx/onnx-tensorflow/archive/master.zip

# Install onnxruntime
RUN python3 -m pip install onnxruntime>=1.7.0

# Install batchgenerators to be compatible with nnUnet
RUN python3 -m pip install batchgenerators<=0.21

# Install nnUnet
COPY nnUnet /workspace/nnUnet
RUN cd /workspace/nnUnet \
 && python3 -m pip install -e . \
 && cd /workspace \
 && rm -rf nnUnet

# Install OpenVINO
RUN wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020 \
 && apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020 \
 && echo "deb https://apt.repos.intel.com/openvino/2020 all main" > /etc/apt/sources.list.d/intel-openvino-2020.list \
 && apt-get update \
 && apt-get install -y intel-openvino-runtime-ubuntu18-2020.3.194

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
 && echo -e "\n%sudo ALL=(ALL:ALL) NOPASSWD:ALL\n" | tee -a /etc/sudoers \
 && echo -e "source /opt/intel/openvino/bin/setupvars.sh" | tee -a /home/${USER}/.bashrc
