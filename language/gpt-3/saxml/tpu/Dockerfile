# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG SAXML_BUILD_SOURCE=github

ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE} AS base-build-image

ARG SAXML_VERSION_GIT_BRANCH=main
ARG SAXML_GIT_COMMIT=HEAD

ARG PYTHON_VERSION=3.8
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED TRUE

LABEL maintainer=morgandu@google.com
LABEL saxml_github_branchtag=${SAXML_VERSION_GIT_BRANCH}
LABEL saxml_github_commit=${SAXML_GIT_COMMIT}

RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
        apt-transport-https \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        gnupg \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        mlocate \
        patch \
        pkg-config \
        software-properties-common \
        sudo \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
      python3-pip \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    python${PYTHON_VERSION} -m pip install -q pip --upgrade && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 0 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 0


# Install google-cloud-cli
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >> /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get -qq update && \
    apt-get -qq install -y --no-install-recommends \
        google-cloud-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install bazel
ARG BAZEL_VERSION=5.4.0
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x bazel-*.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh


# Install numpy
RUN pip install --no-cache-dir --no-deps numpy
RUN rm -rf /root/.cache/pip


FROM base-build-image as base-build-from-github
ONBUILD ARG SAXML_GIT_COMMIT
# Download saxml sources (optionally at specific commit)
ONBUILD WORKDIR /saxml
ONBUILD RUN curl -sSL --retry 5 https://github.com/google/saxml/tarball/${SAXML_GIT_COMMIT} | tar --strip-components=1 -xzf -


FROM base-build-image as base-build-from-local
# Copy saxml local repo
ONBUILD COPY local_saxml_repo /saxml
ONBUILD WORKDIR /saxml


FROM base-build-from-${SAXML_BUILD_SOURCE} AS build-source-image
ARG SAXML_BUILD_SOURCE
RUN echo "Building SAXML from: ${SAXML_BUILD_SOURCE}"


FROM build-source-image AS build-admin-server-image

WORKDIR /saxml

RUN bazel build --color=yes --curses=yes saxml/bin:admin_config && \
    cp bazel-bin/saxml/bin/admin_config_/admin_config /usr/bin/admin_config
RUN bazel build --color=yes --curses=yes saxml/bin:admin_server && \
    cp bazel-bin/saxml/bin/admin_server_/admin_server /usr/bin/admin_server
RUN bazel build --color=yes --curses=yes saxml/bin:saxutil && \
    cp bazel-bin/saxml/bin/saxutil_/saxutil /usr/bin/saxutil

RUN mkdir -p /admin-server && \
    bazel --output_base=/tmp/bazel/output/client build \
        --color=yes \
        --curses=yes \
        --compile_one_dependency \
        saxml/client/python:sax.cc && \
    mv bazel-bin /admin-server/client

WORKDIR /
RUN git clone https://github.com/pybind/pybind11_abseil.git
WORKDIR /pybind11_abseil
RUN bazel build \
        --color=yes \
        --curses=yes \
        pybind11_abseil/status.so && \
    cp bazel-bin/pybind11_abseil/status.so /admin-server/client/saxml/client/python/

RUN bazel clean --expunge --color=yes && \
    rm -rf /root/.cache/bazel && \
    rm -rf /pybind11_abseil


FROM build-source-image AS build-model-server-image

WORKDIR /saxml

COPY patch/requirements.txt requirements.txt
COPY patch/saxml/server/pax/lm/params/BUILD saxml/server/pax/lm/params/BUILD
COPY patch/saxml/server/pax/lm/params/c4.py saxml/server/pax/lm/params/c4.py
COPY patch/saxml/server/tf/BUILD saxml/server/tf/BUILD
COPY patch/saxml/server/pax/lm/params/template.py saxml/server/pax/lm/params/template.py
COPY patch/saxml/server/pax/lm/params/lm_cloud.py saxml/server/pax/lm/params/lm_cloud.py
COPY patch/saxml/server/pax/lm/lm_tokenizer.py saxml/server/pax/lm/lm_tokenizer.py
COPY patch/saxml/server/pax/lm/servable_lm_model.py saxml/server/pax/lm/servable_lm_model.py

RUN mkdir -p /model-server && \
    bazel --output_base=/tmp/bazel/output/server build \
        --color=yes \
        --curses=yes \
        saxml/server:server && \
    mv bazel-bin /model-server/server


# RUN bazel clean --expunge --color=yes && \
#     rm -rf /root/.cache/bazel


FROM build-admin-server-image AS runtime-admin-server-image

WORKDIR /admin-server

ENV SAX_ADMIN_STORAGE_BUCKET=""

ENV PYTHONPATH=/admin-server/client/saxml/client/python

WORKDIR /
RUN git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
WORKDIR /mlperf_inference/loadgen
RUN CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
WORKDIR /mlperf_inference
RUN pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
WORKDIR /

RUN pip install --no-cache-dir --no-deps absl-py

COPY docker_entrypoint.sh /usr/bin/docker_entrypoint.sh
RUN chmod +x /usr/bin/docker_entrypoint.sh

ENTRYPOINT ["/usr/bin/docker_entrypoint.sh"]
CMD ["admin-server"]

FROM build-model-server-image AS runtime-model-server-image

WORKDIR /model-server

ENV SAX_ADMIN_STORAGE_BUCKET=""
ENV PLATFORM_CHIP=""
ENV PLATFORM_TOPOLOGY=""
ENV JAX_PLATFORMS="tpu"

ENV PATH=/model-server/server/saxml/server:${PATH}

COPY docker_entrypoint.sh /usr/bin/docker_entrypoint.sh
RUN chmod +x /usr/bin/docker_entrypoint.sh

ENTRYPOINT ["/usr/bin/docker_entrypoint.sh"]
CMD ["model-server"]
