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

SHELL := /bin/bash

MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /workspace

BUILD_DIR := build
DATA_DIR := $(BUILD_DIR)/data
BERT_DIR := $(DATA_DIR)/bert_tf_v1_1_large_fp32_384_v2
RESULT_DIR := $(BUILD_DIR)/result
MLPERF_CONF := $(BUILD_DIR)/mlperf.conf
FEATURE_CACHE := eval_features.pickle

# Handle different nvidia-docker version
ifneq ($(wildcard /usr/bin/nvidia-docker),)
	DOCKER_RUN_CMD := nvidia-docker run
else
	DOCKER_RUN_CMD := docker run --gpus=all
endif

.PHONY: setup
setup:
	@if [ ! -e $(BUILD_DIR) ]; then \
		mkdir $(BUILD_DIR); \
	fi
	@if [ ! -e $(MLPERF_CONF) ]; then \
		cp ../../mlperf.conf $(MLPERF_CONF); \
	fi
	@$(MAKE) -f $(MAKEFILE_NAME) init_submodule
	@$(MAKE) -f $(MAKEFILE_NAME) download_data
	@$(MAKE) -f $(MAKEFILE_NAME) download_model

.PHONY: init_submodule
init_submodule:
	@git submodule update --init DeepLearningExamples

.PHONY: download_data
download_data:
	@if [ ! -e $(DATA_DIR) ]; then \
		mkdir $(DATA_DIR); \
	fi
	@if [ ! -e $(DATA_DIR)/dev-v1.1.json ]; then \
		wget -O $(DATA_DIR)/dev-v1.1.json https://github.com/rajpurkar/SQuAD-explorer/blob/master/dataset/dev-v1.1.json?raw=true; \
	fi
	@if [ ! -e $(DATA_DIR)/evaluate-v1.1.py ]; then \
		wget -O $(DATA_DIR)/evaluate-v1.1.py https://github.com/allenai/bi-att-flow/raw/master/squad/evaluate-v1.1.py; \
	fi
	@if [ ! -e $(BERT_DIR) ]; then \
		mkdir $(BERT_DIR) ; \
	fi
	@if [ ! -e $(RESULT_DIR) ]; then \
		mkdir $(RESULT_DIR); \
	fi

.PHONY: download_model
download_model:
	@$(MAKE) -f $(MAKEFILE_NAME) download_tf_model
	@$(MAKE) -f $(MAKEFILE_NAME) download_pytorch_model
	@$(MAKE) -f $(MAKEFILE_NAME) download_onnx_model

.PHONY: download_tf_model
download_tf_model:
	@if [ ! -e $(BERT_DIR)/model.ckpt-5474.data-00000-of-00001 ]; then \
		wget -O $(BERT_DIR)/model.ckpt-5474.data-00000-of-00001 https://zenodo.org/record/3733868/files/model.ckpt-5474.data-00000-of-00001?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/model.ckpt-5474.index ]; then \
		wget -O $(BERT_DIR)/model.ckpt-5474.index https://zenodo.org/record/3733868/files/model.ckpt-5474.index?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/model.ckpt-5474.meta ]; then \
		wget -O $(BERT_DIR)/model.ckpt-5474.meta https://zenodo.org/record/3733868/files/model.ckpt-5474.meta?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/vocab.txt ]; then \
		wget -O $(BERT_DIR)/vocab.txt https://zenodo.org/record/3733868/files/vocab.txt?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/model.pb ]; then \
		wget -O $(BERT_DIR)/model.pb https://zenodo.org/record/3939747/files/model.pb?download=1; \
	fi

.PHONY: download_pytorch_model
download_pytorch_model:
	@if [ ! -e $(BERT_DIR)/model.pytorch ]; then \
		wget -O $(BERT_DIR)/model.pytorch https://zenodo.org/record/3733896/files/model.pytorch?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/vocab.txt ]; then \
		wget -O $(BERT_DIR)/vocab.txt https://zenodo.org/record/3733896/files/vocab.txt?download=1; \
	fi

.PHONY: download_onnx_model
download_onnx_model:
	@if [ ! -e $(BERT_DIR)/model.onnx ]; then \
		wget -O $(BERT_DIR)/model.onnx https://zenodo.org/record/3733910/files/model.onnx?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/bert_large_v1_1_fake_quant.onnx ]; then \
		wget -O $(BERT_DIR)/bert_large_v1_1_fake_quant.onnx https://zenodo.org/record/3750364/files/bert_large_v1_1_fake_quant.onnx?download=1; \
	fi
	@if [ ! -e $(BERT_DIR)/vocab.txt ]; then \
		wget -O $(BERT_DIR)/vocab.txt https://zenodo.org/record/3733910/files/vocab.txt?download=1; \
	fi

.PHONY: build_docker
build_docker:
	@docker pull nvcr.io/nvidia/tensorrtserver:19.08-py3
	@cd DeepLearningExamples/TensorFlow/LanguageModeling/BERT && docker build . --rm -t mlperf-inference-bert
	@docker build --build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		--build-arg BASE_IMAGE=mlperf-inference-bert -t mlperf-inference-bert - < Dockerfile

.PHONY: launch_docker
launch_docker:
	@$(DOCKER_RUN_CMD) --rm -it -w /workspace -v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/mnt/${HOME} \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name mlperf-inference-bert-$(UNAME) -h mlperf-inference-bert-$(UNAME) --add-host mlperf-inference-bert-$(UNAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse --cap-add SYS_ADMIN $(DOCKER_ARGS) mlperf-inference-bert

.PHONY: run_tf_performance
run_tf_performance:
	@python3 run.py --backend=tf

.PHONY: run_tf_accuracy
run_tf_accuracy:
	@python3 run.py --backend=tf --accuracy

.PHONY: run_pytorch_performance
run_pytorch_performance:
	@python3 run.py --backend=pytorch

.PHONY: run_pytorch_accuracy
run_pytorch_accuracy:
	@python3 run.py --backend=pytorch --accuracy

.PHONY: run_onnxruntime_performance
run_onnxruntime_performance:
	@python3 run.py --backend=onnxruntime

.PHONY: run_onnxruntime_accuracy
run_onnxruntime_accuracy:
	@python3 run.py --backend=onnxruntime --accuracy

.PHONY: run_onnxruntime_quantized_performance
run_onnxruntime_quantized_performance:
	@python3 run.py --backend=onnxruntime --quantized

.PHONY: run_onnxruntime_quantized_accuracy
run_onnxruntime_quantized_accuracy:
	@python3 run.py --backend=onnxruntime --quantized --accuracy

.PHONY: evaluate
evaluate:
	@python3 $(DATA_DIR)/evaluate-v1.1.py.json $(DATA_DIR)/dev-v1.1.json $(RESULT_DIR)/predictions.json

.PHONY: clean
clean:
	@rm -rf ${BUILD_DIR}
	@rm -f  ${FEATURE_CACHE}
	@rm -f  onnxruntime_profile__*.json
