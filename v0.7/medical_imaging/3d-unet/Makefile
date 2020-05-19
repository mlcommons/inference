# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
ifndef DOWNLOAD_DATA_DIR
	export DOWNLOAD_DATA_DIR := $(HOST_VOL)/$(BUILD_DIR)/MICCAI_BraTS_2019_Data_Training
endif
RAW_DATA_DIR := $(BUILD_DIR)/raw_data
PREPROCESSE_DATA_DIR := $(BUILD_DIR)/preprocessed_data
MODEL_DIR := $(BUILD_DIR)/model
PYTORCH_MODEL := $(MODEL_DIR)/3d-unet.pytorch
RESULT_DIR := $(BUILD_DIR)/result
MLPERF_CONF := $(BUILD_DIR)/mlperf.conf

# Env variables needed by nnUnet
export nnUNet_raw_data_base=$(RAW_DATA_DIR)
export nnUNet_preprocessed=$(PREPROCESSE_DATA_DIR)
export RESULTS_FOLDER=$(RESULT_DIR)

# Handle different nvidia-docker version
ifneq ($(wildcard /usr/bin/nvidia-docker),)
	DOCKER_RUN_CMD := nvidia-docker run
else
	DOCKER_RUN_CMD := docker run --gpus=all
endif

.PHONY: setup
setup: check_download_data_dir
	@echo "Running basic setup..."
	@if [ ! -e $(BUILD_DIR) ]; then \
		mkdir $(BUILD_DIR); \
	fi
	@if [ ! -e $(RESULT_DIR) ]; then \
		mkdir $(RESULT_DIR); \
	fi
	@if [ ! -e $(MLPERF_CONF) ]; then \
		cp ../../mlperf.conf $(MLPERF_CONF); \
	fi
	@$(MAKE) -f $(MAKEFILE_NAME) init_submodule
	@$(MAKE) -f $(MAKEFILE_NAME) download_model

.PHONY: check_download_data_dir
check_download_data_dir:
	@if [ ! -e $(DOWNLOAD_DATA_DIR) ]; then \
		echo "Please set environment variable DOWNLOAD_DATA_DIR to <path/to/MICCAI_BraTS_2019_Data_Training>" && false ; \
	fi

.PHONY: init_submodule
init_submodule:
	@echo "Initialize nnUnet submodule.."
	@git submodule update --init nnUnet

.PHONY: download_model
download_model:
	@echo "Download models..."
	@$(MAKE) -f $(MAKEFILE_NAME) download_pytorch_model

.PHONY: download_pytorch_model
download_pytorch_model:
	# @if [ ! -e $(PYTORCH_MODEL)/model.pytorch ]; then \
	# 	wget -O ; \
	# fi

.PHONY: build_docker
build_docker:
	@echo "Building docker image..."
	@docker pull nvcr.io/nvidia/pytorch:20.03-py3
	@docker build --build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
		--build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:20.03-py3 -t mlperf-inference-3d-unet -f Dockerfile .

.PHONY: launch_docker
launch_docker: check_download_data_dir
	@echo "Launching docker container..."
	@$(DOCKER_RUN_CMD) --rm -it -w $(CONTAINER_VOL) -v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/mnt/${HOME} \
		-v $(DOWNLOAD_DATA_DIR):/downloaded_data_dir \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name mlperf-inference-3d-unet-$(UNAME) -h mlperf-inference-3d-unet-$(UNAME) --add-host mlperf-inference-3d-unet-$(UNAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse --cap-add SYS_ADMIN $(DOCKER_ARGS) mlperf-inference-3d-unet

.PHONY: preprocess_data
preprocess_data:
	@echo "Restructuring raw data to $(RAW_DATA_DIR)..."
	@if [ ! -e $(RAW_DATA_DIR) ]; then \
		mkdir $(RAW_DATA_DIR); \
	fi
	@python3 Task043_BraTS_2019.py --downloaded_data_dir /downloaded_data_dir
	@echo "Preprocessing and saving preprocessed data to $(PREPROCESSE_DATA_DIR)..."
	@if [ ! -e $(PREPROCESSE_DATA_DIR) ]; then \
		mkdir $(PREPROCESSE_DATA_DIR); \
	fi
	@nnUNet_plan_and_preprocess -t 43 -pl2d None

.PHONY: run_pytorch_performance
run_pytorch_performance:
	# @python3 run.py --backend=pytorch
	@nnUNet_predict -i $(RAW_DATA_DIR)/nnUNet_raw_data/Task043_BraTS2019/imagesTr -o $(RESULT_DIR) \
		-t Task043_BraTS2019 -f 0 -tr nnUNetTrainerV2 -m 3d_fullres -p nnUNetPlansv2.mlperf.1

.PHONY: run_pytorch_accuracy
run_pytorch_accuracy:
	@echo "Not implemented yet!"
	# @python3 run.py --backend=pytorch --accuracy

.PHONY: run_onnxruntime_performance
run_onnxruntime_performance:
	@echo "Not implemented yet!"
	# @python3 run.py --backend=onnxruntime

.PHONY: run_onnxruntime_accuracy
run_onnxruntime_accuracy:
	@echo "Not implemented yet!"
	# @python3 run.py --backend=onnxruntime --accuracy

.PHONY: evaluate
evaluate:
	@echo "TODO"

.PHONY: clean
clean:
	@rm -rf build
