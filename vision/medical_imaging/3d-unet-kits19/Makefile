# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 The MLPerf Authors. All Rights Reserved.
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

DOWNLOAD_DATA_DIR := ./kits19_raw_data_dir/kits19/data/
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))
UNAME := $(shell whoami)
UID := $(shell id -u `whoami`)
GROUPNAME := $(shell id -gn `whoami`)
GROUPID := $(shell id -g `whoami`)

HOST_VOL ?= ${PWD}
CONTAINER_VOL ?= /workspace

BUILD_DIR := build
DOWNLOAD_DATA_DIR := $(PWD)/kits19_raw_data_dir/kits19/data
RAW_DATA_DIR := $(BUILD_DIR)/raw_data
PREPROCESSED_DATA_DIR := $(BUILD_DIR)/preprocessed_data
POSTPROCESSED_DATA_DIR := $(BUILD_DIR)/postprocessed_data
CALIBRATION_DATA_DIR := $(BUILD_DIR)/calibration_data
MODEL_DIR := $(BUILD_DIR)/model
MLPERF_CONF := $(BUILD_DIR)/mlperf.conf
PYTORCH_MODEL := $(MODEL_DIR)/3dunet_kits19_pytorch.ptc
PYTORCH_CHECKPOINT_MODEL := $(MODEL_DIR)/3dunet_kits19_pytorch_checkpoint.pth
ONNX_MODEL := $(MODEL_DIR)/3dunet_kits19_128x128x128.onnx
ONNX_DYNAMIC_BS_MODEL := $(MODEL_DIR)/3dunet_kits19_128x128x128_dynbatch.onnx
TENSORFLOW_MODEL := $(MODEL_DIR)/3dunet_kits19_128x128x128.tf
TENSORFLOW_ZIPFILE := 3dunet_kits19_128x128x128.tf.zip
TENSORFLOW_ZIP := $(MODEL_DIR)/$(TENSORFLOW_ZIPFILE)

# ZENODO links
ZENODO_PYTORCH := https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1
ZENODO_PYTORCH_CHECKPOINT := https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch_checkpoint.pth?download=1
ZENODO_ONNX := https://zenodo.org/record/5597155/files/3dunet_kits19_128x128x128.onnx?download=1
ZENODO_ONNX_DYNBS := https://zenodo.org/record/5597155/files/3dunet_kits19_128x128x128_dynbatch.onnx?download=1
ZENODO_TENSORFLOW := https://zenodo.org/record/5597155/files/3dunet_kits19_128x128x128.tf.zip?download=1

# other special ENV
HAS_GPU := $(shell command -v nvidia-smi 2> /dev/null)
ifeq ($(HAS_GPU),)
	DOCKER_RUN_CMD := docker run
else
	# Handle different nvidia-docker version
	ifneq ($(wildcard /usr/bin/nvidia-docker),)
		DOCKER_RUN_CMD := nvidia-docker run
	else
		DOCKER_RUN_CMD := docker run --gpus=all
	endif
endif


.PHONY: setup
setup: create_directories
	@$(MAKE) -f $(MAKEFILE_NAME) download_kits19_and_duplicate
	@$(MAKE) -f $(MAKEFILE_NAME) download_models
	@if [ ! -e $(MLPERF_CONF) ]; then \
		cp ../../../mlperf.conf $(MLPERF_CONF); \
	fi
	@echo "Done basic setup (prepare directories, setup mlperf.conf, download KiTS19 dataset and models)..."

.PHONY: download_kits19_and_duplicate
download_kits19_and_duplicate:
	@$(MAKE) -f $(MAKEFILE_NAME) download_kits19_raw_data
	@$(MAKE) -f $(MAKEFILE_NAME) duplicate_kits19_case_00185

.PHONY: download_kits19_raw_data
download_kits19_raw_data:
	@if [ ! -s $(DOWNLOAD_DATA_DIR)/case_00137/imaging.nii.gz ]; then \
		echo "Cloning KITS19 repo and download RAW data into $(DOWNLOAD_DATA_DIR)..." \
		  && rm -Rf kits19_raw_data_dir \
			&& mkdir -p kits19_raw_data_dir \
			&& cd kits19_raw_data_dir \
			&& git clone https://github.com/neheller/kits19 \
			&& cd kits19 \
			&& pip3 install -r requirements.txt \
			&& python3 -m starter_code.get_imaging \
			export DOWNLOAD_DATA_DIR=$(PWD)/kits19_raw_data_dir/kits19/data; \
	else \
		echo "Valid KITS RAW data likely found in $(DOWNLOAD_DATA_DIR), skipping download..."; \
	fi

.PHONY: duplicate_kits19_case_00185
duplicate_kits19_case_00185:
	@if [ -s $(DOWNLOAD_DATA_DIR)/case_00185/imaging.nii.gz ]; then \
		echo "Duplicating KITS19 case_00185 as case_00400..." \
			&& cp -Rf $(DOWNLOAD_DATA_DIR)/case_00185 $(DOWNLOAD_DATA_DIR)/case_00400; \
	else \
		echo "KITS19 case_00185 not found! please download the dataset first..."; \
	fi

.PHONY: check_download_data_dir
check_download_data_dir:
	@if [ ! -e $(DOWNLOAD_DATA_DIR) ]; then \
		echo "KITS19 RAW data is not found; please do: make setup"; \
	fi

.PHONY: create_directories
create_directories:
	@echo "Preparing directories..."
	@if [ ! -e $(BUILD_DIR) ]; then \
		mkdir $(BUILD_DIR); \
	fi
	@if [ ! -e $(MODEL_DIR) ]; then \
		mkdir $(MODEL_DIR); \
	fi

.PHONY: download_models
download_models:
	@echo "Download models..."
	@$(MAKE) -f $(MAKEFILE_NAME) download_pytorch_model
	@$(MAKE) -f $(MAKEFILE_NAME) download_pytorch_checkpoint_model
	@$(MAKE) -f $(MAKEFILE_NAME) download_onnx_model
	@$(MAKE) -f $(MAKEFILE_NAME) download_tensorflow_model

.PHONY: download_pytorch_model
download_pytorch_model: create_directories
	@echo "Downloading PyTorch model from Zenodo..."
	@if [ ! -e $(PYTORCH_MODEL) ]; then \
		wget -O $(PYTORCH_MODEL) $(ZENODO_PYTORCH); \
	fi

.PHONY: download_pytorch_checkpoint_model
download_pytorch_checkpoint_model: create_directories
	@echo "Downloading PyTorch checkpoint model from Zenodo..."
	@if [ ! -e $(PYTORCH_CHECKPOINT_MODEL) ]; then \
		wget -O $(PYTORCH_CHECKPOINT_MODEL) $(ZENODO_PYTORCH_CHECKPOINT); \
	fi

.PHONY: download_onnx_model
download_onnx_model: create_directories
	@echo "Downloading ONNX model from Zenodo..."
	@if [ ! -e $(ONNX_MODEL) ]; then \
		wget -O $(ONNX_MODEL) $(ZENODO_ONNX); \
	fi
	@if [ ! -e $(ONNX_DYNAMIC_BS_MODEL) ]; then \
		wget -O $(ONNX_DYNAMIC_BS_MODEL) $(ZENODO_ONNX_DYNBS); \
	fi

.PHONY: download_tensorflow_model
download_tensorflow_model: create_directories
	@echo "Downloading TensorFlow model from Zenodo..."
	@if [ ! -e $(TENSORFLOW_MODEL) ]; then \
		wget -O $(TENSORFLOW_ZIP) $(ZENODO_TENSORFLOW) \
		&& cd $(MODEL_DIR) && unzip $(TENSORFLOW_ZIPFILE) && rm $(TENSORFLOW_ZIPFILE) && cd -; \
	fi

.PHONY: convert_onnx_model
convert_onnx_model: download_pytorch_model
	@echo "Converting PyTorch model to ONNX model..."
	@if [ ! -e $(ONNX_MODEL) ]; then \
		python3 unet_pytorch_to_onnx.py; \
	fi

.PHONY: convert_tensorflow_model
convert_tensorflow_model: convert_onnx_model
	@echo "Converting ONNX model to TF model..."
	@if [ ! -e $(TENSORFLOW_MODEL) ]; then \
		python3 unet_onnx_to_tensorflow.py; \
	fi

.PHONY: build_docker
build_docker:
	@echo "Building docker image..."
	@docker pull nvcr.io/nvidia/pytorch:20.12-py3
	@docker build --build-arg GID=$(GROUPID) --build-arg UID=$(UID) --build-arg GROUP=$(GROUPNAME) --build-arg USER=$(UNAME) \
	  --build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy=$(HTTPS_PROXY) --build-arg HTTPS_PROXY=$(HTTPS_PROXY) --build-arg HTTP_PROXY=$(HTTP_PROXY) \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:20.12-py3 -t mlperf-inference-3d-unet-kits19 -f Dockerfile .

# root password is root, $user password is $user
.PHONY: launch_docker
launch_docker: check_download_data_dir
	@mkdir -p $(PREPROCESSED_DATA_DIR)
	@mkdir -p $(POSTPROCESSED_DATA_DIR)
	@echo "Launching docker container..."
	@$(DOCKER_RUN_CMD) --rm -it -w $(CONTAINER_VOL) -v $(HOST_VOL):$(CONTAINER_VOL) -v ${HOME}:/mnt/${HOME} \
		-v $(DOWNLOAD_DATA_DIR):/downloaded_data_dir \
		--shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
		--security-opt apparmor=unconfined --security-opt seccomp=unconfined \
		--name mlperf-inference-3d-unet-kits19-$(UNAME) -h mlperf-inference-3d-unet-kits19-$(UNAME) --add-host mlperf-inference-3d-unet-kits19-$(UNAME):127.0.0.1 \
		--user $(UID):$(GROUPID) --net host --device /dev/fuse --cap-add SYS_ADMIN $(DOCKER_ARGS) mlperf-inference-3d-unet-kits19

.PHONY: preprocess_data
preprocess_data: create_directories
	@echo "Restructuring raw data to $(RAW_DATA_DIR)..."
	@if [ ! -e $(RAW_DATA_DIR) ]; then \
		ln -s /downloaded_data_dir $(BUILD_DIR)/raw_data; \
	fi
	@echo "Preprocessing and saving preprocessed data to $(PREPROCESSED_DATA_DIR)..."
	@if [ ! -e $(PREPROCESSED_DATA_DIR) ]; then \
		mkdir $(PREPROCESSED_DATA_DIR); \
	fi
	@python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode preprocess

.PHONY: preprocess_calibration_data
preprocess_calibration_data: create_directories
	@echo "Restructuring raw data to $(RAW_DATA_DIR)..."
	@if [ ! -e $(RAW_DATA_DIR) ]; then \
		ln -s /downloaded_data_dir $(BUILD_DIR)/raw_data; \
	fi
	@echo "Preprocessing and saving preprocessed calibration data to $(CALIBRATION_DATA_DIR)..."
	@if [ ! -e $(CALIBRATION_DATA_DIR) ]; then \
		mkdir $(CALIBRATION_DATA_DIR); \
	fi
	@python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(CALIBRATION_DATA_DIR) --mode preprocess --calibration

.PHONY: mkdir_postprocessed_data
mkdir_postprocessed_data:
	@if [ ! -e $(POSTPROCESSED_DATA_DIR) ]; then \
		mkdir $(POSTPROCESSED_DATA_DIR); \
	fi

.PHONY: run_pytorch_performance
run_pytorch_performance:
	@python3 run.py --backend=pytorch --model=$(PYTORCH_MODEL)

.PHONY: run_pytorch_accuracy
run_pytorch_accuracy: mkdir_postprocessed_data
	@python3 run.py --backend=pytorch --model=$(PYTORCH_MODEL) --accuracy

.PHONY: run_pytorch_checkpoint_performance
run_pytorch_checkpoint_performance:
	@python3 run.py --backend=pytorch_checkpoint --model=$(PYTORCH_CHECKPOINT_MODEL)

.PHONY: run_pytorch_checkpoint_accuracy
run_pytorch_checkpoint_accuracy: mkdir_postprocessed_data
	@python3 run.py --backend=pytorch_checkpoint --model=$(PYTORCH_CHECKPOINT_MODEL) --accuracy

.PHONY: run_onnxruntime_performance
run_onnxruntime_performance:
	@python3 run.py --backend=onnxruntime --model=$(ONNX_MODEL)

.PHONY: run_onnxruntime_accuracy
run_onnxruntime_accuracy: mkdir_postprocessed_data
	@python3 run.py --backend=onnxruntime --model=$(ONNX_MODEL) --accuracy

.PHONY: run_tensorflow_performance
run_tensorflow_performance:
	@python3 run.py --backend=tensorflow --model=$(TENSORFLOW_MODEL)

.PHONY: run_tensorflow_accuracy
run_tensorflow_accuracy: mkdir_postprocessed_data
	@python3 run.py --backend=tensorflow --model=$(TENSORFLOW_MODEL) --accuracy

.PHONY: evaluate
evaluate:
	@python3 accuracy_kits.py

.PHONY: clean
clean:
	@rm -rf build

.PHONY: deep_clean
deep_clean: clean
	@rm -rf __pycache__
	@rm -rf kits19_raw_data_dir
