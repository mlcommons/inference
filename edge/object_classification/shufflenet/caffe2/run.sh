#!/bin/bash

set -ex

export MAX_JOBS=1

# install python
sudo apt-get update
sudo apt-get -y install python python-pip git-core wget
sudo apt-get -y install wget unzip

sudo pip install virtualenv

# setup virtualenv
VENV_DIR=/tmp/venv
PYTHON="$(which python)"
if [[ "${CIRCLE_JOB}" =~ py((2|3)\\.?[0-9]?\\.?[0-9]?) ]]; then
    PYTHON=$(which "python${BASH_REMATCH[1]}")
fi
$PYTHON -m virtualenv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install -U pip setuptools

# define some variables
FAI_PEP_DIR=/tmp/FAI-PEP
CONFIG_DIR=/tmp/config
LOCAL_REPORTER_DIR=/tmp/reporter
REPO_DIR=/tmp/pytorch

BENCHMARK_FILE=${FAI_PEP_DIR}/specifications/models/caffe2/shufflenet/shufflenet_accuracy_imagenet.json

SCRIPT=$(realpath "$0")
FILE_DIR=$(dirname "$SCRIPT")
IMAGENET_DIR=$1
MODEL_ROOT=${FILE_DIR}
MODEL_DIR=${MODEL_ROOT}/model/$2

mkdir -p "$CONFIG_DIR"
mkdir -p "$LOCAL_REPORTER_DIR"

if [ ! -f "${MODEL_ROOT}/shufflenet.tar.gz" ]; then
  wget -O ${MODEL_ROOT}/shufflenet.tar.gz https://s3.amazonaws.com/download.caffe2.ai/models/shufflenet/new_shufflenet/shufflenet.tar.gz
  tar -xzvf ${MODEL_ROOT}/shufflenet.tar.gz -C ${MODEL_ROOT}
fi

# clone FAI-PEP
if [ ! -d "${FAI_PEP_DIR}" ]; then
  git clone https://github.com/facebook/FAI-PEP.git "$FAI_PEP_DIR"
fi
pip install six requests

# set up default arguments
echo "
{
  \"--commit\": \"master\",
  \"--exec_dir\": \"${CONFIG_DIR}/exec\",
  \"--framework\": \"caffe2\",
  \"--local_reporter\": \"${CONFIG_DIR}/reporter\",
  \"--model_cache\": \"${CONFIG_DIR}/model_cache\",
  \"--platforms\": \"host/incremental\",
  \"--remote_repository\": \"origin\",
  \"--repo\": \"git\",
  \"--repo_dir\": \"${REPO_DIR}\",
  \"--screen_reporter\": null
}
" > ${CONFIG_DIR}/config.txt

# clone/install pytorch
pip install numpy pyyaml mkl mkl-include setuptools cmake cffi typing

if [ ! -d "${REPO_DIR}" ]; then
  git clone --recursive --quiet https://github.com/pytorch/pytorch.git "$REPO_DIR"
fi

# install ninja to speedup the build
pip install ninja

# The downloaded images should be in the following directory structure
# ${IMAGENET_DIR}/labels.txt
# ${IMAGENET_DIR}/val/n*

# install opencv for image conversion
if [ ! -d /tmp/opencv-3.4.3 ]; then
  wget -O /tmp/opencv.zip https://github.com/opencv/opencv/archive/3.4.3.zip
  unzip -q /tmp/opencv.zip -d /tmp/
  cd /tmp/opencv-3.4.3/
  mkdir build
  cd build
  cmake ..
  make -j 1
  sudo make install
fi

python ${FAI_PEP_DIR}/benchmarking/run_bench.py -b ${BENCHMARK_FILE} --string_map "{\"IMAGENET_DIR\": \"${IMAGENET_DIR}\", \"MODEL_DIR\": \"${MODEL_DIR}\"}" --config_dir "${CONFIG_DIR}"
