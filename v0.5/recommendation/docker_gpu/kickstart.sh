#!/bin/bash
printf "Checking out code respositories\n\n"
cd ~/
mkdir ./mlperf
cd ./mlperf
git clone --progress --recurse-submodules https://github.com/mlperf/training.git
git clone --progress --recurse-submodules https://github.com/mlperf/inference.git
export DLRM_DIR=$HOME/mlperf/training/recommendation/dlrm

printf "\nDownloading tb0875_10M.pt model\n"
cd $HOME/mlperf/inference/v0.5/recommendation
mkdir ./model
curl https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt --output model/dlrm_terabyte.pytorch
export MODEL_DIR=$HOME/mlperf/inference/v0.5/recommendation/model

printf "\nCreating fake dataset\n"
cd $HOME/mlperf/inference/v0.5/recommendation/
./tools/make_fake_criteo.sh terabyte0875
export DATA_DIR=$HOME/mlperf/inference/v0.5/recommendation/fake_criteo

printf "\nCompiling loadgen\n"
cd $HOME/mlperf/inference/loadgen
CFLAGS="-std=c++14" python setup.py develop --user

cd $HOME/mlperf/inference/v0.5/recommendation
echo "Setup Complete."

export CUDA_VISIBLE_DEVICES=0
