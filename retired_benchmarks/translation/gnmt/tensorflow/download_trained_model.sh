#!/bin/bash

model_dir=${MODEL_DIR:-./}
mkdir -p $model_dir
pushd $model_dir
wget https://zenodo.org/record/2530924/files/gnmt_model.zip
unzip gnmt_model.zip
rm -f gnmt_model.zip
popd
