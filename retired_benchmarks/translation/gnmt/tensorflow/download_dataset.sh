#!/bin/bash

data_dir=${DATA_DIR:-./}
mkdir -p $data_dir
pushd $data_dir
wget https://zenodo.org/record/3437893/files/gnmt_inference_data.zip
unzip gnmt_inference_data.zip
rm gnmt_inference_data.zip
popd
