#/bin/bash

set -euo pipefail

work_dir=work_dir
local_data_dir=$work_dir/local_data
librispeech_download_dir=$local_data_dir/LibriSpeech5
seed=87
stage=3

mkdir -p $work_dir/local_data/

# stage -1: install dependencies
if [[ $stage -le -1 ]]; then
    sudo yum install sox
    conda create --name mlperf-rnnt python=3.6 absl-py numpy pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
    conda install -c conda-forge sox
    pip install -r requirements.txt
    # TODO: install loadgen
    # TODO: install sclite
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlperf-rnnt
set -u

# stage 0: download model. Check checksum to skip?
if [[ $stage -le 0 ]]; then
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt
fi

# stage 1: download data. Check checksum to skip?
if [[ $stage -le 1 ]]; then
  mkdir -p $librispeech_download_dir
  coverage run -a utils/download_librispeech.py \
         utils/librispeech.csv \
         $librispeech_download_dir \
         -e $work_dir/local_data5
fi

if [[ $stage -le 2 ]]; then
  coverage run -a utils/convert_librispeech.py \
      --input_dir $work_dir/local_data5/LibriSpeech/dev-clean \
      --dest_dir $work_dir/local_data5/dev-clean-wav \
      --output_json $work_dir/local_data5/dev-clean-wav.json
fi

if [[ $stage -le 3 ]]; then
   ipython --pdb -c "%run inference.py \
      --model_toml configs/rnnt.toml \
      --ckpt $work_dir/rnnt.pt \
      --dataset_dir $work_dir/local_data5 \
      --val_manifest $work_dir/local_data5/dev-clean-wav.json \
      --batch_size 2 \
      --seed $seed"
fi
