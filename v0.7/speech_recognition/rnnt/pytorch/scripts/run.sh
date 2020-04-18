#/bin/bash

set -euo pipefail

work_dir=work_dir
librispeech_download_dir=$work_dir/local_data/LibriSpeech5
seed=87
stage=1

mkdir -p $work_dir/local_data/

export CUDA_HOME="$(pwd)/cuda_install"

# stage -1: install dependencies
if [[ $stage -le -1 ]]; then
    # Should maybe really just do docker for non-annoying sox install
    sudo yum install sox
    conda create --name mlperf-rnnt python=3.6 absl-py numpy pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
    conda install -c conda-forge sox
    pip install -r requirements.txt
    wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
    sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit --toolkitpath="$CUDA_HOME" --librarypath="$CUDA_HOME"
    # pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" third_party/NVIDIA-apex
    pip install -v --no-cache-dir third_party/NVIDIA-apex
    # TODO: Install loadgen
    # TODO: install sclite
fi

set +u
source $HOME/ws/miniconda3/etc/profile.d/conda.sh
conda activate mlperf-rnnt
set -u

# stage 0: download model. Check checksum to skip

if [[ $stage -le 0 ]]; then
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt
fi

# stage 1: download data. Check checksum to skip?
if [[ $stage -le 1 ]]; then
  mkdir -p $librispeech_download_dir
  # python -m trace -l -C $work_dir/download_coverage 
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

exit 0

if [[ $stage -le 3 ]]; then
  # TODO: Add trace to this. Strip out unused files.
  coverage run -a inference.py \
      --model_toml configs/rnnt.toml \
      --ckpt $work_dir/rnnt.pt \
      --dataset_dir $work_dir/local_data5 \
      --val_manifest $work_dir/local_data5/dev-clean-wav-small.json \
      --batch_size 1 \
      --seed $seed
fi
