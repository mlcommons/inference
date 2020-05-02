#/bin/bash

set -euo pipefail

work_dir=my_work_dir
local_data_dir=$work_dir/local_data
librispeech_download_dir=$local_data_dir/LibriSpeech
seed=87
stage=3

install_dir=$(readlink -f third_party/install)
mkdir -p $install_dir

# stage -1: install dependencies
if [[ $stage -le -1 ]]; then
    conda create --name mlperf-rnnt python=3.6 absl-py numpy pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
    pip install -r requirements.txt

    wget https://ftp.osuosl.org/pub/xiph/releases/flac/flac-1.3.2.tar.xz -O third_party/flac-1.3.2.tar.xz
    (cd third_party; tar xf flac-1.3.2.tar.xz; cd flac-1.3.2; ./configure --prefix=$install_dir && make && make install)

    wget https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz -O third_party/sox-14.4.2.tar.gz
    (cd third_party; tar zxf sox-14.4.2.tar.gz; cd sox-14.4.2; LDFLAGS="-L${install_dir}/lib" CFLAGS="-I${install_dir}/include" ./configure --prefix=$install_dir --with-flac && make && make install)

    # TODO: install loadgen
    # TODO: install sclite
fi

export PATH="$install_dir/bin/:$PATH"

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
  mkdir -p $local_data_dir
  mkdir -p $librispeech_download_dir
  python utils/download_librispeech.py \
         utils/librispeech-inference.csv \
         $librispeech_download_dir \
         -e $local_data_dir
fi

if [[ $stage -le 2 ]]; then
  python utils/convert_librispeech.py \
      --input_dir $librispeech_download_dir/dev-clean \
      --dest_dir $local_data_dir/dev-clean-wav \
      --output_json $local_data_dir/dev-clean-wav.json
fi

if [[ $stage -le 3 ]]; then
   python inference.py \
      --model_toml configs/rnnt.toml \
      --ckpt $work_dir/rnnt.pt \
      --dataset_dir $local_data_dir \
      --val_manifest $local_data_dir/dev-clean-wav-small.json \
      --batch_size 1 \
      --seed $seed
fi
