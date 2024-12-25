set -e
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu117
pip install liteml-24.0.0-cp310-cp310-linux_x86_64.whl
pip install --upgrade numpy

conda install pybind11==2.10.4 -c conda-forge -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia
python -m pip install transformers==4.34.0 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0


cd ../../loadgen && python3 -m pip install .
