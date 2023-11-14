# Reference Implementation for Falcon-40B

## Prepare environment
```
conda create -n falcon-40b python=3.9
conda activate falcon-40b

# Install packages
conda install pybind11==2.10.4 -c conda-forge -y
python -m pip install torch==2.2.0.dev20231006+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip install transformers==4.31.0 nltk==3.8.1 evaluate==0.4.0 absl-py==1.4.0 rouge-score==0.1.2 sentencepiece==0.1.99 accelerate==0.21.0

export CUR_DIR=${PWD}
cd <inference-repo-root>

git submodule update --init --recursive third_party/pybind/

cd loadgen
python -m pip install .
```

## Get Dataset

## Get Model

## Run Benchmark

