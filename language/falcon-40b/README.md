# Reference Implementation for Falcon-40B
**Basic implementation for Falcon-40B. This is still WIP. It doesn't yet have functional support for First token and remainder output tokens generations. Features will be added over the next couple of weeks**

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
TODO


## Get Model
+ Requires Git Large Files Storage
```
export CHECKPOINT_PATH=${PWD}/Falcon-40B
git lfs install
git clone https://huggingface.co/tiiuae/falcon-40b-instruct ${CHECKPOINT_PATH}
```

## Run Benchmark

### Offline
```
python -u main.py --scenario Offline \
                --model-path ${CHECKPOINT_PATH} \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 1024 \
                --device cpu \
                --output-log-dir offline-logs

```

### Server
```
python -u main.py --scenario Server \
                --model-path ${CHECKPOINT_PATH} \
                --mlperf-conf mlperf.conf \
                --user-conf user.conf \
                --total-sample-count 1024 \
                --device cpu \
                --output-log-dir server-logs
```
