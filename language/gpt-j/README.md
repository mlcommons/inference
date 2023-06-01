# GPT-J Reference Implementation

### Setup Instructions

```bash
WORK_DIR=$PWD
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda activate llm
conda install mkl mkl-include -y
conda install gperftools jemalloc==5.2.1 -c conda-forge -y

# install pytorch
# you can find other nightly version in https://download.pytorch.org/whl/nightly/
pip install https://download.pytorch.org/whl/nightly/cpu-cxx11-abi/torch-2.0.0.dev20230228%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl


# installation
pip install transformers datasets evaluate accelerate simplejson nltk rouge_score

# Setup Environment Variables
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
# IOMP
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
```
### Build Loadgen
```sh
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
```

Build:

```sh
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
cd ../..
```
### Clone 
```sh
git clone https://github.com/badhri-intel/inference.git
cd inference/
git checkout gpt-j/ref_implementation
cd language/gpt-j/
```


### Download & Process Dataset
Downloads the raw data, processes and saves it as json file inside data/
```
python download_cnndm.py
```
### Calibration
Downloads CNN-Daily Mail dataset and creates the calibration dataset (JSON) for post-training quantization
```
pip install datasets
python prepare-calibration.py --calibration-list-file calibration-list.txt --output-dir </path/to/output-folder>
```
### Download GPT-J model
Downloads the model and saves it in model/ directory. If you have access to fine-tuned model, you can use that instead by naming the directory as model/.
```
python download_gptj.py
```
### Running the Benchmark
Replace the model and dataset path arguments with your corresponding paths. For evaluating the ROUGE score after the run, include --accuracy as shown below. For user specific target qps, please include user.conf.
```
python main.py --scenario=[Offline | Server | SingleStream] --model-path=./model/ --dataset-path=./data/cnn_eval.json [--accuracy] --max_examples=[Maximum number of examples to consider]
```
### Evaluate accuracy run 
Evaluates the ROGUE scores from the accuracy logs. Only applicable when specifiying [--accuracy] while running main.py
```
python evaluation.py --mlperf-accuracy-file ./build/logs/mlperf_log_accuracy.json --dataset-file ./data/cnn_eval.json
```
### License:
Apache License Version 2.0.

### Datasets & Models:

To the extent that any data, datasets or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality.  By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. DATASETS [https://github.com/badhri-intel/inference/blob/gpt-j/ref_implementation/language/gpt-j/DATASETS_MODELS.md]

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets or models.
