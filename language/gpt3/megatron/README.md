# GPT-3 Reference Implementation

## Setup Instructions

```bash
WORK_DIR=$PWD
# Create Environment (conda)
conda create -n llm python=3.9 -y
conda activate llm
```

### Download repositories
```bash
cd $HOME
git clone --recurse-submodules https://github.com/mlcommons/inference.git --depth 1
git clone https://github.com/NVIDIA/apex.git
git clone https://github.com/mlcommons/training.git --branch inference-megatron
cd $HOME/apex
git checkout -b language 2d8302a6c12e202f7b40b13a43daa95f326fd0ea
```


### install requirements
```bash
pip install torch==1.13.0 torchvision==0.14.0 datasets evaluate accelerate simplejson nltk rouge_score pybind11 Ninja numpy==1.19.5 sentencepiece  zarr tensorstore
pip install git+https://github.com/NVIDIA/mlperf-common.git
pip install git+https://github.com/mlperf/logging.git
sudo apt install pybind11-dev
```

#### install apex
For `pip >= 23.1`
```bash
cd $HOME/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
Otherwise
```bash
cd $HOME/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
**Warning:** Make sure the Nvidia driver version and the pytorch's version of cuda match

This step takes a several minutes. You can cache this step by running:
For `pip >= 23.1`
```bash
cd $HOME/apex
pip wheel -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
Otherwise
```bash
cd $HOME/apex
pip wheel -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Afterthat, you can store the whl file and simply run
```bash
pip install <wheel_output_file>.whl
```

### Build Loadgen
```sh
cd $HOME/inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py develop
```


### Download & Process Dataset
Downloads the raw data, processes and saves it as json file inside data/
```bash
cd $HOME/inference/language/gpt3/megatron
python download_cnndm.py
```
### Calibration
Downloads CNN-Daily Mail dataset and creates the calibration dataset (JSON) for post-training quantization
```bash
cd $HOME/inference/language/gpt3/megatron
pip install datasets
python prepare-calibration.py --calibration-list-file calibration-list.txt --output-dir </path/to/output-folder>
```
### Download tokenizer files
TODO: Share tokenizer links

Temporary private link:
```bash
cd $HOME/inference/language/gpt3/megatron/data/
gsutil cp gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model .
```
### Download GPT-3 model
TODO: Share checkpoint link

Temporary private link:
```bash
cd $HOME/inference/language/gpt3/megatron/
mkdir model
cd $HOME/inference/language/gpt3/megatron/model/
gcloud auth login
# gcloud storage cp "gs://mlperf-llm-public2/nv_gpt3ckpt_00011000_megatron_06162023/language_model*" .
gsutil -m rsync -r "gs://mlperf-llm-public2/nv_gpt3ckpt_00011000_megatron_06162023/" .
gsutil cp gs://mlperf-llm-public2/nv_gpt3ckpt_00011000_megatron_06162023/metadata.json .
```
### Running the Benchmark - Megatron
First set the `MEGATRON_PATH` environment variable:
```bash
export MEGATRON_PATH=$HOME/training/large_language_model/megatron-lm
```
In one terminal, run the text generation server. For this 8 gpus are necessary:
```bash
cd $HOME/inference/language/gpt3/megatron/
./run_generation_server.sh
```
You can make a debug run with one gpu:
```bash
cd $HOME/inference/language/gpt3/megatron/
./run_generation_server_debug.sh
```

In another terminal run the benchmark. This will query the server each time a query for the SUT is generated
```bash
cd $HOME/inference/language/gpt3/megatron/
python main.py --scenario=[Offline | Server | SingleStream] --model-path=./model/ --dataset-path=./data/cnn_eval.json [--accuracy] --max_examples=[Maximum number of examples to consider]
```
### Evaluate accuracy run 
Evaluates the ROGUE scores from the accuracy logs. Only applicable when specifiying [--accuracy] while running main.py
```bash
pip install rouge_score
python evaluation.py --mlperf-accuracy-file ./build/logs/mlperf_log_accuracy.json --dataset-file ./data/cnn_eval.json
```

### Reference Model - ROUGE scores
TODO: Compute rouge scores

### License:
Apache License Version 2.0.

### Datasets & Models:

To the extent that any data, datasets or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality.  By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. DATASETS [https://github.com/badhri-intel/inference/blob/gpt-j/ref_implementation/language/gpt-j/DATASETS_MODELS.md]

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets or models.
