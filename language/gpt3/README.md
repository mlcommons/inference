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
git clone https://github.com/NVIDIA/Megatron-LM.git
cd $HOME/apex
git checkout -b language 2d8302a6c12e202f7b40b13a43daa95f326fd0ea
cd $HOME/Megatron-LM
git checkout -b language 060415572f4365a2e895f8036c4e37dad0efbdf5
```


### install requirements
```bash
pip install torch==1.13.0 torchvision==0.14.0 datasets evaluate accelerate simplejson nltk rouge_score pybind11 Ninja
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


### Build Loadgen
```sh
cd $HOME/inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py develop
```


### Download & Process Dataset
Downloads the raw data, processes and saves it as json file inside data/
```python
python download_cnndm.py
```
### Calibration
Downloads CNN-Daily Mail dataset and creates the calibration dataset (JSON) for post-training quantization
```
pip install datasets
python prepare-calibration.py --calibration-list-file calibration-list.txt --output-dir </path/to/output-folder>
```
### Download tokenizer files
Download the file [vocab.json](https://huggingface.co/gpt2/resolve/main/vocab.json) and [merges.txt](https://huggingface.co/gpt2/resolve/main/merges.txt) and place them in the `$HOME/infernece/language/gpt3/data` folder
### Download GPT-3 model
TODO: Share checkpoint link

### Running the Benchmark - Megatron
In one terminal, run the text generation server. First set the `MEGATRON_PATH` environment variable:
```bash
export MEGATRON_PATH = $HOME/Megatron-LM
```
Then run the generation server. For this 8 gpus are necessary:
```bash
cd $HOME/inference/language/gpt3/
./run_generation_server.sh
```
You can make a debug run with one gpu:
```bash
cd $HOME/inference/language/gpt3/
./run_generation_server_debug.sh
```

In another terminal run the benchmark. This will query the server each time a query for the SUT is generated
```bash
cd $HOME/inference/language/gpt3/
python main.py --scenario=[Offline | Server | SingleStream] --model-path=./model/ --dataset-path=./data/cnn_eval.json [--accuracy] --max_examples=[Maximum number of examples to consider]
```
### Evaluate accuracy run 
Evaluates the ROGUE scores from the accuracy logs. Only applicable when specifiying [--accuracy] while running main.py
```
python evaluation.py --mlperf-accuracy-file ./build/logs/mlperf_log_accuracy.json --dataset-file ./data/cnn_eval.json
```

### Reference Model - ROUGE scores
The following are the rouge scores obtained when evaluating the GPT-J fp32 model on the entire validation set (using greedy search)

ROUGE 1 - 41.5945  

ROUGE 2 - 18.4929  

ROUGE L - 28.3975  

### License:
Apache License Version 2.0.

### Datasets & Models:

To the extent that any data, datasets or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality.  By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license. DATASETS [https://github.com/badhri-intel/inference/blob/gpt-j/ref_implementation/language/gpt-j/DATASETS_MODELS.md]

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets or models.
