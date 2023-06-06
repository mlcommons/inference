# MLCommons (MLPerf) Inference Benchmarks for Recommendation Task

This is the reference implementation for MLCommons Inference benchmarks.

### Supported Models

**TODO: Decide benchmark name**
| name | framework | acc. | AUC | dataset | weights  | size | prec. | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| dlrm_dcn (debugging) | PyTorch | N/A | N/A | Fake multihot criteo dataset generated with [make_fake_criteo.sh](tools/make_fake_criteo.sh)       | N/A                                                                     | ~1GB | fp32 |                          |
| dlrm_dcn (debugging) | PyTorch | 81.07% | N/A | [Multihot Criteo Sample](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/) | [pytorch](#downloading-model-weights) | ~100GB | fp32 |  |
| dlrm_dcn (official) | PyTorch | N/A | 80.31% | [Multihot Criteo](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/) | [pytorch](#downloading-model-weights) | ~100GB | fp32 |  |

### Disclaimer
This is a reference implementation of the benchmark that is not meant to be the fastest implementation possible.
The official model is the largest model on the order of 100GB, while interface to other models is only provided to facilitate debugging.

The reference implementation includes all required pre-processing of datasets.
It also includes ```--accuracy``` option to validate accuracy and ROC AUC (receiver operating characteritics area under the curve) metrics as required by MLPerf [1].

The reference implementation was tested on a machine with 256GB RAM and 8x32GB GPUs.

[1] [T. Fawcett, An introduction to ROC analysis, Pattern Recognition Letters, 2006](https://dl.acm.org/doi/10.1016/j.patrec.2005.10.010)

## Prerequisites and Installation
We support [PyTorch](http://pytorch.org) and might add TensorFlow backend implementation.

The following steps are **only** needed if you run the benchmark **without Docker**.

Python 3.5, 3.6 or 3.7 is supported and we recommend to use Anaconda.

Install the desired backend. For pytorch:
```
pip install torch torchvision torchrec torchsnapshot
pip install scikit-learn
pip install numpy
pip install pydot
pip install torchviz
pip install protobuf
pip install tqdm
```

### Prepare the code and dataset
1. Download or clone the MLCommons [inference](https://github.com/mlcommons/inference) code
```
cd $HOME
mkdir ./mlcommons && cd ./mlcommons
git clone --recurse-submodules https://github.com/mlcommons/training.git
```
2. Download pre-trained model weights (see links available above)
```
cd $HOME/mlcommons/inference/recommendation/dlrm_v2/pytorch/
mkdir ./model && cd ./model
mv <downloaded_file> dlrm_terabyte.pytorch
export MODEL_DIR=./model
```
3. Download corresponding Criteo dataset (see links available above)
```
cd $HOME/mlcommons/inference/recommendation/dlrm_v2/pytorch/
mkdir ./dataset && cd ./dataset
mv <downloaded_file(s)> ./
export DATA_DIR=./dataset
```
4. Build and install the loadgen
```
cd $HOME/mlcommons/inference/loadgen
CFLAGS="-std=c++14" python setup.py develop --user
```

### Downloading model weights

File name | framework | Size in bytes (`du *`) | MD5 hash (`md5sum *`)
-|-|-|-
N/A | pytorch | <2GB | -
[weight_sharded](https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download) | pytorch | 97.31GB | -

You can download the weights by running:
```
wget https://cloud.mlcommons.org/index.php/s/XzfSeLgW8FYfR3S/download -O weigths.zip
unzip weights.zip
```
(optional) To speed future downloads, we recommend you to save the weights in a bucket (E.g GCP, AWS). For example, after saving the checkpoint in a GCP bucket, you can download the weights faster by running:
```
export BUCKET_NAME=<BUCKET_CONTAINING_MODEL>
cd $HOME/mlcommons/inference/recommendation/dlrm_v2/pytorch/model/
gsutil -m cp -r "gs://$BUCKET_NAME/model_weights/*" .
```

### Downloading dataset
| Original dataset | download link |
| ---- | ---- |
| Criteo Terabyte (day 23) | https://labs.criteo.com/2013/12/download-terabyte-click-logs/ |


1. The Criteo fake dataset can be created in place of the real datasets in order to facilitate debugging and testing. We provide a fake (random) data generator that can be used to quickly generate data samples in a format compatible with the original dataset. Please use the following script in `./tools` to quickly create random samples for the corresponding models, which will be placed into `./fake_criteo` directory
```
./make_fake_criteo.sh
mv ./fake_criteo .. && cd ..
export DATA_DIR=./fake_criteo
```


2. The Multihot Criteo dataset is stored in several files corresponding to 24 days: `day_0.gz`, `day_1.gz`, ..., `day_23.gz` (~343GB). For this benchmark, we only use the validation dataset, which corresponds to first half of `day_23.gz`.
    - The dataset can be constructed from the criteo terabyte dataset. You can find the instructions for constructing the dataset [here](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset)


### Calibration set

For MLPerf Inference, we use the first 128000 rows (user-item pairs) of the second half of `day_23` as the calibration set. Specifically, `day_23` contains 178274637 rows in total, so we use the rows **from the 89137319-th row to the 89265318-th row (both inclusive) in `day_23`** as the calibration set (assuming 0-based indexing).

## Running the benchmark

Download and install all the pre-requisites. Both local and docker environment need to set 3 environment variables:
```
export WORLD_SIZE=<number_of_nodes>
export DATA_DIR=YourCriteoMultihotLocation
export MODEL_DIR=YourModelFileLocation
```
For running the benchmark in cpu, we suggest to run `WORLD_SIZE=1'

### Run local
```
./run_local.sh backend model dataset device [options]

backend is one of [pytorch]
model is one of [dlrm]
dataset is one of [debug|multihot-criteo-sample|multihot-criteo]
device is one of [cpu|gpu]
options are extra arguments that are passed along
```

For example, to run on CPU you may choose to use:

1. Fake Multihot Criteo Dataset (debugging)

Offline scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm debug cpu --scenario Offline --samples-to-aggregate-fix=2048 --max-batchsize=2048
./run_local.sh pytorch dlrm debug cpu --scenario Offline --samples-to-aggregate-fix=2048 --max-batchsize=2048 --samples-per-query-offline=1 --accuracy
```
Server scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm debug cpu --scenario Server --samples-to-aggregate-fix=2048 --max-batchsize=2048
./run_local.sh pytorch dlrm debug cpu --scenario Server --samples-to-aggregate-fix=2048 --max-batchsize=2048 --accuracy
```

2. Multihot Criteo Sample Dataset (debugging)

Offline scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm multihot-criteo-sample cpu --scenario Offline --max-ind-range=10000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048
./run_local.sh pytorch dlrm multihot-criteo-sample cpu --scenario Offline --max-ind-range=10000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=1 --accuracy
```
Server scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm multihot-criteo-sample cpu --scenario Server  --max-ind-range=10000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048
./run_local.sh pytorch dlrm multihot-criteo-sample cpu --scenario Server  --max-ind-range=10000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --accuracy
```

3. Multihot Criteo Dataset (official)

Offline scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm multihot-criteo cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800
./run_local.sh pytorch dlrm multihot-criteo cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800 --accuracy
```
Server scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm multihot-criteo cpu --scenario Server  --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048
./run_local.sh pytorch dlrm multihot-criteo cpu --scenario Server  --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --accuracy
```


Note that this script will pre-process the data during the first run and reuse it over sub-sequent runs. The pre-processing of data can take a significant amount of time during the first run.

In order to use GPU(s), you might need to select the number of GPUs with the environment variable `CUDA_VISIBLE_DEVICES`, and run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./run_local.sh pytorch dlrm multihot-criteo gpu --accuracy
```

### Get started quickly with Docker

Ensure you have a working docker setup on your machine.

#### CPU

Build Dockerfile configuration using the script provided
```
cd $HOME/mlcommons/inference/recommendation/dlrm/pytorch/docker_cpu
./build_docker_cpu.sh

```
The container will have loadgen binary and all other tools needed to run the experiments. DLRM code, Inference code, 
Model, and Data are located on the host machine and can be shared between multiple containers

Edit run_docker.sh to set directories, the defaults are:
HOST_MLCOMMONS_ROOT_DIR=$HOME/mlcommons/inference	# path to mlcommons/inference		
MODEL_DIR=$HOME/mlcommons/model-multihot-criteo		# path to model folder
DATA_DIR=$HOME/mlcommons/data-multihot-criteo			# path to data folder

Run Docker container in interactive mode and enter the docker console
```
cd $HOME/mlcommons/inference/recommendation/dlrm/pytorch/docker_cpu
./run_docker_cpu.sh
```

Example of running multihot-criteo test on CPU in docker console:
```
cd mlcommons/recommendation/dlrm/pytorch
./run_local.sh multihot-criteo cpu --max-ind-range=10000000
```

#### GPU

Build Dockerfile configuration using the script provided
```
cd $HOME/mlcommons/inference/recommendation/dlrm/pytorch/docker_gpu
./build_docker_gpu.sh
```
The container will have loadgen binary and all other tools needed to run the experiments. DLRM code, Inference code, 
Model, and Data are located on the host machine and can be shared between multiple containers

Edit run_docker.sh to set directories, the defaults are:
HOST_MLCOMMONS_ROOT_DIR=$HOME/mlcommons/inference	# path to mlcommons/inference	
MODEL_DIR=$HOME/mlcommons/model-kaggle			# path to model folder
DATA_DIR=$HOME/mlcommons/data-kaggle			# path to data folder
CUDA_VISIBLE_DEVICES=0					# CUDA devices

```
cd $HOME/mlcommons/inference/recommendation/dlrm/pytorch/docker_gpu
./run_docker_gpu.sh
```

Ensure you have a working docker setup with CUDA support (Should return True); If false ensure you have a functioning Docker installation with CUDA and GPU support.
```
python -c "exec(\"import torch\nprint(torch.cuda.is_available())\")"
```
Nvidia docker support is avalable at https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-docker

Example of running multihot-criteo test on GPU in docker console:
```
cd mlcommons/recommendation/dlrm/pytorch
./run_local.sh multihot-criteo gpu
```

### Examples for testing
During development running the full benchmark is unpractical. Here are some options to help:

`--count-samples` limits the number of items in the dataset used for accuracy pass

`--duration` limits the time the benchmark runs

`--max-latency` the latency used for Server scenario

`--accuracy` enables accuracy pass

So if you want to tune for example Server scenario, try:
```
./run_local.sh pytorch dlrm multihot-criteo cpu --scenario Server  --count-samples 1024 --duration 60000 --target-qps 100 --max-latency 0.1

```

If you want run with accuracy pass, try:
```
./run_local.sh pytorch dlrm multihot-criteo cpu --scenario Offline --count-samples 1024 --samples-per-query-offline=1 --samples-to-aggregate-fix=128 --accuracy
```

### Verifying aggregation trace

In the reference implementation, each sample is mapped to 100-700 user-item pairs following the distribution specified by [tools/dist_quantile.txt](tools/dist_quantile.txt). To verify that your sample aggregation trace matches the reference, please follow the steps in [tools/dist_trace_verification.txt](tools/dist_trace_verification.txt). Or simply download the reference [dlrm_trace_of_aggregated_samples.txt from Zenodo](https://zenodo.org/record/3941795/files/dlrm_trace_of_aggregated_samples.txt?download=1) (MD5:3db90209564316f2506c99cc994ad0b2).

### Running accuracy script

To get the accuracy from a LoadGen accuracy json log file,

1. If your SUT outputs the predictions and the ground truth labels in a packed format like the reference implementation then run
```
python tools/accuracy-dlrm.py --mlperf-accuracy-file <LOADGEN_ACCURACY_JSON>
```
For instance, if the output is given in a standard directory then you can run
```
python ./tools/accuracy-dlrm.py --mlperf-accuracy-file=./output/pytorch-cpu/dlrm/mlperf_log_accuracy.json
```

2. If your SUT outputs only the predictions then you need to make sure that the data in day_23 are not shuffled and run
```
python tools/accuracy-dlrm.py --mlperf-accuracy-file <LOADGEN_ACCURACY_JSON> --day-23-file <path/to/day_23> --aggregation-trace-file <path/to/dlrm_trace_of_aggregated_samples.txt>
```

### Usage
```
usage: main.py [-h]
    [--mlperf_conf ../../mlperf.conf]
    [--user_conf user.conf]
    [--model MODEL] --model-path MODEL_PATH
    [--dataset {debug,multihot-criteo-sample,multihot-criteo}] --dataset-path DATASET_PATH
    [--profile {defaults,dlrm-debug-pytorch,dlrm-multihot-criteo-sample-pytorch,dlrm-multihot-criteo-pytorch}]
    [--scenario SCENARIO]
    [--max-ind-range MAX_IND_RANGE] [--data-sub-sample-rate DATA_SUB_SAMPLE_RATE]
    [--max-batchsize MAX_BATCHSIZE] [--mlperf-bin-loader]
    [--output OUTPUT] [--inputs INPUTS] [--outputs OUTPUTS]
    [--backend BACKEND] [--use-gpu] [--threads THREADS] [--duration TIME_IN_MS]
    [--count-samples COUNT] [--count-queries COUNT] [--target-qps QPS]
    [--max-latency MAX_LATENCY]  [--cache CACHE]
    [--samples-per-query-multistream NUM_SAMPLES]
    [--samples-per-query-offline NUM_SAMPLES]
    [--samples-to-aggregate-fix NUM_FIXED_SAMPLES]
    [--samples-to-aggregate-min MIN_NUM_VARIABLE_SAMPLES]
    [--samples-to-aggregate-max MAX_NUM_VARIABLE_SAMPLES]
    [--samples-to-aggregate-quantile-file FILE]
    [--samples-to-aggregate-trace-file FILE]
    [--numpy-rand-seed SEED]
    [--accuracy] [--find-peak-performance]
```

`--mlperf_conf` the mlperf config file to use for rules compliant parameters (default: ../../mlperf.conf)

`--user_conf` the user config file to use for user LoadGen settings such as target QPS (default: user.conf)

`--model` model name, i.e. `dlrm`.

`--model-path MODEL_PATH` path to the file with model weights.

`--dataset` use the specified dataset. Currently we only support Criteo Terabyte.

`--dataset-path` path to the dataset.

`--scenario {SingleStream,MultiStream,Server,Offline}` benchmarking mode to be used.

`--profile {dlrm-debug-pytorch,dlrm-multihot-criteo-sample-pytorch,dlrm-multihot-criteo-pytorch}` this fills in default command line options with the once specified in the profile. Command line options that follow may override the those.

`--backend` only the PyTorch backedn is currently supported. However, we expect to add TensorFlow backend in the future.

`--max-ind-range` the maximum number of vectors allowed in an embedding table.

`--data-sub-sample-rate` the rate of sub-sampling of negative samples, either 0.875 or 0.0.

`--max-batchsize MAX_BATCHSIZE` maximum batchsize we generate to backend (default: 128). If the query contains a very large number of samples it will be broken up into smaller mini-batches of `MAX_BATCHSIZE` samples before forwarding it to the model.

`--output OUTPUT` location of the JSON output.

`--backend BACKEND` which backend to use. Currently supported is PyTorch.

`--use-gpu` flag that enables use of GPU. The number of GPUs used is controlled by `CUDA_VISIBLE_DEVICES` environment variable.

`--threads THREADS` number of worker threads to use (default: the number of processors in the system).

`--duration` duration of the benchmark run in milliseconds (ms).

`--count-samples COUNT` number of samples from the dataset we use (default: use all samples in the dataset).

`--count-queries COUNT` number of queries we use (default: no limit).

`--target-qps QPS` target/expected QPS for the Server and Offline scenarios.

`--max-latency MAX_LATENCY` comma separated list of which latencies (in seconds) we try to reach in the 99 percentile (default: 0.01,0.05,0.100).

`--samples-per-query-multistream` number of (aggregated) samples per query in MultiStream scenario (default: 8).

`--samples-per-query-offline` maximum number of (aggregated) samples per query in Offline scenario.

`--samples-to-aggregate-fix` number of samples to aggregate and treat as a single sample. This number will stay fixed during runs.

`--samples-to-aggregate-min, --samples-to-aggregate-max` number of samples to aggregate and treat as a single sample. This number will vary randomly between min and max during runs.

`--samples-to-aggregate-quantile-file` number of samples to aggregate and treat as a single sample. This number will be sampled according to a custom distribution quantile stored in a file (e.g. tools/dist_quantile.txt).

`--samples-to-aggregate-trace-file` filename for writing the trace of queries. Each query is written on a single line, with a range of aggregated samples indicated in square brackets.

`--numpy-rand-seed` random seed for numpy package.

`--accuracy` perform inference on the entire dataset to validate achieved model accuracy/AUC metric.

`--find-peak-performance` determine the maximum QPS for the Server, while not applicable to other scenarios.

## License

[Apache License 2.0](LICENSE)
