# MLCommons (MLPerf) Inference Benchmarks for Recommendation Task

This is the reference implementation for MLCommons Inference benchmarks.

### Supported Models

| name | framework | acc. | AUC | dataset | weights  | size | prec. | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| dlrm (debugging) | PyTorch | 78.82% | N/A | [Criteo KaggleDAC](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)       | N/A                                                                     | ~1GB | fp32 |                          |
| dlrm (debugging) | PyTorch | 81.07% | N/A | [Criteo Terabyte](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/) | [pytorch](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt), [onnx](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.onnx.tar) | ~10GB | fp32 | --max-ind-range=10000000 --data-sub-sample-rate=0.875 |
| dlrm (official) | PyTorch | N/A | 80.25% | [Criteo Terabyte](https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/) | [pytorch](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt), [onnx](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.onnx.tar) | ~100GB | fp32 | --max-ind-range=40000000 |

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

Install the desired backend. For pytoch:
```
pip install torch torchvision
pip install scikit-learn
pip install numpy
pip install pydot
pip install torchviz
pip install protobuf
pip install tqdm
pip install onnxruntime
```

### Prepare the code and dataset
1. Download or clone the MLCommons [inference](https://github.com/mlcommons/inference) and [training](https://github.com/mlcommons/training) code
```
cd $HOME
mkdir ./mlcommons && cd ./mlcommons
git clone --recurse-submodules https://github.com/mlcommons/training.git
git clone --recurse-submodules https://github.com/mlcommons/inference.git
export DLRM_DIR=$HOME/mlcommons/training/recommendation/dlrm
```
2. Download pre-trained model weights (see links available above)
```
cd $HOME/mlcommons/inference/recommendation
mkdir ./model && cd ./model
mv <downloaded_file> dlrm_terabyte.pytorch
export MODEL_DIR=./model
```
3. Download corresponding Criteo dataset (see links available above)
```
cd $HOME/mlcommons/inference/recommendation
mkdir ./criteo && cd ./criteo
mv <downloaded_file(s)> ./
export DATA_DIR=./criteo
```
4. Build and install the loadgen
```
cd $HOME/mlcommons/inference/loadgen
CFLAGS="-std=c++14" python setup.py develop --user
```

### More information about the model weights

File name | framework | Size in bytes (`du *`) | MD5 hash (`md5sum *`)
-|-|-|-
[tb0875_10M.pt](https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt) | pytorch | 12GB | b7cacffcf75f767faa9cb2af397723aa
[tb00_40M.pt](https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt) | pytorch | 90GB | 2d49a5288cddb37c3c64860a06d79bb9

### More information about the datasets
| dataset | download link |
| ---- | ---- |
| Criteo Kaggle DAC (not active) | https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/ |
| Criteo Terabyte   | https://labs.criteo.com/2013/12/download-terabyte-click-logs/ |

1. The Criteo Kaggle DAC dataset is composed of 7 days, which are stored in file: `train.txt`. This file is expected by the code.


2. The Criteo Terabyte dataset is stored in several files corresponding to 24 days: `day_0.gz`, `day_1.gz`, ..., `day_23.gz` (~343GB).

File name | Size in bytes (`du *`) | MD5 hash (`md5sum *`)
-|-|-
`day_0.gz`  | 15927304 | 6cef23542552c3195e9e6e2bdbc4c235
`day_1.gz`  | 16292832 | 94b73908ee8f912c175420376c8952db
`day_2.gz`  | 16022296 | c3c0272c26cfaa03d932b2856a751ff5
`day_3.gz`  | 14779644 | b727ecfaaf482507bb998248833aa4c2
`day_4.gz`  | 12514396 | b99eaa6e324e49d9df4e6f840e76b6d9
`day_5.gz`  | 14169436 | 1294d0a56a90294aebf80133078d9879
`day_6.gz`  | 16753584 | 68586521483313e13aefb29e7b767fdb
`day_7.gz`  | 16465452 | a2c1c4bfec20fc88b0b504175a16b644
`day_8.gz`  | 15845316 | faabf247fd56140a76effa6a3da63253
`day_9.gz`  | 16195668 | ee3347a28c1dd2fb2c92094e643c132b
`day_10.gz` | 15201768 | d043c2ec0346eb4c71aaae935416688e
`day_11.gz` | 12698432 | 8d4ba32f0c4f654a3860b6f2ae1a8ea7
`day_12.gz` | 14029428 | 908480917ed39be2a2ad2e1c339c40b4
`day_13.gz` | 15953380 | 567d6bfa672dd10a0cf76feaec0cf92b
`day_14.gz` | 15882872 | ed377357aecaccc5f93c754c4819fd8d
`day_15.gz` | 15382724 | 8e91f2a8d3d95202dfc3b22b88064c12
`day_16.gz` | 14606256 | 387269870bf8ec7d285cf0e8ce82e92e
`day_17.gz` | 13483480 | 48d3538fcf04807e0be4d72072dbda0b
`day_18.gz` | 11796984 | f26e23b6ef242f40b0e3fd92c986170c
`day_19.gz` | 13059032 | 3f6f36657b0ff1258428356451eea6c8
`day_20.gz` | 16022492 | db7ff2b830817d3b10960f02bfb68547
`day_21.gz` | 15707044 | f1a4ba7f7a555cb4a7e724a082479f4a
`day_22.gz` | 15463520 | 848ae20c4eab730ae487acc8ddaf52ba
`day_23.gz` | 14591792 | a2748bdbc67dd544b3ac470c4f1a52df


Please unzip all the files:
```
gunzip day_{0..23}.gz
```
to obtain the text files expected by the code: `day_0`, `day_1`, ..., `day_23` (~1.1TB).

File name | Size in bytes (`du *`) | MD5 hash (`md5sum *`)
-|-|-
`day_0`  |  48603344  |  331703bf14b9a699324d589efd676962
`day_1`  |  49518248  |  1df068493bb19edce48c09a1ce1a7fca
`day_2`  |  48820460  |  a5dec1724865504895e03508d6308046
`day_3`  |  44863556  |  c6de260d9eb835a2a1866b9931f4c474
`day_4`  |  37583076  |  9a41d00282b6b87db890518a86ab001f
`day_5`  |  42691840  |  170cba5a53f2b7cea8b75e19f10e152e
`day_6`  |  50857184  |  a47640af3fdffdec69c7a61f74e8b4f6
`day_7`  |  49874324  |  b0ff0428ce74fcb3f690f5330b7803d7
`day_8`  |  48082108  |  89113bd1eed24775ff1e757feb80447a
`day_9`  |  49252856  |  730903c294a98261beefc5d8c1634fc9
`day_10` |  46029928  |  994fb3a9a43c0ebef0e28357a3b34068
`day_11` |  37983540  |  a62384802da42c8e0a82aa0a26e5f99e
`day_12` |  41832600  |  3f4dc3fca55231e7e9cff5a696325a28
`day_13` |  48109928  |  2ad1cddf9a1b93a6315ba29ceb31124c
`day_14` |  48059080  |  969bf0ec8cc212fe1816c153161df367
`day_15` |  46383644  |  fb44b44d68237d694b56f85f603f52cb
`day_16` |  44094432  |  866d4b3ef3381fec2c77afee58dcb987
`day_17` |  40374712  |  89e82426b5bc49e93b40d9dbd1bbf7a4
`day_18` |  35039044  |  294f2d46c8653d36c12c163d9f91b5ac
`day_19` |  38662560  |  74bc0f7b0b6dd324ecc029445c875ea3
`day_20` |  47981304  |  5b02f47b002cd374613d8e27f2daa7ce
`day_21` |  47650792  |  70dfaf349be746e3ea1a7affe97f40b8
`day_22` |  47037532  |  c9abe6cbae0b93f6702af27cda53d229
`day_23` |  44152268  |  08e251af4f3d1e8771ea15e405f39600


3. The Criteo fake dataset can be created in place of the real datasets in order to facilitate debugging and testing. We provide a fake (random) data generator that can be used to quickly generate data samples in a format compatible with both original and mlcommons binary loaders. Please use the following script in `./tools` to quickly create random samples for the corresponding models, which will be placed into `./fake_criteo` directory.
```
./make_fake_criteo.sh [kaggle|terabyte0875|terabyte]
mv ./fake_criteo .. && cd ..
export DATA_DIR=./fake_criteo
```
### Calibration set

For MLPerf Inference, we use the first 128000 rows (user-item pairs) of the second half of `day_23` as the calibration set. Specifically, `day_23` contains 178274637 rows in total, so we use the rows **from the 89137319-th row to the 89265318-th row (both inclusive) in `day_23`** as the calibration set (assuming 0-based indexing).

## Running the benchmark

Download and install all the pre-requisites. Both local and docker environment need to set 3 environment variables:
```
export DATA_DIR=YourCriteoTerabyteLocation
export MODEL_DIR=YourModelFileLocation
export DLRM_DIR=YourDLRMSourceLocation
```

### Run local
```
./run_local.sh backend model dataset device [options]

backend is one of [pytorch]
model is one of [dlrm]
dataset is one of [kaggle|terabyte]
device is one of [cpu|gpu]
options are extra arguments that are passed along
```

For example, to run on CPU you may choose to use:

1. Criteo Kaggle DAC (debugging)

Offline scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm kaggle cpu --scenario Offline --samples-to-aggregate-fix=2048 --max-batchsize=2048
./run_local.sh pytorch dlrm kaggle cpu --scenario Offline --samples-to-aggregate-fix=2048 --max-batchsize=2048 --samples-per-query-offline=1 --accuracy
```
Server scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm kaggle cpu --scenario Server --samples-to-aggregate-fix=2048 --max-batchsize=2048
./run_local.sh pytorch dlrm kaggle cpu --scenario Server --samples-to-aggregate-fix=2048 --max-batchsize=2048 --accuracy
```

2. Criteo Terabyte with 0.875 sub-sampling (debugging)

Offline scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=10000000 --data-sub-sample-rate=0.875 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 [--mlperf-bin-loader]
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=10000000 --data-sub-sample-rate=0.875 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=1 --accuracy [--mlperf-bin-loader]
```
Server scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --max-ind-range=10000000 --data-sub-sample-rate=0.875 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 [--mlperf-bin-loader]
./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --max-ind-range=10000000 --data-sub-sample-rate=0.875 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --accuracy [--mlperf-bin-loader]
```

3. Criteo Terabyte (official)

Offline scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800 [--mlperf-bin-loader]
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800 --accuracy [--mlperf-bin-loader]
```
Server scenario perf and accuracy modes
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 [--mlperf-bin-loader]
./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --accuracy [--mlperf-bin-loader]
```

Note that the code support (i) original and (ii) mlperf binary loader, that have slightly different performance characteristics. The latter loader can be enabled by adding `--mlperf-bin-loader` to the command line.

Note that this script will pre-process the data during the first run and reuse it over sub-sequent runs. The pre-processing of data can take a significant amount of time during the first run.

In order to use GPU(s), select the number of GPUs with the environment variable `CUDA_VISIBLE_DEVICES`, and run
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./run_local.sh pytorch dlrm terabyte gpu --accuracy
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
DLRM_DIR=$HOME/mlcommons/dlrm				# path to DLRM			
MODEL_DIR=$HOME/mlcommons/model-terabyte		# path to model folder
DATA_DIR=$HOME/mlcommons/data-terabyte			# path to data folder

Run Docker container in interactive mode and enter the docker console
```
cd $HOME/mlcommons/inference/recommendation/dlrm/pytorch/docker_cpu
./run_docker_cpu.sh
```

Example of running terabyte test on CPU in docker console:
```
cd mlcommons/recommendation/dlrm/pytorch
./run_local.sh terabyte cpu --max-ind-range=10000000
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
DLRM_DIR=$HOME/mlcommons/dlrm				# path to DLRM			
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

Example of running terabyte test on GPU in docker console:
```
cd mlcommons/recommendation/dlrm/pytorch
./run_local.sh kaggle gpu
```

### Examples for testing
During development running the full benchmark is unpractical. Here are some options to help:

`--count-samples` limits the number of items in the dataset used for accuracy pass

`--duration` limits the time the benchmark runs

`--max-latency` the latency used for Server scenario

`--accuracy` enables accuracy pass

So if you want to tune for example Server scenario, try:
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Server  --count-samples 1024 --max-ind-range=10000000 --data-sub-sample-rate=0.875 --duration 60000 --target-qps 100 --max-latency 0.1

```

If you want run with accuracy pass, try:
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --count-samples 1024 --max-ind-range=10000000 --data-sub-sample-rate=0.875 --samples-per-query-offline=1 --samples-to-aggregate-fix=128 --accuracy [--mlperf-bin-loader]
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
    [--dataset {kaggle,terabyte}] --dataset-path DATASET_PATH
    [--profile {defaults,dlrm-kaggle-pytorch,dlrm-terabyte-pytorch}]
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

`--profile {dlrm-kaggle-pytorch,dlrm-terabyte-pytorch}` this fills in default command line options with the once specified in the profile. Command line options that follow may override the those.

`--backend` only the PyTorch backedn is currently supported. However, we expect to add TensorFlow backend in the future.

`--max-ind-range` the maximum number of vectors allowed in an embedding table.

`--data-sub-sample-rate` the rate of sub-sampling of negative samples, either 0.875 or 0.0.

`--max-batchsize MAX_BATCHSIZE` maximum batchsize we generate to backend (default: 128). If the query contains a very large number of samples it will be broken up into smaller mini-batches of `MAX_BATCHSIZE` samples before forwarding it to the model.

`--mlperf-bin-loader` flag that enables mlperf binary loader to be used.

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
