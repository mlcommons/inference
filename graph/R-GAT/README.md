# MLPerf™ Inference Benchmark for Graph Neural Network

This is the reference implementation for MLPerf Inference Graph Neural Network. The reference implementation currently uses Deep Graph Library (DGL), and pytorch as the backbone of the model.

**Hardware requirements:** The minimun requirements to run this benchmark are ~600GB of RAM and ~2.3TB of disk. This requires to create a memory map for the graph features and not load them to memory all at once.

## Supported Models

| model | accuracy | dataset | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- |
| RGAT | 0.7286 | IGBH | [Illiois Graph Benchmark](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/) | fp32 | - |

## Dataset

| Data | Description | Task |
| ---- | ---- | ---- |
| IGBH | Illinois Graph Benchmark Heterogeneous is a graph dataset consisting of one heterogeneous graph with 547,306,935 nodes and 5,812,005,639 edges. Node types: Author, Conference, FoS, Institute, Journal, Paper. A subset of 1% of the paper nodes are randomly choosen as the validation dataset using the [split seeds script](tools/split_seeds.py). The validation dataset will be used as the input queries for the SUT, however the whole dataset is needed to run the benchmarks, since all the graph connections are needed to achieve the quality target. | Node Classification |
| IGBH (calibration) | We sampled 5000 nodes from the training paper nodes of the IGBH for the calibration dataset. We provide the [Node ids](../../calibration/IGBH/calibration.txt) and the [script](tools/split_seeds.py) to generate them (using the `--calibration` flag). | Node Classification |

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/graph/rgat/) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.
 
## Setup
Set the following helper variables
```bash
export ROOT_INFERENCE=$PWD/inference
export GRAPH_FOLDER=$PWD/inference/graph/R-GAT/
export LOADGEN_FOLDER=$PWD/inference/loadgen
export MODEL_PATH=$PWD/inference/graph/R-GAT/model/
```
### Clone the repository
```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git --depth 1
```


### Install pytorch
**For NVIDIA GPU based runs:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
**For CPU based runs:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### Install requirements (only for running without using docker)
Install requirements:
```bash
cd $GRAPH_FOLDER
pip install -r requirements.txt
```
Install loadgen:
```bash
cd $LOADGEN_FOLDER
CFLAGS="-std=c++14" python setup.py install
```

### Install pytorch geometric

```bash
export TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html
```

### Install DGL
**For NVIDIA GPU based runs:**
```bash
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```
**For CPU based runs:**
```bash
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
```


### Download model through MLCFlow Automation

```
mlcr get,ml-model,rgat,_r2-downloader,_mlcommons --outdirname=<path_to_download>
```

### Download the model using MLC R2 Downloader

Download the model using the MLCommons R2 Downloader (more information about the MLC R2 Downloader, including how to run it on Windows, can be found [here](https://inference.mlcommons-storage.org)):

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/rgat-model.uri
```


### Download and setup dataset
#### Debug Dataset

**MLC Command**
```
mlcr get,dataset,igbh,_debug --outdirname=<path to download>
```

**Download Dataset**
```bash
cd $GRAPH_FOLDER
python3 tools/download_igbh_test.py
```

**Split Seeds**
```bash
cd $GRAPH_FOLDER
python3 tools/split_seeds.py --path igbh --dataset_size tiny
```



#### Full Dataset
**Warning:** This script will download 2.2TB of data

**MLC Command**
```
mlcr get,dataset,igbh,_full --outdirname=<path to download>
```

```bash
cd $GRAPH_FOLDER
./tools/download_igbh_full.sh igbh/
```

**Split Seeds**
```bash
cd $GRAPH_FOLDER
python3 tools/split_seeds.py --path igbh --dataset_size full
```


#### Calibration dataset

The calibration dataset contains 5000 nodes from the training paper nodes of the IGBH dataset. We provide the [Node ids](../../calibration/IGBH/calibration.txt) and the [script](tools/split_seeds.py) to generate them (using the `--calibration` flag). 

**MLC Command**
```
mlcr get,dataset,igbh,_full,_calibration --outdirname=<path to download>
```

### Run the benchmark
#### Debug Run
```bash
# Go to the benchmark folder
cd $GRAPH_FOLDER

# Run the benchmark DGL
python3 main.py --dataset igbh-dgl-tiny --dataset-path igbh/ --profile debug-dgl [--model-path <path_to_ckpt>] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```


#### Local run
```bash
# Go to the benchmark folder
cd $GRAPH_FOLDER

# Run the benchmark DGL
python3 main.py --dataset igbh-dgl --dataset-path igbh/ --profile rgat-dgl-full [--model-path <path_to_ckpt>] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```

### Evaluate the accuracy
```bash
mlcr process,mlperf,accuracy,_igbh --result_dir=<Path to directory where files are generated after the benchmark run>
```

Please click [here](https://github.com/mlcommons/inference/blob/dev/graph/R-GAT/tools/accuracy_igbh.py) to view the Python script for evaluating accuracy for the IGBH dataset.

## Automated command for submission generation via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/submission/) for an automated way to generate submission through MLCFlow. 

#### Run using docker

Not implemented yet

#### Accuracy run
Add the `--accuracy` to the command to run the benchmark
```bash
python3 main.py --dataset igbh --dataset-path igbh/ --accuracy --model-path model/ [--model-path <path_to_ckpt>] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>] [--layout <COO, CSC or CSR>]
```

**NOTE:** For official submissions you should submit the results of the accuracy run in a file called `accuracy.txt` with the following format:
```
accuracy=<accuracy>%, good=<number_of_good_samples>, total=<number_of_total_samples>
hash=<hash>
```

### Docker run
**CPU:**
Build docker image
```bash
docker build . -f dockerfile.cpu -t rgat-cpu
```
Run docker container:
```bash
docker run --rm -it -v $(pwd):/root rgat-cpu
```
Run benchmark inside the docker container:
```bash
python3 main.py --dataset igbh-dgl --dataset-path igbh/ --profile rgat-dgl-full --device cpu [--model-path <path_to_ckpt>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```


**GPU:**
Build docker image
```bash
docker build . -f dockerfile.gpu -t rgat-gpu
```
Run docker container:
```bash
docker run --rm -it -v $(pwd):/workspace/root --gpus all rgat-gpu
```
Go inside the root folder and run benchmark inside the docker container:
```bash
cd root
python3 main.py --dataset igbh-dgl --dataset-path igbh/ --profile rgat-dgl-full --device gpu [--model-path <path_to_ckpt>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```

**NOTE:** For official submissions, this benchmark is required to run in equal issue mode. Please make sure that the flag `rgat.*.sample_concatenate_permutation` is set to one in the [mlperf.conf](../../loadgen/mlperf.conf) file when loadgen is built.
