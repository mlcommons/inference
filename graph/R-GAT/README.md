# MLPerf™ Inference Benchmarks for Text to Image

This is the reference implementation for MLPerf Inference text to image. Two implementation are currently supported, Graphlearn for Pytorch (GLT) and Deep Graph Library (DGL), both using pytorch as the backbone of the model.

## Supported Models

| model | accuracy | dataset | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- |
| RGAT | - | IGBH | [Illiois Graph Benchmark](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/) | fp32 | - |

## Dataset

| Data | Description | Task |
| ---- | ---- | ---- |
| IGBH | Illinois Graph Benchmark Heterogeneous is a graph dataset consisting of one heterogeneous graph with 547,306,935 nodes and 5,812,005,639 edges. Node types: Author, Conference, FoS, Institute, Journal, Paper. A subset of 1% of the paper nodes are randomly choosen as the validation dataset using the [split seeds script](tools/split_seeds.py). The validation dataset will be used as the input queries for the SUT, however the whole dataset is needed to run the benchmarks, since all the graph connections are needed to achieve the quality target. | Node Classification |
| IGBH (calibration) | We sampled x nodes from the training paper nodes of the IGBH for the calibration dataset | Node Classification |

## Automated command to run the benchmark via MLCommons CM

TODO
 
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
git clone --recurse-submodules https://github.com/mlcommmons/inference.git --depth 1
```
Finally copy the `mlperf.conf` file to the stable diffusion folder
```bash
cp $ROOT_INFERENCE/mlperf.conf $GRAPH_FOLDER
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
```bash
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```

### Install graphlearn for pytorch (Only for GLT implementation)

Follow instalation instructions at: https://github.com/alibaba/graphlearn-for-pytorch.git

### Download model

TODO: fix a ckpt url
```bash
mkdir -p $MODEL_PATH
wget <checkpoint url>
```

### Download and setup dataset
#### Debug Dataset

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

**Compress graph (optional, only for GLT implementation)**
```bash
cd $GRAPH_FOLDER
python3 tools/compress_graph.py --path igbh --dataset_size tiny --layout <CSC or CSR>
```

#### Full Dataset
**Warning:** This script will download 2.2TB of data 
```bash
cd $GRAPH_FOLDER
./tools/download_igbh_full.sh igbh/
```

**Split Seeds**
```bash
cd $GRAPH_FOLDER
python3 tools/split_seeds.py --path igbh --dataset_size full
```

**Compress graph (optional, only for GLT implementation)**
```bash
cd $GRAPH_FOLDER
python3 tools/compress_graph.py --path igbh --dataset_size tiny --layout <CSC or CSR>
```

#### Calibration dataset
TODO


### Run the benchmark
#### Debug Run
```bash
# Go to the benchmark folder
cd $GRAPH_FOLDER
# Run the benchmark GLT
python3 main.py --dataset igbh-glt-tiny --dataset-path igbh/ --profile debug-glt [--model-path <path_to_ckpt>] [--in-memory] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>] [--layout <COO, CSC or CSR>]

# Run the benchmark DGL
python3 main.py --dataset igbh-dgl-tiny --dataset-path igbh/ --profile debug-dgl [--model-path <path_to_ckpt>] [--in-memory] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```

#### Local run
```bash
# Go to the benchmark folder
cd $GRAPH_FOLDER
# Run the benchmark GLT
python3 main.py --dataset igbh-glt --dataset-path igbh/ --profile rgat-glt-full [--model-path <path_to_ckpt>] [--in-memory] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>] [--layout <COO, CSC or CSR>]

# Run the benchmark DGL
python3 main.py --dataset igbh-dgl --dataset-path igbh/ --profile rgat-dgl-full [--model-path <path_to_ckpt>] [--in-memory] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```
#### Run using docker

Not implemented yet

#### Accuracy run
Add the `--accuracy` to the command to run the benchmark
```bash
python3 main.py --dataset igbh --dataset-path igbh/ --accuracy --model-path model/ [--model-path <path_to_ckpt>] [--in-memory] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>] [--layout <COO, CSC or CSR>]
```
