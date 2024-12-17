# MLPerf™ Inference Benchmark for Graph Neural Network

This is the reference implementation for MLPerf Inference Graph Neural Network. The reference implementation currently uses Deep Graph Library (DGL), and pytorch as the backbone of the model.

**Hardware requirements:** The minimun requirements to run this benchmark are ~600GB of RAM and ~2.3TB of disk. This requires to create a memory map for the graph features and not load them to memory all at once. If you want to load all the features to ram, you will need ~3TB and can be done by using the flag `--in-memory`

## Supported Models

| model | accuracy | dataset | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- |
| RGAT | - | IGBH | [Illiois Graph Benchmark](https://github.com/IllinoisGraphBenchmark/IGB-Datasets/) | fp32 | - |

## Dataset

| Data | Description | Task |
| ---- | ---- | ---- |
| IGBH | Illinois Graph Benchmark Heterogeneous is a graph dataset consisting of one heterogeneous graph with 547,306,935 nodes and 5,812,005,639 edges. Node types: Author, Conference, FoS, Institute, Journal, Paper. A subset of 1% of the paper nodes are randomly choosen as the validation dataset using the [split seeds script](tools/split_seeds.py). The validation dataset will be used as the input queries for the SUT, however the whole dataset is needed to run the benchmarks, since all the graph connections are needed to achieve the quality target. | Node Classification |
| IGBH (calibration) | We sampled 5000 nodes from the training paper nodes of the IGBH for the calibration dataset. We provide the [Node ids](../../calibration/IGBH/calibration.txt) and the [script](tools/split_seeds.py) to generate them (using the `--calibration` flag). | Node Classification |

## Automated command to run the benchmark via MLCommons CM

Please check the official inference documentation [here](https://docs.mlcommons.org/inference/benchmarks/graph/rgat/)
 
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
Finally copy the `mlperf.conf` file to the stable diffusion folder
```bash
cp $ROOT_INFERENCE/mlperf.conf $GRAPH_FOLDER
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

### Download model using Rclone

To run Rclone on Windows, you can download the executable [here](https://rclone.org/install/#windows).
To install Rclone on Linux/macOS/BSD systems, run:
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
Once Rclone is installed, run the following command to authenticate with the bucket:
```
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```
You can then navigate in the terminal to your desired download directory and run the following commands to download the checkpoints:

**`fp32`**
```
rclone copy mlc-inference:mlcommons-inference-wg-public/R-GAT/RGAT.pt $MODEL_PATH -P
```

### Download model through CM (Collective Minds)

```
cm run script --tags=get,ml-model,rgat -j
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

**CM Command**
```
cm run script --tags=get,dataset,igbh,_debug -j
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

**CM Command**
```
cm run script --tags=get,dataset,igbh,_full -j
```


#### Calibration dataset

The calibration dataset contains 5000 nodes from the training paper nodes of the IGBH dataset. We provide the [Node ids](../../calibration/IGBH/calibration.txt) and the [script](tools/split_seeds.py) to generate them (using the `--calibration` flag). 


### Run the benchmark
#### Debug Run
```bash
# Go to the benchmark folder
cd $GRAPH_FOLDER

# Run the benchmark DGL
python3 main.py --dataset igbh-dgl-tiny --dataset-path igbh/ --profile debug-dgl [--model-path <path_to_ckpt>] [--in-memory] [--device <cpu or gpu>] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```

#### Local run
```bash
# Go to the benchmark folder
cd $GRAPH_FOLDER

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
python3 main.py --dataset igbh-dgl --dataset-path igbh/ --profile rgat-dgl-full --device cpu [--model-path <path_to_ckpt>] [--in-memory] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```


**GPU:**
Build docker image
```bash
docker build . -f dockerfile.gpu -t rgat-gpu
```
Run docker container:
```bash
docker run --rm -it -v $(pwd):/root --gpus all rgat-gpu
```
Run benchmark inside the docker container:
```bash
python3 main.py --dataset igbh-dgl --dataset-path igbh/ --profile rgat-dgl-full --device gpu [--model-path <path_to_ckpt>] [--in-memory] [--dtype <fp16 or fp32>] [--scenario <SingleStream, MultiStream, Server or Offline>]
```
