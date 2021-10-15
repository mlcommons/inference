# Current implementation

## Project setup

```bash
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from PyPI
pip install mlcube==0.0.4 mlcube-docker==0.0.4

# Fetch the translation workload
git clone https://github.com/mlcommons/inference && cd ./inference
git fetch origin pull/1025/head:feature/mlcube_translation && git checkout feature/mlcube_translation
cd ./translation/gnmt/mlcube
```

## Tasks execution

**Important:** We are targeting pull-type installation, so MLCubes should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto # or =always
```

```bash
# Download dataset. Default path = /workspace/data
# To override it, use data_dir=DATA_DIR
mlcube run --task download_data

# Download model. Default path = /workspace/model
# To override it, use model_dir=MODEL_DIR
mlcube run --task download_model

# Evaluate performance. Default ouptut path = ./workspace/output
# Parameters to override: data_dir=DATA_DIR, model_dir=MODEL_DIR, parameters_file=PATH_TO_FILE, output_dir=OUTPUT_DIR
mlcube run --task run_inference

# Evaluate accuracy. Default ouptut path = ./workspace/output
# Parameters to override: data_dir=DATA_DIR, model_dir=MODEL_DIR, parameters_file=PATH_TO_FILE, output_dir=OUTPUT_DIR
mlcube run --task run_loadgen
```

Also, users can override the workspace directory by using:

```bash
mlcube run --task=download --workspace=absolute_path_to_custom_dir
```
