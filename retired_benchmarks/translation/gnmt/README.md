# Current implementation

## Project setup

```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/mlcube && cd mlcube/mlcube
python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..

# Fetch the translation workload
git clone https://github.com/mlcommons/inference && cd ./inference
git fetch origin pull/1022/head:feature/mlcube_translation && git checkout feature/mlcube_translation
cd ./translation/gnmt/mlcube
```

## Tasks execution

```bash
# Download dataset. Default path = /workspace/data
# To override it, use data_dir=DATA_DIR
mlcube run --task download_data

# Download model. Default path = /workspace/model
# To override it, use model_dir=MODEL_DIR
mlcube run --task download_model

# Evaluate performance. Default ouptut path = ./workspace/output
# Parameters to override: data_dir=DATA_DIR, model_dir=MODEL_DIR, parameters_file=PATH_TO_FILE, output_dir=OUTPUT_DIR
mlcube run --task run_performance

# Evaluate accuracy. Default ouptut path = ./workspace/output
# Parameters to override: data_dir=DATA_DIR, model_dir=MODEL_DIR, parameters_file=PATH_TO_FILE, output_dir=OUTPUT_DIR
mlcube run --task run_accuracy
```

**Important:** We are targeting pull-type installation, so MLCubes should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto
```

Also, users can override the workspace directory by using:

```bash
mlcube run --task=download --workspace=absolute_path_to_custom_dir
```
