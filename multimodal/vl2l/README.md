# Reference Implementation for the Vision-language-to-language (VL2L) Benchmark 

## Quick Start

Clone the MLPerf Inference repo via:

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf-inference
```

Then enter the repo: 

```bash
cd mlperf-inference/
```

Follow [this link](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
on how to install Miniconda on your host machine. Then, you can create a new conda 
environment via:

```bash
conda create -n mlperf-inf-mm-vl2l python=3.13
```

### Install LoadGen

Update `libstdc++` in the conda environment:

```bash
conda install -c conda-forge libstdcxx-ng
```

Install `absl-py` and `numpy`:

```bash
conda install absl-py numpy
```

Build and install LoadGen from source:

```bash
cd loadgen/
CFLAGS="-std=c++14 -O3" python -m pip install .
cd ../
```

Run a quick test to validate that LoadGen was installed correctly:

```bash
python loadgen/demos/token_metrics/py_demo_server.py
```

### Install VL2L Benchmark CLI

For users, install `mlperf-inf-mm-vl2l` with:

```bash
pip install multimodal/vl2l/
```

For developers, install `mlperf-inf-mm-vl2l` and the development tools with:

```bash
pip install multimodal/vl2l/[dev]
```

## Developer Guide

### Linting

You can lint the VL2L benchmark source code by running the following script:

```bash
bash multimodal/vl2l/scripts/linters.sh
```