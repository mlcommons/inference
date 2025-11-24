# Reference Implementation for the Vision-language-to-language (VL2L) Benchmark 

## Quick Start

### Create a Conda environment

Follow [this link](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
on how to install Miniconda on your host machine. Then, you can create a new conda 
environment via:

```bash
conda create -n mlperf-inf-mm-vl2l python=3.12
```

### (Optionally) Update `libstdc++` in the conda environment:

```bash
conda install -c conda-forge libstdcxx-ng
```

This is only needed when your local environment doesn't have `libstdc++`.

### Install the VL2L benchmarking CLI

For users, install `mlperf-inf-mm-vl2l` with:

```bash
pip install git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/vl2l
```

For developers, install `mlperf-inf-mm-vl2l` and the development tools with:
1. Clone the MLPerf Inference repo.
```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf-inference
```

2. Install in editable mode with the development tools.
- Bash: `pip install mlperf-inference/multimodal/vl2l/[dev]`
- Zsh: `pip install mlperf-inference/multimodal/vl2l/"[dev]"`

After installation, you can check the CLI flags that `mlperf-inf-mm-vl2l` can take with:

```bash
mlperf-inf-mm-vl2l --help
```

You can enable shell autocompletion for `mlperf-inf-mm-vl2l` with:

```bash
mlperf-inf-mm-vl2l --install-completion
```

> NOTE: Shell auto-completion will take effect once you restart the terminal.

### Start an inference endpoint on your local host machine with vLLM

Please refer to [this guide on how to launch vLLM for various Qwen3 VL MoE models](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html).

```bash
docker run --gpus all \                                 # Use all the GPUs on this host machine.
    -v ~/.cache/huggingface:/root/.cache/huggingface \  # Use the HuggingFace cache from your host machine.
    -p 8000:8000 \                                      # This assumes the endpoint will use port 8000.
    --ipc=host \                                        # The container can access and utilize the host's IPC mechanisms (e.g., shared memory).
    vllm/vllm-openai:nightly \                          # You can also use the `:latest` container or a specific release.
        --model Qwen/Qwen3-VL-235B-A22B-Instruct \      # Specifies the model for vLLM to deploy.
        --tensor-parallel-size 8 \                      # 8-way tensor-parallel inference across 8 GPUs.
        --limit-mm-per-prompt.video 0                   # The input requests will contain images only (i.e., no videos).
```

### Run the benchmark for the Offline scenario

Performance only mode:

```bash
mlperf-inf-mm-vl2l --settings.test.scenario offline --settings.test.mode performance_only
```

Accuracy only mode:

```bash
mlperf-inf-mm-vl2l --settings.test.scenario offline --settings.test.mode accuracy_only
```

### Run the benchmark for the Server scenario

Performance only mode:

```bash
mlperf-inf-mm-vl2l --settings.test.scenario server --settings.test.mode performance_only
```

Accuracy only mode:

```bash
mlperf-inf-mm-vl2l --settings.test.scenario server --settings.test.mode accuracy_only
```

## Developer Guide

### Linting

You can lint the VL2L benchmark source code by running the following script:

```bash
bash multimodal/vl2l/scripts/linters.sh
```