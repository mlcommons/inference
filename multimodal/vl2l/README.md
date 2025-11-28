# Reference Implementation for the Vision-language-to-language (VL2L) Benchmark 

## Quick Start

This guide demonstrates how you can run the benchmark on your local machine.

### Create a Conda environment

Follow [this link](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
on how to install Miniconda on your host machine. Then, you can create a new conda 
environment via:

```bash
conda create -n mlperf-inf-mm-vl2l python=3.12
```

### Install the VL2L benchmarking CLI

#### For users

Install `mlperf-inf-mm-vl2l` with:

```bash
pip install git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/vl2l/
```

#### For developers

Clone the MLPerf Inference repo via:

```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf-inference
```

Then enter the repo: 

```bash
cd mlperf-inference/
```

Install `mlperf-inf-mm-vl2l` and the development tools with:

- On Bash
```bash
pip install multimodal/vl2l/[dev]
```
- On Zsh
```zsh
pip install multimodal/vl2l/"[dev]"
```

### Post VL2L benchmarking CLI installation 

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

## Docker

[docker/](docker/) provides examples of Dockerfiles that install the VL2L benchmarking
CLI into the container images of the inference engine. This is useful when you have to
run both the inference engine and the VL2L benchmarking CLI inside the same container,
for example, in a situation where you must use a GPU cluster managed by 
[Slurm](https://slurm.schedmd.com/) with [enroot](https://github.com/nvidia/enroot) and
[pyxis](https://github.com/NVIDIA/pyxis)

## Developer Guide

### Linting

You can lint the VL2L benchmark source code by running the following script:

```bash
bash multimodal/vl2l/scripts/linters.sh
```