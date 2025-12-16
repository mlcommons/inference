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
pip install -e multimodal/vl2l/[dev]
```
- On Zsh
```zsh
pip install -e multimodal/vl2l/"[dev]"
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

> [!NOTE]
> Shell auto-completion will take effect once you restart the terminal.

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
mlperf-inf-mm-vl2l benchmark endpoint --settings.test.scenario offline --settings.test.mode performance_only
```

Accuracy only mode:

```bash
mlperf-inf-mm-vl2l benchmark endpoint --settings.test.scenario offline --settings.test.mode accuracy_only
```

### Run the benchmark for the Server scenario

Performance only mode:

```bash
mlperf-inf-mm-vl2l benchmark endpoint --settings.test.scenario server --settings.test.mode performance_only
```

Accuracy only mode:

```bash
mlperf-inf-mm-vl2l benchmark endpoint --settings.test.scenario server --settings.test.mode accuracy_only
```

### Evalute the response quality

```bash
mlperf-inf-mm-vl2l evaluate --filename output/mlperf_log_accuracy.json
```

## Docker

[docker/](docker/) provides examples of Dockerfiles that install the VL2L benchmarking
CLI into the container images of the inference engine. This is useful when you have to
run both the inference engine and the VL2L benchmarking CLI inside the same container,
for example, in a situation where you must use a GPU cluster managed by 
[Slurm](https://slurm.schedmd.com/) with [enroot](https://github.com/nvidia/enroot) and
[pyxis](https://github.com/NVIDIA/pyxis).

As an illustrative example, assuming that you are at the root directory of the MLPerf 
Inference repo:

1. You can build a container image against the vLLM's
`vllm/vllm-openai:v0.12.0` release by

```bash
docker build \
    --build-arg BASE_IMAGE_URL=vllm/vllm-openai:v0.12.0 \
    --build-arg MLPERF_INF_MM_VL2L_INSTALL_URL=multimodal/vl2l \
    -f multimodal/vl2l/docker/vllm-cuda.Dockerfile \
    -t mlperf-inf-mm-vl2l:vllm-openai-v0.12.0 \
    .
```
> [!NOTE]
> `MLPERF_INF_MM_VL2L_INSTALL_URL` can also take in a remote GitHub location, such as
> `git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/vl2l/`.

2. Afterwards, you can start the container in the interactive mode by

```bash
docker run --rm -it --gpus all -v ~/.cache:/root/.cache --ipc=host mlperf-inf-mm-vl2l:vllm-openai-v0.12.0
```

### Benchmark against vLLM inside the container

If you are running `mlperf-inf-mm-vl2l` inside a local environment that has access to
vLLM (such as inside a container that was created using the 
[docker/vllm-cuda.Dockerfile](docker/vllm-cuda.Dockerfile)), you can use a single
`mlperf-inf-mm-vl2l benchmark vllm` command to achieve:

1. Deploy an endpoint using vLLM.
2. Wait for the endpoint to be healthy.
3. Run the benchmark against that endpoint.

For example, inside the container, you can run the Offline scenario Accuracy only
mode with:

```bash
mlperf-inf-mm-vl2l benchmark vllm \
    --settings.test.scenario offline \
    --settings.test.mode accuracy_only \
    --dataset.token ... \
    --vllm.cli=--async-scheduling \
    --vllm.cli=--max-model-len=32768 \
    --vllm.cli=--max-num-seqs=1024 \
    --vllm.cli=--compilation-config='{
        "cudagraph_capture_sizes": [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
            256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
            496, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768
        ]
    }' \
    --vllm.cli=--limit-mm-per-prompt.video=0 \
    --vllm.cli=--tensor-parallel-size=8 
```

## Slurm

[scripts/slurm/](scripts/slurm/) provide example scripts of running both the benchmark 
and the response quality evaluation in a GPU cluster managed by 
[Slurm](https://slurm.schedmd.com/) with [enroot](https://github.com/nvidia/enroot) and
[pyxis](https://github.com/NVIDIA/pyxis). Specifically,

- [scripts/slurm/benchmark.sh](scripts/slurm/benchmark.sh) is a sbatch script that 
  runs the benchmarking job.
- [scripts/slurm/evaluate.sh](scripts/slurm/evaluate.sh) is a sbatch script that runs
  the evaluation job.
- [scripts/slurm/submit.sh](scripts/slurm/submit.sh) is a Bash script that submits both
  jobs, where the evaluation job would only run if the benchmarking job has succeeded.

You can check the CLI flags that [scripts/slurm/submit.sh](scripts/slurm/submit.sh) can
take via:

```bash
bash submit.sh --help
```

> [!NOTE]
> Slurm clusters are often highly customized per organization. If you are unfamiliar
> with Slurm, you should check with the cluster administrator of your organization
> first, get a good understanding of what those example scripts do, and adapt the 
> example scripts to the specific settings for the Slurm cluster that you are going
> to use, before you try to launch any jobs.

## Reference Implementation Specification

- v6.0 Round
  - vLLM version: [v0.12.0](https://github.com/vllm-project/vllm/releases/tag/v0.12.0)
  - Model:
    - [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
    - Commit SHA: [710c13861be6c466e66de3f484069440b8f31389](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/tree/710c13861be6c466e66de3f484069440b8f31389)
  - Dataset:
    - [Shopify/product-catalogue](https://huggingface.co/datasets/Shopify/product-catalogue)
    - Commit SHA: [d5c517c509f5aca99053897ef1de797d6d7e5aa5](https://huggingface.co/datasets/Shopify/product-catalogue/tree/d5c517c509f5aca99053897ef1de797d6d7e5aa5)
  - Constraint:
    - Model quality:
      - Category Hierarchical F1 Score >= `0.7824`.
    - Server Scenario:
      - Target latency percentile = `0.99`.
      - Target latency <= 12 seconds.

## Developer Guide

### Linting

You can lint the VL2L benchmark source code by running the following script:

```bash
bash multimodal/vl2l/scripts/linters.sh
```