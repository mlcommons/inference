# MLPerf Inference DeepSeek Reference Implementation

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/language/deepseek-r1/) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do pip install mlc-scripts and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Model & Dataset Download

> **Model**: [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (revision: `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad`)

- DeepSeek-R1 model is automatically downloaded as part of setup
- Checkpoint conversion is done transparently when needed.

**Using MLCFlow Automation**

Download the model using the MLCFlow Automation:

```
mlcr get,ml-model,deepseek-r1,_r2-downloader,_mlc --outdirname=<path to download> -j
```

**Using the MLC R2 Downloader**

Download the model using the MLCommons R2 Downloader:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/deepseek-r1-0528.uri
```

To specify a custom download directory, use the `-d` flag:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /path/to/download/directory \
  https://inference.mlcommons-storage.org/metadata/deepseek-r1-0528.uri
```

## Dataset Download

The dataset is an ensemble of the datasets: AIME, MATH500, gpqa, MMLU-Pro, livecodebench(code_generation_lite). They are covered by the following licenses:

- AIME: [CC0](https://creativecommons.org/public-domain/cc0/)
- MATH500: [MIT](https://opensource.org/license/mit), based on [original paper](https://arxiv.org/pdf/2103.03874)
- gpqa: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- MMLU-Pro: [MIT](https://opensource.org/license/mit)
- livecodebench(code_generation_lite): [CC](https://creativecommons.org/share-your-work/cclicenses/)

### Preprocessed & Calibration

**Using the MLC R2 Downloader**

Download the full preprocessed dataset and calibration dataset using the MLCommons R2 Downloader:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
-d ./ https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri
```

This will download the full preprocessed dataset file (`mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl`) and the calibration dataset file (`mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl`).

To specify a custom download directory, use the `-d` flag:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d /path/to/download/directory \
  https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri
```

### Preprocessed

**Using MLCFlow Automation**

```
mlcr get,preprocessed,dataset,deepseek-r1,_validation,_mlc,_r2-downloader --outdirname=<path to download> -j
```

### Calibration

**Using MLCFlow Automation**

```
mlcr get,preprocessed,dataset,deepseek-r1,_calibration,_mlc,_r2-downloader --outdirname=<path to download> -j
```

## Docker

The MLPerf DeepSeek reference implementation includes a comprehensive Docker launch system that supports multiple backends and provides advanced features like user management, persistent storage, and flexible configuration.

### Launch Backend Specific Container

Launch a Docker container with your preferred backend:

```bash
# Launch PyTorch backend
./launch_docker.sh --backend pytorch

# Launch vLLM backend
./launch_docker.sh --backend vllm

# Launch SGLang backend
./launch_docker.sh --backend sglang

# See launch_docker.sh for full list of args
./launch_docker.sh --backend vllm --gpu-count 2 --extra-mounts "/data:/data,/models:/models" --local-user 0
```

### Available Backends

- **pytorch**: via [DeepSeek-Ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) (reference implementation by DeepSeek-Ai)
- **vllm**: vLLM's LLM api-based inference
- **sglang**: sglang's OpenAI endpoint-based inference

**NOTE**: `sglang` backend uses `sglang==0.5.4` installed into `lmsysorg/sglang:v0.5.2-cu129-b200` base image.

## Backend-Specific Setup

After launching any Docker container, run the setup script which automatically detects your backend:

```bash
# Automatic backend detection and setup
setup.sh
```

The setup script creates a virtual environment and configures it differently based on the backend:

#### All Backends

- Virtual environment is **activated** after `setup.sh`
- Activate backend-specific venv using `source .venv_[pytorch|vllm|sglang]/bin/activate`
- All commands are to be run using the virtual environment

## Running Evaluations

### PyTorch Backend (Distributed)

> ⚠️ **IMPORTANT NOTE**: The PyTorch reference implementation takes approximately upto 8 days to run on an H200x8 system. This is because large max-OSL (20K) limits concurrency (max-BS=16), and unoptimized pytorch forward and decode logics.

PyTorch backend uses distributed execution with `torchrun` and `run_eval_mpi.py`:

```bash
# Regular inference evaluation
(.venv_pytorch) $ torchrun --nproc_per_node=8 run_eval_mpi.py --input-file <input_dataset>.pkl --output-file pytorch_output.pkl --num-samples 32

# MLPerf performance benchmarks
(.venv_pytorch) $ torchrun --nproc_per_node=8 run_mlperf_mpi.py --mode offline --input-file <input_dataset>.pkl --output-dir mlperf_results

# MLPerf accuracy mode
(.venv_pytorch) $ torchrun --nproc_per_node=8 run_mlperf_mpi.py --mode offline --accuracy --input-file <input_dataset>.pkl --output-dir mlperf_results
```

### vLLM and SGLang Backends

For vLLM and SGLang, use single-process execution in `run_eval.py`:

```bash
# Regular inference evaluation
(.venv_vllm) $ python run_eval.py --input-file <input_dataset>.pkl
(.venv_sglang) $ python run_eval.py --input-file <input_dataset>.pkl

# MLPerf performance benchmarks
(.venv_vllm) $ python run_mlperf.py --mode offline --input-file <input_dataset>.pkl --output-dir mlperf_results
(.venv_sglang) $ python run_mlperf.py --mode server --input-file <input_dataset>.pkl --output-dir mlperf_results
```

## MLPerf Inference Support

The reference implementation includes full support for MLPerf inference benchmarks through a System Under Test (SUT) wrapper that integrates with MLPerf LoadGen.

### Running MLPerf Benchmarks

#### Offline Scenario

```bash
(.venv_BACKEND) $ python run_mlperf.py \
    --mode offline \
    --input-file <input_dataset>.pkl \
    --output-dir mlperf_results
```

#### Server Scenario

```bash
(.venv_BACKEND) $ python run_mlperf.py \
    --mode server \
    --input-file <input_dataset>.pkl \
    --output-dir mlperf_results
```

#### Interactive Scenario

```bash
(.venv_BACKEND) $ python run_mlperf.py \
    --mode interactive \
    --input-file <input_dataset>.pkl \
    --output-dir mlperf_results
```

**NOTE:** to enable Speculative Decoding for Sglang Backend, toggle `BACKEND_REGISTRY['sglang']['enable_speculative_decode']` in `utils/backend_registry.py` (disabled by default).

#### Pytorch Backend for Mlperf

PyTorch backend uses distributed execution with `torchrun` and `run_mlperf_mpi.py`:

```bash
# PyTorch MLPerf offline scenario
(.venv_BACKEND) $ torchrun --nproc_per_node=8 run_mlperf_mpi.py \
    --mode offline \
    --input-file <input_dataset>.pkl \
    --output-dir mlperf_results
```

### MLPerf Command Line Options

| Option         | Description                                | Default          |
| -------------- | ------------------------------------------ | ---------------- |
| `--mode`       | Scenario mode (offline/server/interactive) | `offline`        |
| `--accuracy`   | Run accuracy test                          | `False`          |
| `--output-dir` | Output directory for results               | `mlperf_results` |

### Backend Support Matrix

The following table shows which backends support different evaluation and MLPerf operations:

| Backend     | `run_eval.py` | `run_mlperf.py --mode=offline` | `run_mlperf.py --mode=server` | `run_mlperf.py --mode=interactive` |
| ----------- | ------------- | ------------------------------ | ----------------------------- | ---------------------------------- |
| pytorch-fp8 | x             | x                              |                               |                                    |
| vllm-fp8    | x             | x                              |                               |                                    |
| sglang-fp8  | x             | x                              | x                             | x                                  |

> **Note**: For PyTorch backend, use the `_mpi` versions with `torchrun`. For vLLM and SGLang backends, use the single-process versions without `_mpi`.

## Speculative Decoding

For the DeepSeek-R1 Interactive Scenario, users can enable Speculative Decoding Optimization for the SGLANG Backend by setting the `enable_speculative_decode` flag to `True` in `language/deepseek-r1/utils/backend_registry.py`.

When Enabled, SGLANG backend will run the allowed configuration as per [Inference Policies](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) (appendix-speculative-decoding):

| Benchmark   | Scenario    | Speculative Decoding Algorithm                             | Configuration                                           | MTP Head                                       |
| :---------- | :---------- | :--------------------------------------------------------- | :------------------------------------------------------ | :--------------------------------------------- |
| DeepSeek-r1 | Interactive | EAGLE-style decoding with deepseek-ai/deepseek-r1 MTP head | `speculative-num-steps=3`, `speculative-eagle-topk=1.0` | https://huggingface.co/deepseek-ai/DeepSeek-R1 |

> Note: ONLY Sglang backend supports speculative-decoding

## Accuracy Evaluation

**Using MLCFlow Automation**

```
mlcr run,accuracy,mlperf,_dataset_deepseek-r1 --result_dir=<Path to directory where files are generated after the benchmark run>
```

**Using Native method**

Accuracy evaluation is handled uniformly across all backends:

```bash
# within container, with virtualenv activated
(.venv_BACKEND) $ python3 eval_accuracy.py --input-file <input_file>.pkl
```

### Reference Evals

Pytorch reference scores:

```bash
Evaluation Results: {
  "mean-accuracy": 81.3582,
  "mean-output-tok-len": 3886.2274,
  "num-samples": 4388
}
```
