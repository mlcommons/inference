# Mlperf Inference DeepSeek Reference Implementation

## Automated command to run the benchmark via MLFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/language/deepseek-r1/) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do pip install mlc-scripts and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## Model & Dataset Download

> **Model**: [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) (revision: `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad`)

- DeepSeek-R1 model is automatically downloaded as part of setup
- Checkpoint conversion is done transparently when needed.

## Dataset Download

The dataset is an ensemble of the datasets: AIME, MATH500, gpqa, MMLU-Pro, livecodebench(code_generation_lite). They are covered by the following licenses:

- AIME: [CC0](https://creativecommons.org/public-domain/cc0/)
- MATH500: [MIT](https://opensource.org/license/mit), based on [original paper](https://arxiv.org/pdf/2103.03874)
- gpqa: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- MMLU-Pro: [MIT](https://opensource.org/license/mit)
- livecodebench(code_generation_lite): [CC](https://creativecommons.org/share-your-work/cclicenses/)

### Preprocessed

**Using MLCFlow Automation**

```
mlcr get,dataset,whisper,_preprocessed,_mlc,_rclone --outdirname=<path to download> -j
```

**Using Native method**

You can use Rclone to download the preprocessed dataset from a Cloudflare R2 bucket.

To run Rclone on Windows, you can download the executable [here](https://rclone.org/install/#windows).
To install Rclone on Linux/macOS/BSD systems, run:
```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```
Once Rclone is installed, run the following command to authenticate with the bucket:
```
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```
You can then navigate in the terminal to your desired download directory and run the following command to download the dataset:

```
rclone copy mlc-inference:mlcommons-inference-wg-public/deepseek_r1/datasets/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl ./ -P
```

### Calibration

**Using MLCFlow Automation**

```
mlcr get,preprocessed,dataset,deepseek-r1,_calibration,_mlc,_rclone --outdirname=<path to download> -j
```

**Using Native method**

Download and install Rclone as described in the previous section.

Then navigate in the terminal to your desired download directory and run the following command to download the dataset:

```
rclone copy mlc-inference:mlcommons-inference-wg-public/deepseek_r1/datasets/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl ./ -P
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

> ⚠️ **IMPORTANT NOTE**: The PyTorch reference implementation takes approximately 8 days to run on an H200x8 system. This is because large max-OSL (32K) limits concurrency (max-BS=16), and unoptimized pytorch forward and decode logics.

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

| Option         | Description                    | Default          |
| -------------- | ------------------------------ | ---------------- |
| `--mode`       | Scenario mode (offline/server) | `offline`        |
| `--accuracy`   | Run accuracy test              | `False`          |
| `--output-dir` | Output directory for results   | `mlperf_results` |

### Backend Support Matrix

The following table shows which backends support different evaluation and MLPerf operations:

| Backend     | `run_eval.py` | `run_mlperf.py --mode=offline` | `run_mlperf.py --mode=server` |
| ----------- | ------------- | ------------------------------ | ----------------------------- |
| pytorch-fp8 | x             | x                              |                               |
| vllm-fp8    | x             | x                              |                               |
| sglang-fp8  | x             | x                              | x                             |

> **Note**: For PyTorch backend, use the `_mpi` versions with `torchrun`. For vLLM and SGLang backends, use the single-process versions without `_mpi`.

## Accuracy Evaluation

**Using MLCFlow Automation**

```
TBD
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
  "mean-accuracy": 81.67730173199635,
  "mean-output-tok-len": 4043.449863263446,
  "num-samples": 4388
}
```
