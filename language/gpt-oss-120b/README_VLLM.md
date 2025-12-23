# vLLM Backend Implementation for gpt-oss-120b

This document describes the vLLM backend implementation for gpt-oss-120b, replicating the structure from deepseek-r1.

## Overview

The vLLM backend provides high-performance inference for the gpt-oss-120b model using vLLM's tensor parallelism and optimized CUDA kernels.

## File Structure

```
gpt-oss-120b/
├── backends/
│   ├── __init__.py              # Updated with VLLMBackend
│   ├── base_backend.py          # Existing base class
│   ├── sglang_backend.py        # Existing SGLang backend
│   └── vllm_backend.py          # NEW: vLLM backend implementation
├── docker/
│   ├── Dockerfile.vllm          # NEW: vLLM Docker image
│   ├── evaluation_requirements.txt  # NEW: Python dependencies
│   ├── setup_scripts/
│   │   ├── common.sh            # NEW: Shared setup functions
│   │   └── setup_vllm.sh        # NEW: vLLM-specific setup
│   └── launch_scripts/
│       └── launch_vllm.sh       # NEW: vLLM Docker launcher
├── setup.sh                     # NEW: Main setup script
├── launch_docker.sh             # NEW: Main Docker launcher
└── run_mlperf.py                # UPDATED: Added vLLM support

## Implementation Details

### 1. Backend Implementation (backends/vllm_backend.py)

The `VLLMBackend` class implements the `BaseBackend` interface with:
- **Initialization**: Creates vLLM LLM instance with tensor parallelism
- **Generation**: Batch inference using `prompt_token_ids`
- **Cleanup**: Proper resource cleanup (CUDA, Ray, distributed)

Key features:
- Tensor parallelism across 8 GPUs
- Prefix caching for efficiency
- Chunked prefill support
- FlashInfer attention backend
- Configurable memory utilization (90%)

### 2. Docker Setup

#### Dockerfile.vllm
- Base: `nvidia/cuda:12.6.0-devel-ubuntu22.04`
- PyTorch: 2.7.0 with CUDA 12.6
- vLLM: 0.9.0
- UV package manager for fast dependency installation

#### Setup Scripts
- **common.sh**: Shared functions for all backends
  - Virtual environment setup
  - MLPerf LoadGen installation
  - Build dependencies

- **setup_vllm.sh**: vLLM-specific setup
  - Creates `.venv_vllm` virtual environment
  - Installs MLPerf LoadGen
  - Installs flashinfer

#### Launch Scripts
- **launch_vllm.sh**: Docker container launcher
  - Builds Docker image
  - Mounts workspace and data directories
  - Sets up HuggingFace cache
  - Configures GPU access

### 3. Main Scripts

#### setup.sh
Detects backend from `MLPERF_BACKEND` environment variable and runs appropriate setup:
```bash
export MLPERF_BACKEND=vllm
./setup.sh
```

#### launch_docker.sh
Launches Docker container for specified backend:
```bash
export MLPERF_BACKEND=vllm
./launch_docker.sh
```

### 4. MLPerf Integration (run_mlperf.py)

Updated to support vLLM backend:
- Added `VLLMBackend` import
- Added `"vllm"` to backend choices
- Backend instantiation with proper configuration

## Usage

### 1. Set Environment Variable
```bash
export MLPERF_BACKEND=vllm
```

### 2. Launch Docker Container
```bash
./launch_docker.sh
```

### 3. Inside Container - Setup
```bash
export MLPERF_BACKEND=vllm
./setup.sh
```

### 4. Run MLPerf Benchmark
```bash
# Offline scenario (accuracy)
python run_mlperf.py \
    --backend vllm \
    --scenario offline \
    --accuracy \
    --input-file data/accuracy_eval_tokenized.pkl

# Offline scenario (performance)
python run_mlperf.py \
    --backend vllm \
    --scenario offline \
    --input-file data/accuracy_eval_tokenized.pkl

# Server scenario
python run_mlperf.py \
    --backend vllm \
    --scenario server \
    --input-file data/accuracy_eval_tokenized.pkl
```

## Configuration

### vLLM Backend Parameters

The vLLM backend uses these default parameters:

```python
VLLMBackend(
    model_name="gpt2-oss/gpt-oss-120b-2t",
    tensor_parallel_size=8,          # 8 GPUs
    max_model_len=4096,              # Max sequence length
    max_num_seqs=64,                 # Batch size
    gpu_memory_utilization=0.90,     # 90% GPU memory
    trust_remote_code=True,
    dtype="auto",                    # Auto-detect (BF16/FP16)
    enforce_eager=False,             # Use CUDA graphs
    enable_prefix_caching=True,      # Enable prefix caching
    enable_chunked_prefill=True,     # Enable chunked prefill
)
```

### Environment Variables

The vLLM backend sets these environment variables:
- `CUDA_MODULE_LOADING=LAZY`
- `NCCL_TREE_THRESHOLD=0`
- `VLLM_ATTENTION_BACKEND=FLASHINFER`
- `VLLM_DISABLE_TQDM=1`
- `VLLM_USE_V1=0`
- `VLLM_ENGINE_ITERATION_TIMEOUT_S=0`

## Comparison with SGLang Backend

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Architecture** | Native Python library | Server-client |
| **Communication** | Direct function calls | HTTP API |
| **Deployment** | Single process | Separate server |
| **Concurrency** | Internal batching | External queuing |
| **Use Case** | Offline inference | Server scenarios |

## Directory Structure Comparison

This implementation mirrors the deepseek-r1 structure:

```
deepseek-r1/                      gpt-oss-120b/
├── backends/                     ├── backends/
│   └── vllm_backend.py          │   └── vllm_backend.py ✓
├── docker/                       ├── docker/
│   ├── Dockerfile.vllm          │   ├── Dockerfile.vllm ✓
│   ├── setup_scripts/           │   ├── setup_scripts/
│   │   ├── common.sh            │   │   ├── common.sh ✓
│   │   └── setup_vllm.sh        │   │   └── setup_vllm.sh ✓
│   └── launch_scripts/          │   └── launch_scripts/
│       └── launch_vllm.sh       │       └── launch_vllm.sh ✓
├── setup.sh                     ├── setup.sh ✓
└── launch_docker.sh             └── launch_docker.sh ✓
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `gpu_memory_utilization` (default: 0.90)
   - Reduce `max_num_seqs` (default: 64)
   - Reduce `max_model_len` (default: 4096)

2. **Model Not Found**
   - Ensure HuggingFace cache is accessible
   - Check `HF_HOME` environment variable
   - Model will auto-download on first run

3. **CUDA Errors**
   - Verify CUDA 12.6+ is installed
   - Check GPU compatibility
   - Ensure sufficient GPU memory (8x GPUs recommended)

4. **Ray Initialization Errors**
   - vLLM uses Ray for multi-GPU coordination
   - Cleanup: `ray stop` between runs
   - Check Ray logs in `/tmp/ray/`

## Performance Tuning

### For Maximum Throughput
```python
max_num_seqs=128           # Larger batch size
gpu_memory_utilization=0.95 # More aggressive memory use
enable_prefix_caching=True  # Cache common prefixes
```

### For Lower Latency
```python
max_num_seqs=32            # Smaller batch size
enforce_eager=True          # Disable CUDA graphs
enable_chunked_prefill=True # Process long prompts in chunks
```

## Next Steps

1. **Performance Benchmarking**: Compare vLLM vs SGLang throughput
2. **Accuracy Validation**: Run accuracy evaluation
3. **Parameter Tuning**: Optimize for your specific hardware
4. **Multi-Node**: Extend to pipeline parallelism across nodes

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [MLPerf Inference](https://github.com/mlcommons/inference)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
