"""Central registry for all backend configurations and metadata."""
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

# Configuration constants
MAX_ISL = 3136  # max input sequence length
MAX_OSL = 32 * 1024  # max output sequence length
MAX_TEMPLATE_TOKS = 4  # max template tokens
MODEL_REVISION = "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"

# Backend Registry with all metadata
BACKEND_REGISTRY = {
    'pytorch': {
        'class_path': 'backends.pytorch_backend.PyTorchBackend',
        'input_type': 'tokenized',
        'uses_chat_template': True,
        'supports_async': False,
        'supports_streaming': False,
        'compatible_runners': ['eval_mpi', 'mlperf_mpi'],
        'required_torchrun': True,
        'config': {
            "model_name": "deepseek-ai/DeepSeek-R1",
            "model_revision": MODEL_REVISION,
            "batch_size": 16,
            "max_seq_len": MAX_ISL + MAX_OSL + MAX_TEMPLATE_TOKS,
            "max_new_tokens": MAX_OSL,
            "temperature": 0.0,
            "device": "cuda",
            "dtype": "bfloat16",
            "seed": 965,  # pytorch will use special seed (same as ref-outputs)
            "num_threads": 8,
        },
        'env_vars': {
            # PyTorch reads REF_DSINFER_PATH but doesn't set it
            # Also reads WORLD_SIZE, RANK, LOCAL_RANK from torchrun/mpirun
            'OMP_NUM_THREADS': '8',
        }
    },
    'vllm': {
        'class_path': 'backends.vllm_backend.VLLMBackend',
        'input_type': 'text',
        'uses_chat_template': True,
        'supports_async': True,
        'supports_streaming': False,
        'compatible_runners': ['eval', 'mlperf'],
        'required_torchrun': False,
        'config': {
            "model": "deepseek-ai/DeepSeek-R1",
            "model_revision": MODEL_REVISION,
            "tokenizer": "deepseek-ai/DeepSeek-R1",
            "tensor_parallel_size": 8,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.90,
            "trust_remote_code": True,
            "dtype": "auto",
            "max_input_len": MAX_ISL + MAX_TEMPLATE_TOKS,
            "max_output_len": MAX_OSL,
            "max_model_len": MAX_ISL + MAX_OSL + MAX_TEMPLATE_TOKS,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        'env_vars': {
            'CUDA_MODULE_LOADING': 'LAZY',
            'NCCL_TREE_THRESHOLD': '0',
            'OMP_NUM_THREADS': '8',
            'VLLM_ATTENTION_BACKEND': 'FLASHINFER',
            'VLLM_DISABLE_TQDM': '1',
            'DISABLE_TQDM': '1',
            'VLLM_USE_V1': '0',
            'VLLM_ENGINE_ITERATION_TIMEOUT_S': '0',
            'VLLM_API_TIMEOUT': '0',
            'VLLM_RPC_TIMEOUT': '0',
            'VLLM_WORKER_TIMEOUT': '0',
            'VLLM_ASYNC_ENGINE_TIMEOUT': '0',
        }
    },
    'sglang': {
        'class_path': 'backends.sglang_backend.SGLangBackend',
        'input_type': 'text',
        'uses_chat_template': False,
        'supports_async': True,
        'supports_streaming': True,
        'compatible_runners': ['eval', 'mlperf'],
        'required_torchrun': False,
        'config': {
            "model": "deepseek-ai/DeepSeek-R1",
            "model_revision": MODEL_REVISION,
            "served_model_name": "deepseek-r1",
            "tokenizer": "deepseek-ai/DeepSeek-R1",
            "host": "0.0.0.0",
            "api_key": None,
            "tensor_parallel_size": 8,
            # NOTE(vir): sg-lang crash without +2 additional
            "context_length": MAX_ISL + MAX_OSL + MAX_TEMPLATE_TOKS + 2,
            "max_tokens": MAX_OSL,
            "mem_fraction_static": 0.90,
            "random_seed": 42,
            "dtype": "auto",
            "trust_remote_code": True,
            "enable_torch_compile": True,
            "enable_flashinfer": True,
            "enable_dp_attention": True,
            "dp": 8,
            "cuda_graph_max_bs": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "max_running_requests": 512,  # concurrency
            "request_timeout": None,
            "server_startup_timeout": 1800,
            "health_check_interval": 5,
        },
        'env_vars': {
            'CUDA_MODULE_LOADING': 'LAZY',
            'NCCL_TREE_THRESHOLD': '0',
        }
    }
}


def detect_backend() -> str:
    """
    Detect backend from MLPERF_BACKEND environment variable.

    Returns:
        Backend name from MLPERF_BACKEND

    Raises:
        RuntimeError: If MLPERF_BACKEND is not set or invalid
    """
    if "MLPERF_BACKEND" not in os.environ:
        supported = get_supported_backends()
        raise RuntimeError(
            "MLPERF_BACKEND environment variable is required but not set.\n\n"
            "Please set it to one of the supported backends:\n" +
            "\n".join(
                f"  export MLPERF_BACKEND={backend}" for backend in supported)
        )

    backend = os.environ["MLPERF_BACKEND"]

    # Validate the backend value
    supported = get_supported_backends()
    if backend not in supported:
        raise RuntimeError(
            f"Invalid MLPERF_BACKEND value: {backend}\n\n"
            f"Supported backends: {', '.join(supported)}\n\n"
            "Please set it to one of:\n" +
            "\n".join(f"  export MLPERF_BACKEND={b}" for b in supported)
        )

    return backend


def validate_backend(backend: str) -> None:
    """
    Validate that the backend name is supported.

    Args:
        backend: Backend name to validate

    Raises:
        ValueError: If backend is not supported
    """
    supported_backends = get_supported_backends()
    if backend not in supported_backends:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: {', '.join(supported_backends)}")


def _get_compatibility_error_message(
        backend: str, runner_type: str, compatible: List[str]) -> str:
    """
    Generate error message for incompatible backend/runner combinations.

    Args:
        backend: Detected backend name
        runner_type: Type of runner
        compatible: List of compatible backends for this runner

    Returns:
        Error message string
    """
    if runner_type in ['eval_mpi', 'mlperf_mpi']:
        return (
            f"MPI runners (run_eval_mpi.py, run_mlperf_mpi.py) only support: {', '.join(compatible)}.\n"
            f"Detected backend: {backend}\n\n"
            f"To use {backend} backend, use the non-MPI runners:\n"
            f"  - run_eval.py\n"
            f"  - run_mlperf.py\n\n"
            f"To use MPI runners:\n"
            f"  export MLPERF_BACKEND=pytorch"
        )
    else:
        # For non-MPI runners
        if backend == 'pytorch':
            return (
                f"For PyTorch backend, use the MPI runner instead:\n"
                f"  torchrun --nproc_per_node=8 run_eval_mpi.py [args]\n\n"
                f"{runner_type} runner only supports: {', '.join(compatible)}"
            )
        else:
            return (
                f"{runner_type} runner does not support {backend} backend.\n"
                f"Compatible backends: {', '.join(compatible)}"
            )


def validate_runner_for_backend(runner_type: str) -> str:
    """
    Validate that the runner is compatible with the detected backend.

    Args:
        runner_type: One of 'eval', 'mlperf', 'eval_mpi', 'mlperf_mpi'

    Returns:
        Backend name from MLPERF_BACKEND env var (if valid)

    Raises:
        RuntimeError: If MLPERF_BACKEND not set or runner incompatible with backend
    """
    backend = detect_backend()

    # Check compatibility using registry
    if not is_backend_compatible_with_runner(backend, runner_type):
        supported = get_supported_backends()
        compatible = [
            b for b in supported if is_backend_compatible_with_runner(b, runner_type)]
        error_msg = _get_compatibility_error_message(
            backend, runner_type, compatible)
        raise RuntimeError(error_msg)

    return backend


def supports_streaming(backend_name: Optional[str] = None) -> bool:
    """
    Check if backend supports streaming generation.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        True if backend supports streaming
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")

    return BACKEND_REGISTRY[backend_name].get('supports_streaming', False)


def supports_async(backend_name: Optional[str] = None) -> bool:
    """
    Check if backend supports async generation.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        True if backend supports async
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")

    return BACKEND_REGISTRY[backend_name].get('supports_async', False)


def requires_torchrun(backend_name: Optional[str] = None) -> bool:
    """
    Check if backend requires torchrun for execution (supports MPI/distributed execution).

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        True if backend requires torchrun
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")

    return BACKEND_REGISTRY[backend_name].get('required_torchrun', False)


def get_backend_config(backend_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific backend.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        Backend configuration dictionary
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[backend_name]['config'].copy()


def get_backend_class_path(backend_name: Optional[str] = None) -> str:
    """Get the class path for a backend.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        Backend class path string
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[backend_name]['class_path']


def uses_text_input(backend_name: Optional[str] = None) -> bool:
    """Check if a backend uses text input (vs tokenized).

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        True if backend uses text input, False if tokenized
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[backend_name]['input_type'] == 'text'


def uses_chat_template(backend_name: Optional[str] = None) -> bool:
    """Check if a backend uses chat templates.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        True if backend uses chat templates
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[backend_name]['uses_chat_template']


def get_supported_backends() -> List[str]:
    """Get list of all supported backends."""
    return list(BACKEND_REGISTRY.keys())


def get_backend_instance(backend_name: Optional[str] = None):
    """Create and return a backend instance.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        Backend instance
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")

    class_path = get_backend_class_path(backend_name)
    module_path, class_name = class_path.rsplit('.', 1)

    # Import the module and get the class
    import importlib
    module = importlib.import_module(module_path)
    backend_class = getattr(module, class_name)

    return backend_class()


def is_backend_compatible_with_runner(
        backend_name: Optional[str] = None, runner_type: str = None) -> bool:
    """Check if a backend is compatible with a specific runner type.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
        runner_type: Type of runner to check compatibility with

    Returns:
        True if backend is compatible with the runner type
    """
    if runner_type is None:
        raise ValueError("runner_type is required")

    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        return False
    return runner_type in BACKEND_REGISTRY[backend_name]['compatible_runners']


def get_backend_env_vars(backend_name: Optional[str] = None) -> Dict[str, str]:
    """Get environment variables for a backend.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        Dictionary of environment variables
    """
    if backend_name is None:
        backend_name = detect_backend()

    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}")

    # Get static env vars
    env_vars = BACKEND_REGISTRY[backend_name]['env_vars'].copy()

    # Handle dynamic env vars (e.g., OMP_NUM_THREADS based on
    # tensor_parallel_size)
    if backend_name == 'vllm':
        config = get_backend_config(backend_name)
        env_vars['OMP_NUM_THREADS'] = str(
            config.get('tensor_parallel_size', 8))

    return env_vars


def apply_backend_env_vars(backend_name: Optional[str] = None) -> None:
    """Apply environment variables for a backend.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
    """
    if backend_name is None:
        backend_name = detect_backend()

    env_vars = get_backend_env_vars(backend_name)
    for key, value in env_vars.items():
        os.environ[key] = value
