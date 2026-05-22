#!/usr/bin/env python3
"""
General utilities for document processing, deterministic operations, and other common functions.
"""

import json
import os
import requests
import torch
from pathlib import Path
from typing import Dict, Optional, Union, Any



def load_url_mapping(directory: str) -> Dict[str, str]:
    """Load URL mapping from url_mapping.json in specified directory."""
    mapping_path = Path(directory) / "url_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_base_filename(filename: str) -> str:
    """Extract base filename without extension."""
    if '.' in filename:
        return '.'.join(filename.split('.')[:-1])
    return filename


def save_url_mapping(directory: str, url_mapping: Dict[str, str]) -> None:
    """Save URL mapping to url_mapping.json in specified directory."""
    mapping_path = Path(directory) / "url_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(url_mapping, f, indent=2, ensure_ascii=False)


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set seeds for reproducible results across all components.

    Covers: Python random, NumPy, PyTorch (CPU + all CUDA/XPU devices).
    Note: LLM responses are stochastic and cannot be made deterministic via seed.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_dataset_by_difficulty(df, difficulty: int = 0):
    """
    Filter dataset by minimum number of answer links (difficulty level).
    
    Args:
        df: pandas DataFrame with dataset
        difficulty: Minimum number of answer links required (0 = no filtering)
        
    Returns:
        Filtered DataFrame with queries having >= difficulty answer links
    """
    if difficulty <= 0:
        return df
    
    # Count answer links for each row
    link_counts = df.apply(
        lambda row: sum(1 for col in df.columns 
                       if col.startswith('wikipedia_link_') and row.notna()[col]), 
        axis=1
    )
    
    filtered_df = df[link_counts >= difficulty].reset_index(drop=True)
    print(f"Filtered dataset by difficulty >= {difficulty}: {len(filtered_df)} queries remaining (from {len(df)} total)")
    
    return filtered_df


def _parse_cpulist(s: str) -> list:
    """Parse a Linux cpulist string ("0-3,7,9-11") into a list of ints."""
    result = []
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def _physical_cores_for_node(node: int) -> list:
    """Return one logical CPU per physical core on the given NUMA node.

    Reads /sys/devices/system/node/nodeN/cpulist for the node's CPUs, then
    filters HT siblings via /sys/devices/system/cpu/cpuN/topology/thread_siblings_list.
    """
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            node_cpus = set(_parse_cpulist(f.read()))
    except OSError:
        return []

    seen_cores = set()
    primary = []
    for cpu in sorted(node_cpus):
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                siblings = _parse_cpulist(f.read())
        except OSError:
            primary.append(cpu)
            continue
        core_key = min(siblings)
        if core_key in seen_cores:
            continue
        seen_cores.add(core_key)
        primary.append(cpu)
    return primary


def apply_numa_pinning() -> None:
    """Pin CPU affinity to one NUMA node's physical cores.

    Honors E2E_DISABLE_NUMA / E2E_NUMA_NODE / E2E_NUMA_CORES.
    Memory locality relies on Linux first-touch (good enough for our small
    Python working set). For strict membind, invoke under `numactl --membind=N`.
    """
    if os.environ.get("E2E_DISABLE_NUMA") == "1":
        print("  NUMA pinning disabled via E2E_DISABLE_NUMA=1")
        return

    if not hasattr(os, "sched_setaffinity"):
        print("  NUMA pinning unavailable (os.sched_setaffinity missing)")
        return

    cores_env = os.environ.get("E2E_NUMA_CORES")
    if cores_env:
        try:
            cores = _parse_cpulist(cores_env)
        except ValueError:
            print(f"  Invalid E2E_NUMA_CORES={cores_env!r}; skipping pinning")
            return
    else:
        node = int(os.environ.get("E2E_NUMA_NODE", "0"))
        cores = _physical_cores_for_node(node)
        if not cores:
            print(f"  NUMA node {node} cpulist not readable; skipping pinning")
            return

    try:
        os.sched_setaffinity(0, set(cores))
    except OSError as e:
        print(f"  sched_setaffinity failed: {e}; skipping pinning")
        return

    print(f"  Pinned CPU affinity to {len(cores)} cores: {cores[0]}..{cores[-1]}")


def apply_cpu_threading_env() -> None:
    """Pin CPU affinity to a NUMA node and configure OpenMP env vars.

    Only call this when a model is actually being placed on CPU.
    Never overwrites a user-set env var.
    """
    apply_numa_pinning()

    if "OMP_NUM_THREADS" not in os.environ:
        override = os.environ.get("E2E_OMP_NUM_THREADS")
        if override:
            n_threads = override
        else:
            try:
                # sched_getaffinity reflects the pinning we just applied.
                n_threads = str(len(os.sched_getaffinity(0)))
            except (AttributeError, OSError):
                n_threads = str(os.cpu_count() or 1)
        os.environ["OMP_NUM_THREADS"] = n_threads
        print(f"  Set OMP_NUM_THREADS={n_threads}")

    vendor = detect_cpu_vendor()
    if vendor == "intel" and "KMP_AFFINITY" not in os.environ:
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        print("  Set KMP_AFFINITY (Intel OpenMP)")
    elif vendor != "intel":
        print(f"  Skipping KMP_AFFINITY (CPU vendor={vendor})")


def detect_cpu_vendor() -> str:
    """Detect host CPU vendor from /proc/cpuinfo.

    Returns "intel", "amd", or "unknown". Used to gate vendor-specific tuning
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("vendor_id"):
                    vendor = line.split(":", 1)[1].strip()
                    if vendor == "GenuineIntel":
                        return "intel"
                    if vendor == "AuthenticAMD":
                        return "amd"
                    return "unknown"
    except OSError:
        pass
    return "unknown"


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            print(f"Using AMD ROCm GPU (torch.version.hip={torch.version.hip})")
        else:
            print("Using NVIDIA CUDA GPU")
        return "cuda"

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Using Intel XPU GPU")
        return "xpu"

    try:
        import habana_frameworks.torch.core as htcore  # noqa: F401
        if torch.hpu.is_available():
            os.environ["PT_HPU_LAZY_MODE"] = "1"
            print("Using Habana HPU")
            return "hpu"
    except ImportError:
        pass

    print("Using CPU")
    return "cpu"


def get_model_info_from_service(service_url: str) -> Optional[Dict]:
    """Get model information from LLM service."""
    try:
        # Try OpenAI-compatible API first
        models_response = requests.get(f"{service_url.rstrip('/v1/chat/completions').rstrip('/v1')}/v1/models", timeout=10)
        if models_response.status_code == 200:
            models_data = models_response.json()
            if "data" in models_data and len(models_data["data"]) > 0:
                return models_data["data"][0]
        
        # Try alternative endpoints
        base_url = service_url.rstrip('/v1/chat/completions').rstrip('/v1')
        for endpoint in ["/models", "/info", "/v1/model"]:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
                
    except Exception as e:
        print(f"Warning: Could not auto-detect model from {service_url}: {e}")
    
    return None


def get_model_name_from_service(service_url: str) -> str:
    """Auto-detect model name from LLM service."""
    model_info = get_model_info_from_service(service_url)
    
    if model_info:
        # Try different possible fields for model name
        for field in ["id", "model", "name", "model_name"]:
            if field in model_info:
                return model_info[field]
    
    # Default fallback
    return "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"


def get_max_tokens_from_service(service_url: str) -> int:
    """Auto-detect max tokens from LLM service."""
    model_info = get_model_info_from_service(service_url)
    
    if model_info:
        # Try different possible fields for max tokens
        for field in ["max_tokens", "max_length", "context_length", "max_context_length"]:
            if field in model_info and isinstance(model_info[field], int):
                return model_info[field]
    
    # Default fallback based on common models
    return 10240


def resolve_config_value(value: Union[str, int], auto_func, *args) -> Union[str, int]:
    """Resolve configuration value that might be 'auto'."""
    if value == "auto":
        return auto_func(*args)
    return value


def get_device_config():
    """Get comprehensive device configuration."""
    config = {
        "device_type": detect_device(),
        "device_count": 1,
        "device_memory": None
    }
    
    if torch is None:
        return config
    
    if config["device_type"] == "hpu":
        config["device_count"] = torch.hpu.device_count()
    
    elif config["device_type"] == "cuda":
        config["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            config["device_memory"] = torch.cuda.get_device_properties(0).total_memory
    
    elif config["device_type"] == "xpu":
        config["device_count"] = torch.xpu.device_count()
    
    return config


def setup_llm_config(args):
    """Setup LLM configuration with auto-detection and OpenRouter support."""
    # Resolve device
    device = resolve_config_value(args.device, detect_device)

    # Resolve model name
    model_name = resolve_config_value(
        args.llm_model,
        get_model_name_from_service,
        args.llm_service_url
    )

    # Resolve max tokens
    if isinstance(args.max_tokens, str):
        max_tokens = resolve_config_value(
            args.max_tokens,
            get_max_tokens_from_service,
            args.llm_service_url
        )
    else:
        max_tokens = args.max_tokens

    # Resolve query model name (for generate_search_queries); falls back to model_name if not set
    query_model_arg = getattr(args, 'query_model', None)
    query_model_name = query_model_arg if query_model_arg else model_name

    # OpenRouter URLs and model names for different components
    # If no API key is set, use the local LLM service URL for all components
    openrouter_key = os.environ.get('OPENROUTER_API_KEY', '')
    if openrouter_key:
        grader_service_url = "https://openrouter.ai/api/v1/chat/completions"
        grader_model_name = "openai/gpt-oss-20b"
        query_service_url = "https://openrouter.ai/api/v1/chat/completions"
        query_model_name = "openai/gpt-oss-120b"
        sufficiency_service_url = "https://openrouter.ai/api/v1/chat/completions"
        sufficiency_model_name = "openai/gpt-oss-120b"
    else:
        grader_service_url = args.llm_service_url
        grader_model_name = model_name
        query_service_url = args.llm_service_url
        query_model_name = model_name
        sufficiency_service_url = args.llm_service_url
        sufficiency_model_name = model_name

    return {
        "service_url": args.llm_service_url,
        "model_name": model_name,
        "query_model_name": query_model_name,
        "max_tokens": max_tokens,
        "device": device,
        # OpenRouter-specific endpoints
        "grader_service_url": grader_service_url,
        "grader_model_name": grader_model_name,
        "query_service_url": query_service_url,
        "sufficiency_service_url": sufficiency_service_url,
        "sufficiency_model_name": sufficiency_model_name,
    }
