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


def serialize_cli_args(args: Any) -> Dict[str, Any]:
    """Convert parsed CLI args into JSON-serializable primitive types."""
    params: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = value
        else:
            params[key] = str(value)
    return params


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set PyTorch seed for reproducible results.
    
    Based on systematic analysis, torch.manual_seed() is the only component
    needed to ensure deterministic behavior in our retrieval system.
    """
    import torch
    torch.manual_seed(seed)


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


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch is None:
        return "cpu"

    try:
        import habana_frameworks.torch.core as htcore
        if torch.hpu.is_available():
            os.environ["PT_HPU_LAZY_MODE"] = "1"
            return "hpu"
    except ImportError:
            pass
    
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.xpu.is_available():
        return "xpu"
    
    # Default to CPU
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
    """Setup LLM configuration with auto-detection."""
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
    
    return {
        "service_url": args.llm_service_url,
        "model_name": model_name,
        "max_tokens": max_tokens,
        "device": device
    }