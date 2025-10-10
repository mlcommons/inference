#!/usr/bin/env python3
"""
General utilities for document processing, deterministic operations, and other common functions.
"""

import json
from pathlib import Path
from typing import Dict


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
    """Set PyTorch seed for reproducible results.
    
    Based on systematic analysis, torch.manual_seed() is the only component
    needed to ensure deterministic behavior in our retrieval system.
    """
    import torch
    torch.manual_seed(seed)