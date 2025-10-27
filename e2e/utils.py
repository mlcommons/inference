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