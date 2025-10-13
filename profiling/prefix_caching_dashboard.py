#!/usr/bin/env python3
"""
Prefix Efficiency Dashboard
---------------------------
Analyze dataset prefix caching potential with multiple metrics and optional
integration with vLLM Prometheus metrics.

Metrics computed (static, dataset-based):
- PRR (Prefix Reuse Ratio) for a chosen prefix length
- PDI (Prefix Diversity Index) curve across multiple prefix lengths
- PE  (Prefix Entropy) for a chosen prefix length
- PUS (Prefix Uniqueness Score): mean (#others sharing the same prefix)
- POL (Prefix Overlap Length) histogram via randomized pair sampling

Optional runtime correlation (engine-based):
- vLLM Prometheus scrape to compute actual runtime Prefix Cache Hit Rate (PCR):
    PCR = prefill_tokens_reused_total / prefill_tokens_cached_total

Inputs supported:
- Hugging Face datasets (via --dataset and optional --field)
- Local files: .txt, .jsonl, .json, .pickle/.pkl (via --file and optional --field)

Outputs:
- Printed metrics summary
- Plots (PNG) if --plots is set:
    * pdi_curve.png
    * prefix_overlap_histogram.png
    * metric_summary.txt (human-readable report)

Usage examples:
    # Text data (requires tokenizer)
    python3 prefix_efficiency_dashboard.py \
        --dataset tatsu-lab/alpaca --field instruction \
        --tokenizer meta-llama/Meta-Llama-3-8B \
        --prefix-len 128 --sample-size 4000 --plots

    # Pre-tokenized data (no tokenizer needed)
    python3 prefix_efficiency_dashboard.py \
        --file tokenized_data.pkl --token-column token_ids \
        --prefix-len 128 --sample-size 4000 --plots

    # Auto-detect pre-tokenized data
    python3 prefix_efficiency_dashboard.py \
        --file tokenized_data.pkl --inspect

    # Export specific fields to text file
    python3 prefix_efficiency_dashboard.py \
        --file data.pkl --export-fields prompt,response --export-file prompts.txt

    # Export to JSON format
    python3 prefix_efficiency_dashboard.py \
        --file data.pkl --export-fields prompt,response --export-format json --export-file data.json

Author: ChatGPT
"""

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
import math
import re
import warnings
from collections import Counter, defaultdict
from typing import List, Sequence, Dict, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress NumPy deprecation warnings when loading pickle files
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import requests
except Exception:
    requests = None

from transformers import AutoTokenizer


# -------------------------------
# Pre-tokenized Data Detection
# -------------------------------
def detect_tokenized_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect if DataFrame contains pre-tokenized data.
    Returns dict with detection results and token column info.
    """
    token_columns = []
    token_info = {}
    
    # Common names for token columns
    token_column_names = [
        'token_ids', 'tokens', 'input_ids', 'token_ids_list', 
        'encoded_tokens', 'tokenized', 'ids'
    ]
    
    for col in df.columns:
        # Check by column name
        if any(token_name in col.lower() for token_name in token_column_names):
            token_columns.append(col)
            continue
            
        # Check by data type and content
        sample_values = df[col].dropna().head(5)
        if len(sample_values) > 0:
            first_val = sample_values.iloc[0]
            
            # Check if it's a list of integers (token IDs)
            if isinstance(first_val, (list, tuple)):
                if len(first_val) > 0 and isinstance(first_val[0], int):
                    token_columns.append(col)
                    token_info[col] = {
                        'type': 'list_of_ints',
                        'sample_length': len(first_val),
                        'sample_tokens': first_val[:10]  # First 10 tokens
                    }
                    continue
            
            # Check if it's a numpy array of integers
            elif hasattr(first_val, 'shape') and hasattr(first_val, 'dtype'):
                if 'int' in str(first_val.dtype) and len(first_val.shape) == 1:
                    token_columns.append(col)
                    token_info[col] = {
                        'type': 'numpy_array',
                        'sample_length': len(first_val),
                        'sample_tokens': first_val[:10].tolist()
                    }
                    continue
    
    return {
        'is_tokenized': len(token_columns) > 0,
        'token_columns': token_columns,
        'token_info': token_info,
        'primary_token_column': token_columns[0] if token_columns else None
    }


# -------------------------------
# Data Loading with Pandas
# -------------------------------
def load_dataframe(args) -> pd.DataFrame:
    """
    Load data from various sources into a pandas DataFrame.
    Returns a DataFrame with a 'text' column containing the text data.
    Also detects if data is pre-tokenized and preserves token information.
    """
    if args.dataset:
        if load_dataset is None:
            raise RuntimeError("datasets library is not installed, cannot use --dataset")
        ds = load_dataset(args.dataset, split=args.split)
        field = args.field or "text"
        if isinstance(ds, dict):
            ds = ds[args.split]
        
        # Convert to list of dicts first
        data_list = []
        for row in ds:
            if field in row and row[field]:
                data_list.append({field: str(row[field])})
        
        df = pd.DataFrame(data_list)
        if field not in df.columns:
            df['text'] = df.iloc[:, 0] if len(df.columns) > 0 else ""
        else:
            df['text'] = df[field]
            
    elif args.file:
        path = args.file
        field = args.field
        
        if path.endswith(".pickle") or path.endswith(".pkl"):
            # Load pickle file
            with open(path, "rb") as f:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    data = pickle.load(f)
            
            if isinstance(data, pd.DataFrame):
                # If the pickle file contains a DataFrame directly, use it
                df = data
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame({"value": [data]})
                
        elif path.endswith(".jsonl"):
            # Load JSONL file
            data_list = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        data_list.append(obj)
                    except json.JSONDecodeError:
                        data_list.append({"raw_line": line})
            df = pd.DataFrame(data_list)
            
        elif path.endswith(".json"):
            # Load JSON file
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame({"value": [data]})
                
        else:
            # Load text file
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({"text": lines})
        
        # Handle field selection and text column creation
        if field is not None and field in df.columns:
            df['text'] = df[field].astype(str)
        else:
            # Try common field names, prioritizing those with content
            common_fields = ['text', 'prompt', 'instruction', 'content', 'input']
            text_field = None
            for cf in common_fields:
                if cf in df.columns:
                    # Check if this field has non-empty content
                    non_empty_count = df[cf].astype(str).str.strip().ne('').sum()
                    if non_empty_count > 0:
                        text_field = cf
                        break
            
            if text_field:
                df['text'] = df[text_field].astype(str)
            elif len(df.columns) > 0:
                # Use first column as text
                df['text'] = df.iloc[:, 0].astype(str)
            else:
                df['text'] = ""
    else:
        raise ValueError("Specify either --dataset or --file")
    
    # Store original row count before filtering
    original_row_count = len(df)
    
    # Clean and filter the data
    df = df.dropna(subset=['text'])  # Remove rows with NaN text
    df = df[df['text'].str.strip() != '']  # Remove empty strings
    
    # Shuffle for unbiased sampling
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Apply sample size limit
    if args.sample_size and args.sample_size > 0:
        df = df.head(args.sample_size)
    
    # Store metadata in DataFrame attributes
    df.attrs['original_row_count'] = original_row_count
    df.attrs['sample_size'] = args.sample_size
    df.attrs['is_sampled'] = args.sample_size and args.sample_size > 0 and args.sample_size < original_row_count
    
    return df


# -------------------------------
# Data Export
# -------------------------------
def export_fields(df: pd.DataFrame, fields: List[str], output_file: str, export_format: str = "txt") -> None:
    """
    Export specified fields from DataFrame to a file.
    
    Args:
        df: DataFrame containing the data
        fields: List of field names to export
        output_file: Output file path
        export_format: Export format ('txt', 'json', 'csv')
    """
    # Validate fields exist in DataFrame
    available_fields = list(df.columns)
    missing_fields = [field for field in fields if field not in available_fields]
    
    if missing_fields:
        print(f"Warning: Fields not found in data: {missing_fields}")
        print(f"Available fields: {available_fields}")
        fields = [field for field in fields if field in available_fields]
    
    if not fields:
        print("Error: No valid fields to export")
        return
    
    # Create export DataFrame with only specified fields
    export_df = df[fields].copy()
    
    print(f"Exporting {len(export_df)} rows with fields: {fields}")
    print(f"Output file: {output_file}")
    
    if export_format == "txt":
        # Export as text file, one field per line for each row
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in export_df.iterrows():
                f.write(f"Row {idx + 1}:\n")
                for field in fields:
                    value = row[field]
                    # Handle different data types
                    if pd.isna(value):
                        f.write(f"  {field}: <null>\n")
                    elif isinstance(value, (list, tuple)):
                        f.write(f"  {field}: {value}\n")
                    elif hasattr(value, 'shape'):  # NumPy array
                        f.write(f"  {field}: {value.tolist()}\n")
                    else:
                        f.write(f"  {field}: {value}\n")
                f.write("\n")
    
    elif export_format == "json":
        # Export as JSON
        export_data = export_df.to_dict('records')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    elif export_format == "csv":
        # Export as CSV
        export_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Exported {len(export_df)} rows to {output_file}")


# -------------------------------
# Data Inspection
# -------------------------------
def inspect_dataframe(df: pd.DataFrame, file_path: str, field: str = None, max_rows: int = 5) -> Dict[str, Any]:
    """
    Simple inspection of pandas DataFrame - just column info and sample data.
    Returns a dictionary with basic file info, columns, and sample rows.
    """
    # Check if this is a nested DataFrame (single row containing a DataFrame)
    print(f"Initial DataFrame shape: {df.shape}")
    print(f"Initial DataFrame columns: {list(df.columns)}")
    
    if len(df) == 1 and len(df.columns) == 1:
        first_value = df.iloc[0, 0]
        print(f"First value type: {type(first_value)}")
        if isinstance(first_value, pd.DataFrame):
            print(f"Detected nested DataFrame. Extracting inner DataFrame...")
            print(f"Inner DataFrame shape: {first_value.shape}")
            print(f"Inner DataFrame columns: {list(first_value.columns)}")
            df = first_value
    
    # Get original row count and sampling info
    original_row_count = df.attrs.get('original_row_count', len(df))
    is_sampled = df.attrs.get('is_sampled', False)
    sample_size = df.attrs.get('sample_size', None)
    
    # Detect pre-tokenized data
    token_detection = detect_tokenized_data(df)
    
    file_info = {
        "file_path": file_path,
        "file_type": "dataframe",
        "columns": list(df.columns),
        "column_types": {},
        "sample_rows": [],
        "total_rows": len(df),
        "original_total_rows": original_row_count,
        "is_sampled": is_sampled,
        "sample_size": sample_size,
        "field_used": field,
        "memory_usage": df.memory_usage(deep=True).sum(),
        "token_detection": token_detection
    }
    
    # Get column types - simplified
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = df[col].count()
        null_count = len(df) - non_null_count
        
        # Simple type description
        if dtype.startswith('object'):
            # Check if it's actually strings, lists, dicts, etc.
            try:
                sample_values = df[col].dropna().head(3)
                if len(sample_values) > 0:
                    sample_types = [type(val).__name__ for val in sample_values]
                    if len(set(sample_types)) == 1:
                        dtype = f"object ({sample_types[0]})"
                    else:
                        dtype = f"object (mixed: {', '.join(set(sample_types))})"
            except (ValueError, TypeError):
                # Handle cases where dropna() fails with complex objects
                dtype = "object (complex)"
        
        file_info["column_types"][col] = f"{dtype} (non-null: {non_null_count}, null: {null_count})"
    
    # Get sample rows - simplified
    sample_df = df.head(max_rows)
    for idx, row in sample_df.iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            # Handle different data types for display
            try:
                if pd.isna(value):
                    row_dict[col] = "<null>"
                elif isinstance(value, (list, tuple, dict)):
                    row_dict[col] = f"<{type(value).__name__} len={len(value)}>"
                elif hasattr(value, 'shape'):  # NumPy array
                    row_dict[col] = f"<numpy.ndarray{value.shape} dtype={value.dtype}>"
                elif isinstance(value, str) and len(value) > 100:
                    row_dict[col] = value[:97] + "..."
                else:
                    row_dict[col] = value
            except (ValueError, TypeError):
                # Handle cases where pd.isna() fails (e.g., with DataFrames)
                row_dict[col] = f"<{type(value).__name__}>"
        file_info["sample_rows"].append(row_dict)
    
    return file_info


def print_file_inspection(file_info: Dict[str, Any]):
    """Print formatted file inspection results."""
    print("\n" + "="*60)
    print("FILE INSPECTION REPORT")
    print("="*60)
    print(f"File: {file_info['file_path']}")
    print(f"Type: {file_info['file_type']}")
    
    # Show row count information
    if file_info.get('is_sampled', False):
        print(f"Total rows in dataset: {file_info['original_total_rows']}")
        print(f"Sample size: {file_info['sample_size']}")
        print(f"Rows analyzed: {file_info['total_rows']}")
    else:
        print(f"Total rows: {file_info['total_rows']}")
    
    if 'error' in file_info:
        print(f"Error: {file_info['error']}")
        return
    
    # Show memory usage if available
    if 'memory_usage' in file_info:
        memory_mb = file_info['memory_usage'] / (1024 * 1024)
        print(f"Memory usage: {memory_mb:.2f} MB")
    
    # Show pre-tokenized data detection
    token_detection = file_info.get('token_detection', {})
    if token_detection.get('is_tokenized', False):
        print(f"\n🔍 Pre-tokenized data detected!")
        print(f"Token columns: {', '.join(token_detection['token_columns'])}")
        primary_col = token_detection['primary_token_column']
        if primary_col and primary_col in token_detection['token_info']:
            token_info = token_detection['token_info'][primary_col]
            print(f"Primary token column: {primary_col}")
            print(f"Token type: {token_info['type']}")
            print(f"Sample token length: {token_info['sample_length']}")
            print(f"Sample tokens: {token_info['sample_tokens']}")
    else:
        print(f"\n📝 Text data detected (will need tokenization)")
    
    print(f"\nColumns ({len(file_info['columns'])}):")
    for col in file_info['columns']:
        col_type = file_info['column_types'].get(col, 'unknown')
        print(f"  - {col}: {col_type}")
    
    if file_info['field_used']:
        print(f"\nField filter: {file_info['field_used']}")
    
    print(f"\nSample rows (first {len(file_info['sample_rows'])}):")
    for i, row in enumerate(file_info['sample_rows']):
        print(f"\nRow {i+1}:")
        if isinstance(row, dict):
            for key, value in row.items():
                # Truncate long values for display
                display_value = str(value)
                if len(display_value) > 100:
                    display_value = display_value[:97] + "..."
                print(f"  {key}: {display_value}")
        else:
            display_value = str(row)
            if len(display_value) > 100:
                display_value = display_value[:97] + "..."
            print(f"  value: {display_value}")
    
    print("="*60)


# -------------------------------
# Data Loading
# -------------------------------
def load_texts(args) -> List[str]:
    texts: List[str] = []
    if args.dataset:
        if load_dataset is None:
            raise RuntimeError("datasets library is not installed, cannot use --dataset")
        ds = load_dataset(args.dataset, split=args.split)
        field = args.field or "text"
        if isinstance(ds, dict):
            ds = ds[args.split]
        for row in ds:
            if field in row and row[field]:
                texts.append(str(row[field]))
    elif args.file:
        path = args.file
        field = args.field
        
        if path.endswith(".pickle") or path.endswith(".pkl"):
            # Handle pickle files
            with open(path, "rb") as f:
                # Suppress NumPy deprecation warnings during pickle loading
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    data = pickle.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if field is None:
                            # try common keys
                            val = item.get("text") or item.get("prompt") or item.get("input") or ""
                        else:
                            val = item.get(field, "")
                        texts.append(str(val))
                    else:
                        texts.append(str(item))
            elif isinstance(data, dict):
                if field is None:
                    # try common keys
                    val = data.get("text") or data.get("prompt") or data.get("input") or ""
                else:
                    val = data.get(field, "")
                texts.append(str(val))
            else:
                texts.append(str(data))
        else:
            # Handle text-based files (json, jsonl, txt)
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith(".jsonl"):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if field is None:
                                # if no field, try common keys else raw line
                                val = obj.get("text") or obj.get("prompt") or obj.get("input") or ""
                            else:
                                val = obj.get(field, "")
                            texts.append(str(val))
                        except Exception:
                            # if not valid json, fall back to raw line
                            if field is None:
                                texts.append(line)
                elif path.endswith(".json"):
                    obj = json.load(f)
                    if isinstance(obj, list):
                        if field is None:
                            # try common keys
                            for item in obj:
                                if isinstance(item, dict):
                                    val = item.get("text") or item.get("prompt") or item.get("input") or ""
                                    texts.append(str(val))
                                else:
                                    texts.append(str(item))
                        else:
                            for item in obj:
                                if isinstance(item, dict):
                                    texts.append(str(item.get(field, "")))
                                else:
                                    texts.append("")
                    else:
                        texts.append(str(obj))
                else:
                    # .txt or other
                    for line in f:
                        line = line.strip()
                        if line:
                            texts.append(line)
    else:
        raise ValueError("Specify either --dataset or --file")
    # shuffle for unbiased sampling
    random.shuffle(texts)
    if args.sample_size and args.sample_size > 0:
        texts = texts[: args.sample_size]
    return [t for t in texts if t is not None and t != ""]


# -------------------------------
# Tokenization helpers
# -------------------------------
def tokenize_dataframe(df: pd.DataFrame, tokenizer, prefix_len: int, token_column: str = None) -> pd.DataFrame:
    """
    Tokenize text column in DataFrame and add token_ids column.
    If pre-tokenized data is detected, use it instead of tokenizing.
    Returns DataFrame with 'token_ids' column.
    """
    # Check if data is already tokenized
    if token_column and token_column in df.columns:
        print(f"Using specified token column: {token_column}")
        primary_token_col = token_column
    else:
        token_detection = detect_tokenized_data(df)
        if token_detection['is_tokenized']:
            primary_token_col = token_detection['primary_token_column']
            print(f"Using auto-detected pre-tokenized data from column: {primary_token_col}")
        else:
            primary_token_col = None
    
    if primary_token_col:
        
        df = df.copy()
        # Truncate pre-tokenized sequences to prefix_len
        df['token_ids'] = df[primary_token_col].apply(
            lambda tokens: tokens[:prefix_len] if isinstance(tokens, (list, tuple)) 
            else tokens[:prefix_len].tolist() if hasattr(tokens, '__getitem__')
            else tokens
        )
        return df
    else:
        if tokenizer is None:
            raise ValueError("Tokenizer is required for text data but was not provided")
        
        print(f"Tokenizing {len(df)} texts...")
        token_ids = []
        for text in tqdm(df['text'], desc="Tokenizing"):
            toks = tokenizer(text).input_ids[:prefix_len]
            token_ids.append(toks)
        
        df = df.copy()
        df['token_ids'] = token_ids
        return df


def hash_prefix(token_ids: Sequence[int], hash_name: str = "md5") -> str:
    # Convert token IDs to bytes properly - handle large integers
    b = bytes(str(token_ids), 'utf-8')
    if hash_name == "md5":
        return hashlib.md5(b).hexdigest()
    elif hash_name == "sha1":
        return hashlib.sha1(b).hexdigest()
    elif hash_name == "sha256":
        return hashlib.sha256(b).hexdigest()
    else:
        # fallback to python hash but stabilized via sha256 of repr
        return hashlib.sha256(repr(token_ids).encode("utf-8")).hexdigest()


# -------------------------------
# Metrics
# -------------------------------
def compute_prr(prefix_hashes: pd.Series) -> float:
    """Compute Prefix Reuse Ratio from pandas Series of hashes."""
    total = len(prefix_hashes)
    unique = prefix_hashes.nunique()
    reused = total - unique
    return reused / total if total > 0 else 0.0


def compute_pus(prefix_hashes: pd.Series) -> float:
    """Compute Prefix Uniqueness Score from pandas Series of hashes."""
    # mean #others sharing same prefix
    value_counts = prefix_hashes.value_counts()
    n = len(prefix_hashes)
    if n == 0:
        return 0.0
    # For each item, count (freq-1), average across all items
    total_others = sum((freq - 1) * freq for freq in list(value_counts.values)) / max(1, n)
    return total_others


def compute_entropy(prefix_hashes: pd.Series) -> float:
    """Compute entropy from pandas Series of hashes."""
    value_counts = prefix_hashes.value_counts()
    n = sum(list(value_counts.values))
    if n == 0:
        return 0.0
    H = 0.0
    for c in list(value_counts.values):
        p = c / n
        H -= p * math.log(p + 1e-12, 2)
    return H


def compute_pdi_curve(df: pd.DataFrame, lengths: List[int], hash_name: str) -> Dict[int, float]:
    """
    Compute PDI curve from DataFrame with token_ids column.
    Return dict: prefix_len -> PDI (unique_prefixes / total)
    """
    result = {}
    for L in tqdm(lengths, desc="Computing PDI"):
        # Apply hash function to truncated token sequences
        hashes = df['token_ids'].apply(lambda ids: hash_prefix(ids[:L], hash_name=hash_name))
        total = len(hashes)
        unique = hashes.nunique()
        result[L] = (unique / total) if total > 0 else 0.0
    return result


def sample_prefix_overlap_lengths(df: pd.DataFrame, num_pairs: int) -> np.ndarray:
    """Sample prefix overlap lengths from DataFrame with token_ids column."""
    n = len(df)
    if n < 2 or num_pairs <= 0:
        return np.array([])
    
    overlaps = []
    for _ in tqdm(range(min(num_pairs, n * (n - 1) // 2)), desc="Sampling overlaps"):
        i, j = random.sample(range(n), 2)
        a, b = df.iloc[i]['token_ids'], df.iloc[j]['token_ids']
        k = 0
        m = min(len(a), len(b))
        while k < m and a[k] == b[k]:
            k += 1
        overlaps.append(k)
    return np.array(overlaps, dtype=np.int32)


# -------------------------------
# vLLM metrics scraping
# -------------------------------
VLLM_KEYS = [
    "prefill_tokens_cached_total",
    "prefill_tokens_reused_total",
    "kv_cache_prefix_cache_hits_total",
    "kv_cache_prefix_cache_lookups_total",
]

def parse_prometheus_text(text: str) -> Dict[str, float]:
    metrics = {}
    # simple Prometheus exposition parser for counters
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for key in VLLM_KEYS:
            # match key{...} value   OR  key value
            if line.startswith(key):
                try:
                    val = float(line.split()[-1])
                    metrics[key] = val
                except Exception:
                    pass
    return metrics


def fetch_vllm_metrics(url: str) -> Dict[str, float]:
    if requests is None:
        raise RuntimeError("requests not installed; cannot scrape --vllm-metrics-url")
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    return parse_prometheus_text(resp.text)


def load_vllm_metrics_file(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_prometheus_text(text)


def compute_pcr_from_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    reused = metrics.get("prefill_tokens_reused_total", 0.0)
    cached = metrics.get("prefill_tokens_cached_total", 0.0)
    lookups = metrics.get("kv_cache_prefix_cache_lookups_total", 0.0)
    hits = metrics.get("kv_cache_prefix_cache_hits_total", 0.0)
    pcr_tokens = (reused / cached) if cached > 0 else 0.0
    pcr_requests = (hits / lookups) if lookups > 0 else 0.0
    return {"pcr_tokens": pcr_tokens, "pcr_requests": pcr_requests}


# -------------------------------
# Plot helpers (one chart per figure; no explicit colors)
# -------------------------------
def plot_pdi_curve(pdi: Dict[int, float], out_path: str):
    xs = sorted(pdi.keys())
    ys = [pdi[x] for x in xs]
    plt.figure(figsize=(8,5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Prefix length (tokens)")
    plt.ylabel("PDI = unique_prefixes / total")
    plt.title("Prefix Diversity Index (PDI) Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")


def plot_overlap_histogram(overlaps: np.ndarray, out_path: str):
    if overlaps.size == 0:
        print("No overlaps to plot.")
        return
    plt.figure(figsize=(8,5))
    plt.hist(overlaps, bins=50, alpha=0.8)
    plt.xlabel("Shared Prefix Tokens")
    plt.ylabel("Frequency")
    plt.title("Prefix Overlap Length Histogram")
    # annotate stats
    stats = f"min={int(np.min(overlaps))}, median={int(np.median(overlaps))}, max={int(np.max(overlaps))}"
    plt.text(0.98, 0.98, stats, transform=plt.gca().transAxes, ha="right", va="top")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prefix Efficiency Dashboard")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=str, help="HF dataset name (e.g. tatsu-lab/alpaca)")
    src.add_argument("--file", type=str, help="Local file: .txt, .jsonl, .json, .pickle/.pkl")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--field", type=str, help="Field to read from dataset/JSONL/JSON (default: text/prompt/input heuristics)")
    parser.add_argument("--token-column", type=str, help="Column name containing pre-tokenized data (auto-detected if not specified)")

    parser.add_argument("--tokenizer", type=str, help="HF tokenizer name (required if using text data, not needed for pre-tokenized data)")
    parser.add_argument("--prefix-len", type=int, default=128, help="Prefix length for PRR/PE/PUS (default: 128)")
    parser.add_argument("--sample-size", type=int, default=2000, help="Max samples to analyze (default: 2000)")
    parser.add_argument("--hash", type=str, default="md5", choices=["md5","sha1","sha256","pyhash"], help="Hash function for prefix identity (default: md5)")

    parser.add_argument("--pdi-lens", type=str, default="16,32,64,128,256,512", help="Comma-separated prefix lengths for PDI curve")
    parser.add_argument("--overlap-pairs", type=int, default=3000, help="Random pairs to sample for overlap histogram (default: 3000)")
    parser.add_argument("--plots", action="store_true", help="Generate plots (PDI curve + overlap histogram)")

    # vLLM metrics correlation
    parser.add_argument("--vllm-metrics-url", type=str, help="Prometheus text endpoint for vLLM (e.g. http://localhost:8000/metrics)")
    parser.add_argument("--vllm-metrics-file", type=str, help="Path to a text dump of /metrics")

    parser.add_argument("--outdir", type=str, default=".", help="Directory to write plots and report (default: .)")
    
    # File inspection options
    parser.add_argument("--inspect", action="store_true", help="Inspect file contents and show columns, types, and sample rows")
    parser.add_argument("--inspect-rows", type=int, default=5, help="Number of sample rows to show during inspection (default: 5)")
    
    # Data export options
    parser.add_argument("--export-fields", type=str, help="Comma-separated list of fields to export to text file")
    parser.add_argument("--export-file", type=str, help="Output file path for exported fields (default: exported_data.txt)")
    parser.add_argument("--export-format", type=str, default="txt", choices=["txt", "json", "csv"], help="Export format (default: txt)")

    args = parser.parse_args()

    # Load data into pandas DataFrame
    print("Loading data...")
    df = load_dataframe(args)
    if len(df) == 0:
        print("No data found. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Check if tokenizer is needed
    needs_tokenizer = True
    if args.token_column and args.token_column in df.columns:
        needs_tokenizer = False
        print(f"Using pre-tokenized data from column '{args.token_column}' - tokenizer not needed")
    else:
        # Check if auto-detection finds tokenized data
        token_detection = detect_tokenized_data(df)
        if token_detection['is_tokenized']:
            needs_tokenizer = False
            print(f"Auto-detected pre-tokenized data in column '{token_detection['primary_token_column']}' - tokenizer not needed")
    
    # Validate tokenizer requirement
    if needs_tokenizer and not args.tokenizer:
        print("Error: --tokenizer is required when using text data (not pre-tokenized)", file=sys.stderr)
        print("Either specify --tokenizer for text data or --token-column for pre-tokenized data", file=sys.stderr)
        sys.exit(1)

    # Handle file inspection if requested
    if args.inspect and args.file:
        print("Inspecting file contents...")
        file_info = inspect_dataframe(df, args.file, args.field, args.inspect_rows)
        print_file_inspection(file_info)
        
        # If only inspection is requested, exit after showing the report
        if not any([args.plots, args.vllm_metrics_url, args.vllm_metrics_file, args.export_fields]):
            print("\nFile inspection complete. Use other flags to run analysis.")
            return

    # Handle field export if requested
    if args.export_fields:
        fields_to_export = [field.strip() for field in args.export_fields.split(',')]
        output_file = args.export_file or "exported_data.txt"
        
        # If no extension specified, add based on format
        if not any(output_file.endswith(ext) for ext in ['.txt', '.json', '.csv']):
            output_file += f".{args.export_format}"
        
        export_fields(df, fields_to_export, output_file, args.export_format)
        
        # If only export is requested, exit after exporting
        if not any([args.plots, args.vllm_metrics_url, args.vllm_metrics_file]):
            print("\nData export complete.")
            return

    # Only proceed with metrics computation if not just inspecting or exporting
    if not (args.inspect and not any([args.plots, args.vllm_metrics_url, args.vllm_metrics_file, args.export_fields])):
        # Load tokenizer only if needed
        tokenizer = None
        if needs_tokenizer:
            print(f"Loading tokenizer: {args.tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        else:
            print("Skipping tokenizer loading - using pre-tokenized data")

        # Tokenize (up to max prefix length needed for any metric)
        required_max_len = max([args.prefix_len] + [int(x) for x in args.pdi_lens.split(",") if x.strip().isdigit()])
        df = tokenize_dataframe(df, tokenizer, required_max_len, args.token_column)

        # Compute PRR / PUS / PE at prefix_len
        prefix_hashes = df['token_ids'].apply(lambda ids: hash_prefix(ids[:args.prefix_len], hash_name=args.hash))
        prr = compute_prr(prefix_hashes)
        pus = compute_pus(prefix_hashes)
        pe  = compute_entropy(prefix_hashes)

        # Compute PDI curve
        pdi_lens = [int(x) for x in args.pdi_lens.split(",") if x.strip().isdigit()]
        pdi = compute_pdi_curve(df, pdi_lens, hash_name=args.hash)

        # Overlap sampling (POL histogram)
        overlaps = sample_prefix_overlap_lengths(df, num_pairs=args.overlap_pairs)
        
        pol_stats = {}
        if overlaps.size > 0:
            pol_stats = {
                "min": int(np.min(overlaps)),
                "p50": int(np.median(overlaps)),
                "p90": int(np.percentile(overlaps, 90)),
                "max": int(np.max(overlaps)),
                "mean": float(np.mean(overlaps)),
            }

        # vLLM metrics correlation (optional)
        vllm_metrics = {}
        pcr = {}
        if args.vllm_metrics_url:
            try:
                vllm_metrics = fetch_vllm_metrics(args.vllm_metrics_url)
                pcr = compute_pcr_from_metrics(vllm_metrics)
            except Exception as e:
                print(f"Warning: failed to scrape vLLM metrics from URL: {e}", file=sys.stderr)
        elif args.vllm_metrics_file:
            try:
                vllm_metrics = load_vllm_metrics_file(args.vllm_metrics_file)
                pcr = compute_pcr_from_metrics(vllm_metrics)
            except Exception as e:
                print(f"Warning: failed to parse vLLM metrics file: {e}", file=sys.stderr)

        # Output
        os.makedirs(args.outdir, exist_ok=True)
        report_path = os.path.join(args.outdir, "metric_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Prefix Efficiency Dashboard — Summary\n")
            f.write("====================================\n\n")
            f.write(f"Samples analyzed: {len(df)}\n")
            f.write(f"Data source: {args.file or args.dataset}\n")
            f.write(f"Tokenizer: {args.tokenizer}\n")
            f.write(f"Prefix length (PRR/PE/PUS): {args.prefix_len}\n")
            f.write(f"Hash: {args.hash}\n")
            f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB\n\n")
            f.write(f"PRR  (Prefix Reuse Ratio): {prr:.4f}\n")
            f.write(f"PUS  (Prefix Uniqueness Score, mean #others sharing prefix): {pus:.4f}\n")
            f.write(f"PE   (Prefix Entropy, bits): {pe:.4f}\n\n")
            if pol_stats:
                f.write("POL  (Prefix Overlap Length) stats (random pairs):\n")
                for k, v in pol_stats.items():
                    f.write(f"  - {k}: {v}\n")
                f.write("\n")
            f.write("PDI  (Prefix Diversity Index) curve:\n")
            for L in sorted(pdi.keys()):
                f.write(f"  - L={L:>4d}: {pdi[L]:.4f}\n")
            f.write("\n")
            if pcr:
                f.write("vLLM Runtime Correlation:\n")
                f.write(f"  - PCR_tokens   (reused/cached tokens): {pcr.get('pcr_tokens', 0.0):.4f}\n")
                f.write(f"  - PCR_requests (hits/lookups)       : {pcr.get('pcr_requests', 0.0):.4f}\n")
                for k in VLLM_KEYS:
                    if k in vllm_metrics:
                        f.write(f"    * {k}: {vllm_metrics[k]:.0f}\n")
                f.write("\n")
        print(f"✅ Wrote summary: {report_path}")

        if args.plots:
            plot_pdi_curve(pdi, os.path.join(args.outdir, "pdi_curve.png"))
            if overlaps.size > 0:
                plot_overlap_histogram(overlaps, os.path.join(args.outdir, "prefix_overlap_histogram.png"))


if __name__ == "__main__":
    main()
