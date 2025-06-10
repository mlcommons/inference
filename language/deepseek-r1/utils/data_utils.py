"""
Data utilities for loading datasets and saving results.

Provides common functionality for data handling across all backends.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from utils.validation import ValidationError, validate_dataset_extended


def generate_timestamped_filename(
        output_file: str, add_timestamp: bool = True) -> str:
    """
    Generate the actual filename that will be used when saving, with timestamp if requested.

    Args:
        output_file: Base output file path
        add_timestamp: Whether to add timestamp to filename

    Returns:
        Actual filename that will be used for saving
    """
    if not add_timestamp:
        return output_file

    timestamp_suffix = time.strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(output_file)
    return f"{base_name}_{timestamp_suffix}{ext}"


def load_dataset(
        file_path: str, num_samples: Optional[int] = None, skip_samples: int = 0) -> pd.DataFrame:
    """
    Load dataset from pickle file.

    Args:
        file_path: Path to the pickle file
        num_samples: Optional limit on number of samples to load
        skip_samples: Number of samples to skip from the beginning

    Returns:
        Loaded DataFrame

    Raises:
        ValidationError: If file doesn't exist or validation fails
        Exception: If file can't be loaded
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"Input file not found: {file_path}")

    print(f"Loading dataset from {file_path}...")

    try:
        with open(file_path, "rb") as f:
            df = pd.read_pickle(f)
    except Exception as e:
        raise ValidationError(f"Failed to load dataset: {str(e)}")

    print(f"Loaded {len(df)} samples")

    # Skip samples if specified
    if skip_samples > 0:
        if skip_samples >= len(df):
            raise ValidationError(
                f"skip_samples ({skip_samples}) must be less than total samples ({len(df)})"
            )
        original_length = len(df)
        df = df.iloc[skip_samples:].reset_index(drop=True)
        print(
            f"Skipped first {skip_samples} samples (from {original_length} total)")

    # Limit number of samples if specified
    if num_samples is not None:
        original_length = len(df)
        df = df.head(num_samples)
        print(
            f"Limited to {len(df)} samples (from {original_length} total after skipping)")

    return df


def save_results(df: pd.DataFrame,
                 output_file: str,
                 add_timestamp: bool = True) -> str:
    """
    Save results DataFrame to pickle file.

    Args:
        df: DataFrame to save
        output_file: Output file path
        add_timestamp: Whether to add timestamp to filename

    Returns:
        Actual output file path used

    Raises:
        ValidationError: If save operation fails
    """
    # Add timestamp to filename if requested
    if add_timestamp:
        timestamp_suffix = time.strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(output_file)
        output_file = f"{base_name}_{timestamp_suffix}{ext}"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving results to {output_file}...")

    # Reset index before saving
    df_to_save = df.reset_index(drop=True)

    try:
        with open(output_file, "wb") as f:
            pickle.dump(df_to_save, f)
        print(
            f"Save completed: {len(df_to_save)} samples saved to {output_file}")
    except Exception as e:
        raise ValidationError(f"Failed to save results: {str(e)}")

    return output_file


def prepare_output_dataframe(input_df: pd.DataFrame,
                             backend_name: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare output DataFrame by cleaning up old columns.

    Args:
        input_df: Input DataFrame
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Returns:
        Cleaned DataFrame ready for new results
    """
    if backend_name is None:
        from utils.backend_registry import detect_backend
        backend_name = detect_backend()

    df_output = input_df.copy()

    # Define columns to drop (old model outputs and unwanted columns)
    columns_to_drop = [
        # specify columns to drop here
    ]

    # Also drop any existing backend-specific columns
    backend_columns = [
        col for col in df_output.columns if col.startswith(f'{backend_name}_')]
    columns_to_drop.extend(backend_columns)

    # Drop columns that exist
    df_output = df_output.drop(
        columns=[col for col in columns_to_drop if col in df_output.columns]
    )

    return df_output


def add_standardized_columns(df: pd.DataFrame,
                             results: List[Dict[str, Any]],
                             tokenized_prompts: List[List[int]] = None) -> pd.DataFrame:
    """
    Add standardized output columns to DataFrame.

    Args:
        df: Input DataFrame
        results: List of result dictionaries from backend
        tokenized_prompts: List of tokenized input prompts (deprecated, not used)

    Returns:
        DataFrame with added standardized columns
    """
    # Add results columns with new naming convention
    df['model_output'] = [r.get('model_output', '') for r in results]
    df['tok_model_output'] = [r.get('tok_model_output', []) for r in results]
    df['tok_model_output_len'] = [
        r.get(
            'tok_model_output_len',
            0) for r in results]
    df['model_backend'] = [r.get('model_backend', '') for r in results]

    return df


def validate_dataset(df: pd.DataFrame,
                     backend_name: Optional[str] = None) -> None:
    """
    Validate that the dataset has required columns.

    Args:
        df: DataFrame to validate
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.

    Raises:
        ValidationError: If required columns are missing or validation fails
    """
    # Use centralized validation function
    validate_dataset_extended(df, backend_name)
