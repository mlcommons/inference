"""Centralized validation utilities."""
from functools import wraps
from typing import Optional, List, Dict, Any
import pandas as pd
import argparse


class BackendError(RuntimeError):
    """Base exception for backend errors."""
    pass


class BackendNotInitializedError(BackendError):
    """Raised when backend operation is called before initialization."""

    def __init__(self, backend_name: str = "Backend"):
        super().__init__(
            f"{backend_name} not initialized. Call initialize() first.")


class ValidationError(ValueError):
    """Base exception for validation errors."""
    pass


def require_initialized(method):
    """Decorator to ensure backend is initialized before method execution."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, 'is_initialized', False):
            backend_name = getattr(self, 'backend_name', 'Backend')
            raise BackendNotInitializedError(backend_name)
        return method(self, *args, **kwargs)
    return wrapper


def validate_prompts_input(backend_name: Optional[str] = None,
                           tokenized_prompts: Optional[List[List[int]]] = None,
                           text_prompts: Optional[List[str]] = None,
                           input_type: str = None) -> None:
    """
    Centralized prompt validation with backend-specific requirements.

    Args:
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
        tokenized_prompts: List of tokenized prompts
        text_prompts: List of text prompts
        input_type: Expected input type from registry ('text' or 'tokenized')

    Raises:
        ValidationError: If validation fails
    """
    if backend_name is None:
        from .backend_registry import detect_backend
        backend_name = detect_backend()

    if tokenized_prompts is None and text_prompts is None:
        raise ValidationError(
            f"{backend_name} backend requires either text_prompts or tokenized_prompts")

    if input_type == 'text' and tokenized_prompts is not None and text_prompts is None:
        raise ValidationError(
            f"{backend_name} backend requires text_prompts, not tokenized_prompts")

    if input_type == 'tokenized' and text_prompts is not None and tokenized_prompts is None:
        raise ValidationError(
            f"{backend_name} backend requires tokenized_prompts, not text_prompts")

    # Additional validation for tokenized prompts
    if tokenized_prompts is not None:
        if not tokenized_prompts:
            raise ValidationError("tokenized_prompts cannot be empty")
        for i, prompt in enumerate(tokenized_prompts):
            if not isinstance(prompt, list):
                raise ValidationError(
                    f"tokenized_prompts[{i}] must be a list of integers")
            if not prompt:
                raise ValidationError(
                    f"tokenized_prompts[{i}] cannot be empty")

    # Additional validation for text prompts
    if text_prompts is not None:
        if not text_prompts:
            raise ValidationError("text_prompts cannot be empty")
        for i, prompt in enumerate(text_prompts):
            if not isinstance(prompt, str):
                raise ValidationError(f"text_prompts[{i}] must be a string")


def validate_dataset_extended(df: pd.DataFrame,
                              backend_name: Optional[str] = None,
                              required_columns: Optional[List[str]] = None) -> None:
    """
    Extended dataset validation with backend-specific requirements.

    Args:
        df: DataFrame to validate
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
        required_columns: Optional list of required columns (defaults to ['text_input'])

    Raises:
        ValidationError: If validation fails
    """
    if backend_name is None:
        from .backend_registry import detect_backend
        backend_name = detect_backend()

    if required_columns is None:
        required_columns = ['text_input']

    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(
            f"Dataset missing required columns: {missing_columns}")

    # Check for empty prompts
    empty_prompts = df['text_input'].isna().sum()
    if empty_prompts > 0:
        print(f"Warning: Found {empty_prompts} empty text_input in dataset")

    # Backend-specific validation could be added here
    if backend_name:
        # Example: validate max sequence length for specific backends
        from .backend_registry import get_backend_config, BACKEND_REGISTRY
        if backend_name in BACKEND_REGISTRY:
            config = get_backend_config(backend_name)
            # Add backend-specific validation based on config if needed

    print(
        f"Dataset validation passed: {len(df)} samples with required columns")


def validate_runner_args(args: argparse.Namespace, runner_type: str) -> None:
    """
    Centralized argument validation for runners.

    Args:
        args: Parsed command line arguments
        runner_type: Type of runner for specific validation

    Raises:
        ValidationError: If validation fails
    """
    # Common validations
    if hasattr(
            args, 'num_samples') and args.num_samples is not None and args.num_samples <= 0:
        raise ValidationError("--num-samples must be positive")

    if hasattr(args, 'skip_samples') and args.skip_samples < 0:
        raise ValidationError("--skip-samples cannot be negative")

    # Runner-specific validations
    if runner_type in ['mlperf', 'mlperf_mpi']:
        if hasattr(args, 'mode') and args.mode not in ['offline', 'server']:
            raise ValidationError(
                f"Invalid mode: {args.mode}. Must be 'offline' or 'server'")
