#!/usr/bin/env python3
"""Tokenization utilities for gpt-oss."""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/gpt-oss-120b"


class StandardTokenizer:
    """Standard tokenizer wrapper for gpt-oss model."""

    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize the tokenizer.
        
        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self.tokenizer = None
        logger.info(f"Initializing tokenizer for {model_name}")

    def load(self) -> None:
        """Load the tokenizer."""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            self.load()
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            self.load()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text: str) -> List[int]:
        """Encode text to token IDs (callable interface).
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return self.encode(text)


def load_tokenized_dataset(
    dataset_path: str,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Load a tokenized dataset from pickle file.
    
    Args:
        dataset_path: Path to the pickle file containing tokenized data
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        Dictionary containing:
            - prompts: List of tokenized prompts
            - dataframe: Original DataFrame
            - metadata: Additional metadata
    """
    logger.info(f"Loading tokenized dataset from {dataset_path}")
    
    # Load DataFrame from pickle
    df = pd.read_pickle(dataset_path)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")

    # Limit samples if specified
    if max_samples is not None:
        df = df.head(max_samples)
        logger.info(f"Limited to {max_samples} samples")

    # Extract tokenized prompts
    if 'tok_input' not in df.columns:
        raise ValueError("Dataset must have 'tok_input' column with tokenized prompts")

    # Verify tokenization
    failed_mask = df['tok_input'].isna()
    if failed_mask.any():
        failed_count = failed_mask.sum()
        logger.error(f"Found {failed_count} samples with failed tokenization")
        raise ValueError(f"{failed_count} samples have invalid tokenization")

    prompts = df['tok_input'].tolist()
    logger.info(f"Loaded {len(prompts)} tokenized prompts")

    # Log statistics
    prompt_lengths = [len(p) for p in prompts]
    logger.info(
        f"Prompt length stats - "
        f"min: {min(prompt_lengths)}, "
        f"max: {max(prompt_lengths)}, "
        f"mean: {sum(prompt_lengths)/len(prompt_lengths):.1f}"
    )

    return {
        "prompts": prompts,
        "dataframe": df,
        "metadata": {
            "num_samples": len(prompts),
            "min_length": min(prompt_lengths),
            "max_length": max(prompt_lengths),
            "mean_length": sum(prompt_lengths) / len(prompt_lengths)
        }
    }

