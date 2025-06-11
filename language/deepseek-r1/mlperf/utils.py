"""MLPerf-specific utilities for tokenization and dataset handling."""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from utils import load_dataset, validate_dataset
from utils.backend_registry import uses_text_input, uses_chat_template
from utils.tokenization import StandardTokenizer


def prepare_mlperf_dataset(input_file: str,
                           backend_name: Optional[str] = None,
                           tokenizer: StandardTokenizer = None,
                           num_samples: Optional[int] = None,
                           skip_samples: int = 0,
                           use_chat_template: Optional[bool] = None) -> Dict[str, Any]:
    """
    Prepare dataset for MLPerf inference.

    Args:
        input_file: Path to input pickle file
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
                      (Kept for backward compatibility but not used in our codebase)
        tokenizer: StandardTokenizer instance
        num_samples: Number of samples to use
        skip_samples: Number of samples to skip
        use_chat_template: Whether to use chat template (if None, determined by registry)

    Returns:
        Dictionary with prepared dataset components
    """
    if backend_name is None:
        from utils.backend_registry import detect_backend
        backend_name = detect_backend()

    # Load and validate dataset
    df = load_dataset(input_file, num_samples, skip_samples)
    validate_dataset(df)

    prompts = df['text_input'].tolist()
    print(f"[MLPerf] Loaded {len(prompts)} prompts from dataset")

    # Check if backend uses text prompts from registry
    uses_text_prompts = uses_text_input()

    # Determine chat template usage from registry if not specified
    if use_chat_template is None:
        use_chat_template = uses_chat_template()
        print(
            f"[MLPerf] Using chat template from registry: {use_chat_template}")

    if uses_text_prompts:
        print(f"[MLPerf] Backend {backend_name} uses text prompts directly")
        return {
            'dataframe': df,
            'prompts': prompts,
            'tokenized_prompts': prompts,  # For compatibility
            'processed_strings': prompts,
            'uses_text_prompts': True
        }
    else:
        print(f"[MLPerf] Tokenizing prompts for {backend_name} backend...")
        tokenized_prompts, processed_strings = tokenizer.tokenize_prompts(
            prompts, use_chat_template
        )
        print(f"[MLPerf] Tokenized {len(tokenized_prompts)} prompts")

        return {
            'dataframe': df,
            'prompts': prompts,
            'tokenized_prompts': tokenized_prompts,
            'processed_strings': processed_strings,
            'uses_text_prompts': False
        }


def process_mlperf_results(sut_results: List[Dict[str, Any]],
                           tokenizer: Optional[StandardTokenizer] = None,
                           backend_name: Optional[str] = None,
                           uses_text_prompts: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    Process MLPerf SUT results into standardized format.

    Args:
        sut_results: Raw results from MLPerf SUT
        tokenizer: StandardTokenizer for decoding
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
                      (Kept for backward compatibility but not used in our codebase)
        uses_text_prompts: Whether backend uses text prompts (if None, determined by registry)

    Returns:
        List of processed result dictionaries
    """
    from utils.tokenization import process_inference_results

    if backend_name is None:
        from utils.backend_registry import detect_backend
        backend_name = detect_backend()

    # Determine text prompt usage from registry if not specified
    if uses_text_prompts is None:
        uses_text_prompts = uses_text_input()

    # Reuse the general inference result processing
    return process_inference_results(
        sut_results, tokenizer, uses_text_prompts=uses_text_prompts)


def create_mlperf_output_dataframe(input_df: pd.DataFrame,
                                   results: List[Dict[str, Any]],
                                   backend_name: Optional[str] = None) -> pd.DataFrame:
    """
    Create output dataframe with MLPerf results.

    Args:
        input_df: Input dataframe
        results: Processed MLPerf results
        backend_name: Optional backend name override. If None, uses MLPERF_BACKEND env var.
                      (Kept for backward compatibility but not used in our codebase)

    Returns:
        Output dataframe with results
    """
    if backend_name is None:
        from utils.backend_registry import detect_backend
        backend_name = detect_backend()

    df_output = input_df.copy()

    # Add result columns
    df_output['model_output'] = [r['model_output'] for r in results]
    df_output['tok_model_output'] = [r['tok_model_output'] for r in results]
    df_output['tok_model_output_len'] = [
        r['tok_model_output_len'] for r in results]
    df_output['model_backend'] = backend_name

    return df_output
