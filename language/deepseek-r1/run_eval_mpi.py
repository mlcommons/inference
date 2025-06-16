#!/usr/bin/env python3
from backends import BaseBackend
from utils.data_utils import load_dataset
from utils.validation import validate_runner_args, ValidationError
from utils.runner_utils import create_base_argument_parser, print_runner_header
from utils.backend_registry import uses_chat_template, get_backend_instance, detect_backend, validate_runner_for_backend
from utils import save_results, generate_timestamped_filename, StandardTokenizer
from backends.pytorch_backend import PyTorchBackend
import os
import sys
import argparse
import pandas as pd
from typing import Optional
import builtins

import torch
import torch.distributed as dist

# Import utilities and backend registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(
    input_pickle_path: str,
    output_pickle_path: str,
    num_samples: Optional[int] = None,
    skip_samples: int = 0,
) -> None:
    """
    Main function to load the model, process prompts from a DataFrame, and save results.
    """
    _print = builtins.print  # Capture the original built-in print

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Override print for non-rank 0 processes
    if rank != 0:
        print = lambda *_, **__: None

    # Detect backend from environment
    backend_name = detect_backend()

    # Validate backend
    validate_runner_for_backend('eval_mpi')

    # Get chat template usage from registry
    use_chat_template = uses_chat_template()

    # Generate the actual filename with timestamp that will be used for saving
    actual_output_file = generate_timestamped_filename(
        output_pickle_path, add_timestamp=True)

    if rank == 0:
        _print("=" * 80)
        _print("Distributed PyTorch Backend Evaluation System")
        _print("=" * 80)
        _print(f"Backend: {backend_name}")
        _print(f"World size: {world_size}")
        _print(f"Input file: {input_pickle_path}")
        _print(f"Output file: {actual_output_file}")
        if num_samples:
            _print(f"Sample limit: {num_samples}")
        if skip_samples:
            _print(f"Skip samples: {skip_samples}")
        _print(
            f"Chat template: {'enabled' if use_chat_template else 'disabled'} (from registry)")
        _print("=" * 80)

    # Initialize PyTorch backend
    backend = PyTorchBackend()
    backend.initialize()

    # Initialize StandardTokenizer
    tokenizer = StandardTokenizer()

    # Only rank 0 handles data
    prompts_text_list = []
    df_for_results = None

    if rank == 0:
        # Load DataFrame
        _print(f"Loading input DataFrame from {input_pickle_path}...")
        try:
            df_for_results = pd.read_pickle(input_pickle_path)
            _print(
                f"Loaded DataFrame with {len(df_for_results)} rows and columns: {df_for_results.columns.tolist()}")

            # Apply skip_samples if specified
            if skip_samples > 0:
                if skip_samples >= len(df_for_results):
                    _print(
                        f"Error: skip_samples ({skip_samples}) is greater than or equal to total samples ({len(df_for_results)})")
                    backend.shutdown()
                    if world_size > 1:
                        dist.destroy_process_group()
                    return
                _print(f"Skipping first {skip_samples} samples")
                df_for_results = df_for_results.iloc[skip_samples:].copy()
                # Reset index to ensure sequential indices starting from 0
                df_for_results = df_for_results.reset_index(drop=True)

            # Apply num_samples limit if specified
            if num_samples is not None and num_samples < len(df_for_results):
                _print(
                    f"Limiting to first {num_samples} samples (out of {len(df_for_results)} total after skipping)")
                df_for_results = df_for_results.head(num_samples).copy()
                # Reset index to ensure sequential indices starting from 0
                df_for_results = df_for_results.reset_index(drop=True)

        except Exception as e:
            _print(f"Error loading input pickle file: {e}")
            backend.shutdown()
            if world_size > 1:
                dist.destroy_process_group()
            return

        if 'text_input' not in df_for_results.columns:
            _print("Error: 'text_input' column not found in the input DataFrame.")
            backend.shutdown()
            if world_size > 1:
                dist.destroy_process_group()
            return

        prompts_text_list = df_for_results['text_input'].tolist()
        _print(
            f"Extracted {len(prompts_text_list)} prompts from 'text_input' column.")

        # Pre-initialize output columns
        df_for_results['model_output'] = ""
        df_for_results['tok_model_output'] = None
        df_for_results['tok_model_output'] = df_for_results['tok_model_output'].astype(
            'object')
        df_for_results['tok_model_output_len'] = 0
        df_for_results['model_backend'] = backend_name

    # Broadcast the number of prompts to all ranks
    if world_size > 1:
        if rank == 0:
            num_prompts_tensor = torch.tensor(
                len(prompts_text_list), dtype=torch.long, device="cuda")
        else:
            num_prompts_tensor = torch.empty(
                1, dtype=torch.long, device="cuda")
        dist.broadcast(num_prompts_tensor, src=0)
        num_total_prompts = num_prompts_tensor.item()
    else:
        num_total_prompts = len(prompts_text_list)

    batch_size = backend.config['batch_size']

    # Process prompts in batches
    for i in range(0, num_total_prompts, batch_size):
        current_batch_num = (i // batch_size) + 1
        current_batch_prompt_texts = None
        current_batch_prompt_tokens = None

        if rank == 0:
            current_batch_prompt_texts = prompts_text_list[i:i + batch_size]
            # Tokenize on rank 0 using StandardTokenizer
            current_batch_prompt_tokens, _ = tokenizer.tokenize_prompts(
                current_batch_prompt_texts, use_chat_template
            )

            _print(
                f"Processing batch {current_batch_num}, size {len(current_batch_prompt_tokens)}")

        # All ranks call generate_batch_distributed
        generated_tokens_for_batch = backend.generate_batch_distributed(
            current_batch_prompt_tokens if rank == 0 else None
        )

        if rank == 0:
            # Validate that we received valid tokens
            if not generated_tokens_for_batch:
                raise RuntimeError(
                    f"Backend returned empty tokens for batch {current_batch_num}")

            for batch_idx, tokens in enumerate(generated_tokens_for_batch):
                if not isinstance(tokens, (list, tuple)) or len(tokens) == 0:
                    raise RuntimeError(
                        f"Backend returned empty or invalid tokens for batch {current_batch_num}, item {batch_idx}: {tokens}")

            # Decode tokens to text using StandardTokenizer
            decoded_texts_for_batch = tokenizer.batch_decode(
                generated_tokens_for_batch
            )

            # Update DataFrame for the current batch
            start_index_in_df = i
            num_items_in_batch_output = len(decoded_texts_for_batch)

            for batch_idx in range(num_items_in_batch_output):
                original_df_idx = start_index_in_df + batch_idx
                if original_df_idx < len(df_for_results):
                    # Use at for assignments with list values
                    df_for_results.at[original_df_idx,
                                      'model_output'] = decoded_texts_for_batch[batch_idx]
                    df_for_results.at[original_df_idx,
                                      'tok_model_output'] = generated_tokens_for_batch[batch_idx]
                    df_for_results.at[original_df_idx, 'tok_model_output_len'] = len(
                        generated_tokens_for_batch[batch_idx])

            _print(f"Batch {current_batch_num} completed.")

    if rank == 0 and df_for_results is not None:
        _print(f"All batches processed. Saving results...")

        # Keep only required columns in the same order as run_eval.py
        output_columns = [
            'text_input',
            'ground_truth',
            'question',
            'dataset',
            'model_output',
            'tok_model_output',
            'tok_model_output_len',
            'model_backend']
        # Filter to only columns that exist
        output_columns = [
            col for col in output_columns if col in df_for_results.columns]
        df_output = df_for_results[output_columns]

        try:
            saved_file = save_results(
                df_output, output_pickle_path, add_timestamp=True)
            _print(f"Successfully saved results to {saved_file}")
        except Exception as e:
            _print(f"Error saving output pickle file: {e}")

    # Clean up
    backend.shutdown()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = create_base_argument_parser(
        "Run distributed inference with PyTorch backend using MPI/torchrun"
    )
    # Note: --no-chat-template is not in the base parser and no longer needed

    args = parser.parse_args()

    # Validate arguments
    try:
        validate_runner_args(args, 'eval_mpi')
    except ValidationError as e:
        print(f"Argument validation error: {e}")
        sys.exit(1)

    # Set default output file if not specified
    if args.output_file is None:
        args.output_file = "data/pytorch_output.pkl"

    main(
        args.input_file,
        args.output_file,
        args.num_samples,
        args.skip_samples,
    )
