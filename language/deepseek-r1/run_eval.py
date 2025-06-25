#!/usr/bin/env python3
from utils import (
    load_dataset, save_results, validate_dataset, generate_timestamped_filename,
    validate_runner_for_backend, uses_text_input, uses_chat_template,
    StandardTokenizer, process_inference_results,
    get_backend_instance, create_base_argument_parser, print_runner_header,
    setup_output_paths, validate_runner_args, handle_runner_error,
    validate_dataset_extended, supports_async
)
from backends import BaseBackend
import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with shared arguments only."""
    parser = create_base_argument_parser(
        "Modular backend evaluation system for MLPerf DeepSeek reference implementation"
    )

    # Add runner-specific arguments
    parser.add_argument("--async", action="store_true",
                        help="Use async generation instead of synchronous")

    return parser


async def run_async_inference(backend: BaseBackend,
                              tokenized_prompts: List[List[int]],
                              text_prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Run async inference with proper error handling and progress bar that updates as tasks complete."""
    try:
        # Get futures from backend
        if uses_text_input():
            futures = backend.generate_async(text_prompts=text_prompts)
        else:
            futures = backend.generate_async(
                tokenized_prompts=tokenized_prompts)

        # Create a list to store results in order
        results = [None] * len(futures)

        # Create enumerated futures with their original indices for tracking
        indexed_futures = [(i, future) for i, future in enumerate(futures)]

        # Track completion for debugging
        completed_indices = set()

        # Process tasks with progress bar that updates as tasks complete
        with async_tqdm(total=len(futures), desc="Async inference", unit="prompt") as pbar:
            # Use asyncio.wait with FIRST_COMPLETED to handle out-of-order
            # completion
            pending = {future for _, future in indexed_futures}

            while pending:
                # Wait for at least one future to complete
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # Process all completed futures in this batch
                for completed_future in done:
                    # Find the original index for this completed future
                    original_idx = None
                    for idx, future in indexed_futures:
                        if future is completed_future:
                            original_idx = idx
                            break

                    if original_idx is None:
                        print(
                            f"\nWarning: Could not find original index for completed future")
                        continue

                    # Check for duplicate completion
                    if original_idx in completed_indices:
                        print(
                            f"\nWarning: Prompt {original_idx} completed multiple times!")
                        continue

                    try:
                        # Get the result from the completed future
                        result = await completed_future

                        # Store the result in the correct position
                        results[original_idx] = result
                        completed_indices.add(original_idx)

                    except Exception as e:
                        print(
                            f"\nError processing prompt {original_idx}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exception(type(e), e, e.__traceback__)

                        # Raise the error instead of using empty tokens
                        raise RuntimeError(
                            f"Backend failed to generate tokens for prompt {original_idx}: {e}")

                    # Update progress bar after each completion
                    pbar.update(1)

        # Verify all results are populated
        if len(completed_indices) != len(futures):
            missing_count = len(futures) - len(completed_indices)
            raise RuntimeError(
                f"Missing results: completed {len(completed_indices)} != {len(futures)} total ({missing_count} missing)")

        for i, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"Missing result for prompt {i}")

        print(f"\nCompleted all {len(completed_indices)} prompts successfully")

        return results
    except Exception as e:
        print(f"Error during async inference: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_sync_inference(backend: BaseBackend,
                       tokenized_prompts: List[List[int]],
                       text_prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Run sync inference with proper error handling."""
    try:
        if uses_text_input():
            results = backend.generate(text_prompts=text_prompts)
        else:
            results = backend.generate(tokenized_prompts=tokenized_prompts)
        return results
    except Exception as e:
        print(f"Error during sync inference: {e}")
        raise


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Validate arguments
        validate_runner_args(args, 'eval')

        # Detect backend early
        backend_name = validate_runner_for_backend('eval')

        # Set up output paths
        output_dir, output_file = setup_output_paths(args)
        if args.output_file is None:
            args.output_file = output_file

        # Generate the actual filename with timestamp that will be used for
        # saving
        actual_output_file = generate_timestamped_filename(
            args.output_file, add_timestamp=True)

        # Get async flag using getattr since 'async' is a reserved keyword
        use_async = getattr(args, 'async', False)

        # Check if backend supports async
        if use_async and not supports_async():
            raise RuntimeError(
                f"Backend {backend_name} does not support async generation")

        # Print header
        print_runner_header(
            "Modular Backend Evaluation System",
            backend_name,
            args)
        print(f"Mode: {'Async' if use_async else 'Sync'}")
        print("=" * 80)

        # Load and validate dataset
        df = load_dataset(args.input_file, args.num_samples, args.skip_samples)
        validate_dataset_extended(df)

        prompts = df['text_input'].tolist()

        # Initialize tokenizer
        tokenizer = StandardTokenizer()

        # Determine whether to use chat template based on registry
        use_chat_template = uses_chat_template()

        # For text-prompt backends, we'll pass the prompts directly
        # For tokenized-prompt backends, we need to tokenize first
        if uses_text_input():
            print(f"Backend {backend_name} uses text prompts directly")
            tokenized_prompts = None
            processed_strings = prompts
        else:
            # Tokenize prompts before initializing backend
            print("Tokenizing prompts...")
            print(f"Using chat template: {use_chat_template}")
            tokenized_prompts, processed_strings = tokenizer.tokenize_prompts(
                prompts, use_chat_template
            )
            print(f"Tokenized {len(tokenized_prompts)} prompts")
            print(f"Tokenizer Max length: {tokenizer.max_length}")

        # Initialize backend using registry
        print(f"\nInitializing {backend_name.upper()} backend...")
        backend = get_backend_instance(backend_name)

        with backend:
            # Create new output dataframe with only required columns
            df_output = pd.DataFrame()

            # Copy all columns from input dataframe first
            for col in df.columns:
                df_output[col] = df[col]

            # Run inference with appropriate prompt format
            if use_async:
                print("Running async inference...")
                raw_results = asyncio.run(run_async_inference(
                    backend, tokenized_prompts, text_prompts=prompts))
            else:
                print("Running sync inference...")
                raw_results = run_sync_inference(
                    backend, tokenized_prompts, text_prompts=prompts)

            # Process raw results into standardized format using shared utility
            print("Processing results...")
            standardized_results = process_inference_results(
                raw_results, tokenizer
            )

            # Add generated columns
            df_output['model_output'] = [r['model_output']
                                         for r in standardized_results]
            df_output['tok_model_output'] = [r['tok_model_output']
                                             for r in standardized_results]
            df_output['tok_model_output_len'] = [
                r['tok_model_output_len'] for r in standardized_results]
            df_output['model_backend'] = [r['model_backend']
                                          for r in standardized_results]

            # Save results
            output_file = save_results(
                df_output, args.output_file, add_timestamp=True)

            print(f"\nEvaluation completed successfully!")
            print(f"Results saved to: {output_file}")
            print(f"Output columns: {list(df_output.columns)}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        handle_runner_error(e, "run_eval.py")


if __name__ == "__main__":
    main()
