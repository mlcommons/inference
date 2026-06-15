#!/usr/bin/env python3
"""
Evaluate MLPerf accuracy logs for gpt-oss-120b.

This script takes MLPerf accuracy JSON logs and a reference pickle file,
evaluates the outputs, and generates accuracy scores by dataset and overall.

Usage:
    python eval_mlperf_accuracy.py \
        --mlperf-log mlperf_logs_offline_x8_acc/offline/accuracy/mlperf_log_accuracy.json \
        --reference-data data/accuracy_eval_tokenized_filtered.pkl \
        --output-file accuracy_results.json
"""

from eval_accuracy import (
    get_evaluator, validate_dataset_name, validate_text_input, DATASET_EVALUATORS,
    evaluate_livecodebench_worker, load_lcb_benchmark
)
import argparse
import json
import logging
import pickle
import struct
import multiprocessing
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Import evaluation functions from the existing script
import sys
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded repeats per dataset for final score calculation
# Final score = sum(dataset_correct / dataset_repeats)
DATASET_REPEATS = {
    'aime25': 8,
    'gpqa_diamond': 5,
    'livecodebench_v6': 3,
}


def load_mlperf_log(log_path: str) -> List[Dict[str, Any]]:
    """Load MLPerf accuracy JSON log.

    Args:
        log_path: Path to mlperf_log_accuracy.json

    Returns:
        List of log entries with seq_id, qsl_idx, data (hex), token_count
    """
    logger.info(f"Loading MLPerf log from {log_path}")
    with open(log_path, 'r') as f:
        log_data = json.load(f)

    logger.info(f"Loaded {len(log_data)} log entries")

    return log_data


def decode_hex_to_tokens(hex_data: str) -> List[int]:
    """Decode hex string to list of token IDs (int32).

    MLPerf stores token IDs as hex-encoded int32 array.

    Args:
        hex_data: Hex string like "450D0300..."

    Returns:
        List of token IDs
    """
    # Convert hex string to bytes
    data_bytes = bytes.fromhex(hex_data)

    # Unpack as int32 array (little-endian)
    num_tokens = len(data_bytes) // 4
    token_ids = struct.unpack(f'<{num_tokens}i', data_bytes)

    return list(token_ids)


def detokenize(token_ids: List[int], tokenizer) -> str:
    """Convert token IDs to text.

    Args:
        token_ids: List of integer token IDs
        tokenizer: HuggingFace tokenizer

    Returns:
        Decoded text string
    """
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def process_livecodebench_batch(
    entries: List[Dict[str, Any]],
    reference_df: pd.DataFrame,
    tokenizer,
    evaluator: Dict[str, Any],
    lcb_executor: ProcessPoolExecutor,
    dataset_name: str,
    args
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of LiveCodeBench entries in parallel.

    Args:
        entries: List of MLPerf log entries for this dataset
        reference_df: Reference DataFrame
        tokenizer: HuggingFace tokenizer
        evaluator: Evaluator functions dict
        lcb_executor: ProcessPoolExecutor for parallel evaluation
        dataset_name: Dataset name
        args: Command line arguments

    Returns:
        Tuple of (results_list, outputs_list)
    """
    # First pass: decode and parse all entries
    work_items = []
    # Store (entry, qsl_idx, ref_row, token_ids, model_output)
    entry_metadata = []

    logger.info(f"Parsing {len(entries)} {dataset_name} entries...")
    for entry in tqdm(entries, desc=f"Parsing {dataset_name}", unit="entry"):
        seq_id = entry['seq_id']
        qsl_idx = entry['qsl_idx']
        hex_data = entry['data']

        ref_row = reference_df.iloc[qsl_idx]
        ground_truth = ref_row.get('ground_truth', None)

        # Decode tokens to text
        token_ids = decode_hex_to_tokens(hex_data)
        model_output = detokenize(token_ids, tokenizer)

        # Parse code from model output
        extracted_code = evaluator['parse'](model_output)

        entry_metadata.append({
            'entry': entry,
            'qsl_idx': qsl_idx,
            'ref_row': ref_row,
            'token_ids': token_ids,
            'model_output': model_output,
            'extracted_code': extracted_code,
            'ground_truth': ground_truth
        })

        # Add to work queue if code was extracted
        if extracted_code is not None and not pd.isna(ground_truth):
            work_items.append((extracted_code, ground_truth))
        else:
            work_items.append(None)  # Placeholder for skipped items

    # Second pass: batch evaluate code in parallel
    logger.info(
        f"Evaluating {len([w for w in work_items if w is not None])} {dataset_name} code samples with parallel workers...")

    results_list = []
    outputs_list = []

    # Submit all work items
    future_to_idx = {}
    for idx, work_item in enumerate(work_items):
        if work_item is not None:
            future = lcb_executor.submit(
                evaluate_livecodebench_worker, work_item)
            future_to_idx[future] = idx

    # Collect results with progress bar
    eval_results = [None] * len(work_items)

    for future in tqdm(as_completed(future_to_idx.keys(), timeout=1200),
                       total=len(future_to_idx),
                       desc=f"Evaluating {dataset_name}",
                       unit="sample"):
        idx = future_to_idx[future]
        try:
            question_id, is_correct, detailed_reason = future.result(
                timeout=80)
            eval_results[idx] = (is_correct, detailed_reason)
        except TimeoutError:
            logger.warning(
                f"Timeout evaluating sample {idx}: Test execution exceeded 80s timeout")
            eval_results[idx] = (
                False, "Timeout: Test execution exceeded time limit")
        except Exception as e:
            logger.error(f"Error evaluating sample {idx}: {e}")
            eval_results[idx] = (False, f"Error: {e}")

    # Third pass: compile final results
    for idx, metadata in enumerate(entry_metadata):
        entry = metadata['entry']
        qsl_idx = metadata['qsl_idx']
        token_ids = metadata['token_ids']
        model_output = metadata['model_output']
        extracted_code = metadata['extracted_code']
        ground_truth = metadata['ground_truth']

        # Get evaluation result
        if extracted_code is None or pd.isna(ground_truth):
            is_correct = False
            eval_details = "No code extracted from model output" if extracted_code is None else "No ground truth available"
        else:
            is_correct, eval_details = eval_results[idx]

        # Record result
        result = {
            'seq_id': entry['seq_id'],
            'qsl_idx': qsl_idx,
            'dataset': dataset_name,
            'is_correct': is_correct,
            'extracted_answer': str(extracted_code)[:200] if extracted_code is not None else None,
            'ground_truth': str(ground_truth) if not pd.isna(ground_truth) else None,
            'evaluation_details': eval_details,
            'token_count': len(token_ids),
            'model_output_preview': model_output[:200] if args.verbose else None
        }
        results_list.append(result)

        # Store output data if requested
        if args.save_outputs:
            output_record = {
                'qsl_idx': qsl_idx,
                'seq_id': entry['seq_id'],
                'dataset': dataset_name,
                'ground_truth': ground_truth,
                'model_output': model_output,
                'output_token_ids': token_ids,
                'extracted_answer': extracted_code,
                'is_correct': is_correct,
                'evaluation_details': eval_details
            }
            outputs_list.append(output_record)

    return results_list, outputs_list


def evaluate_single_entry(
    model_output: str,
    ground_truth: str,
    dataset_name: str
) -> Tuple[bool, Any, str]:
    """Evaluate a single model output.

    Args:
        model_output: Generated text from model
        ground_truth: Expected answer
        dataset_name: Dataset name (e.g., 'gpqa', 'math500')

    Returns:
        Tuple of (is_correct, extracted_answer, evaluation_details)
    """
    evaluator = get_evaluator(dataset_name)

    # Parse answer from model output
    extracted = evaluator['parse'](model_output)

    # Evaluate correctness
    is_correct = False
    evaluation_details = ""

    if extracted is None or pd.isna(extracted):
        evaluation_details = "No answer extracted from model output"
    else:
        if not pd.isna(ground_truth):
            try:
                is_correct = evaluator['evaluate'](extracted, ground_truth)
                if is_correct:
                    evaluation_details = "Correct"
                else:
                    evaluation_details = f"Incorrect (extracted: {extracted}, ground_truth: {ground_truth})"
            except Exception as e:
                evaluation_details = f"Evaluation error: {e}"
                logger.warning(f"Error evaluating: {e}")
        else:
            evaluation_details = "No ground truth available"

    return is_correct, extracted, evaluation_details


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLPerf accuracy logs for gpt-oss-120b"
    )
    parser.add_argument(
        "--mlperf-log",
        type=str,
        required=True,
        help="Path to mlperf_log_accuracy.json"
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        required=True,
        help="Path to reference parquet or pickle file (DataFrame with dataset, ground_truth, etc.)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="openai/gpt-oss-120b",
        help="HuggingFace tokenizer name or path"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--save-outputs",
        type=str,
        default=None,
        help="Save detokenized outputs to pickle file (ordered by qsl_idx) for debugging"
    )
    parser.add_argument(
        "--num-lcb-workers",
        type=int,
        default=64,
        help="Number of parallel workers for LiveCodeBench evaluation (default: 64)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load MLPerf log
    mlperf_log = load_mlperf_log(args.mlperf_log)

    # Load reference data
    logger.info(f"Loading reference data from {args.reference_data}")
    if args.reference_data.endswith('.parquet'):
        reference_df = pd.read_parquet(args.reference_data)
        logger.info("Loaded reference data from Parquet file")
    elif args.reference_data.endswith('.pkl') or args.reference_data.endswith('.pickle'):
        with open(args.reference_data, 'rb') as f:
            reference_df = pickle.load(f)
        logger.info("Loaded reference data from Pickle file")
    else:
        # Try parquet first, then pickle
        try:
            reference_df = pd.read_parquet(args.reference_data)
            logger.info("Auto-detected Parquet format")
        except Exception:
            with open(args.reference_data, 'rb') as f:
                reference_df = pickle.load(f)
            logger.info("Auto-detected Pickle format")

    # Convert numpy arrays to native Python types for JSON serialization
    for col in reference_df.columns:
        # Check if column contains numpy arrays
        if reference_df[col].dtype == object:
            reference_df[col] = reference_df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )

    logger.info(f"Reference data shape: {reference_df.shape}")
    logger.info(f"Reference columns: {list(reference_df.columns)}")

    # Validate required columns exist
    required_columns = ['dataset', 'ground_truth']
    missing_columns = [
        col for col in required_columns if col not in reference_df.columns]
    if missing_columns:
        raise ValueError(
            f"Reference data missing required columns: {missing_columns}")

    # Log unique datasets in reference data
    if 'dataset' in reference_df.columns:
        unique_datasets = reference_df['dataset'].unique()
        dataset_counts = reference_df['dataset'].value_counts()
        logger.info(
            f"Unique datasets in reference data ({len(unique_datasets)} total):")
        for ds in sorted(unique_datasets):
            logger.info(f"  '{ds}' ({dataset_counts[ds]} samples)")

        logger.info("\nSample rows from reference data:")
        for idx in [0, 1, 2]:
            if idx < len(reference_df):
                logger.info(
                    f"  Row {idx}: dataset='{reference_df.iloc[idx]['dataset']}'")

        # Show how each will be mapped to evaluators
        logger.info("\nExpected Dataset → Evaluator mapping:")
        for ds in sorted(unique_datasets):
            try:
                ds_lower = validate_dataset_name(ds)
                # Find which evaluator key matches
                matched_key = None
                for key in DATASET_EVALUATORS.keys():
                    if key in ds_lower:
                        matched_key = key
                        break
                logger.info(
                    f"  '{ds}' (normalized: '{ds_lower}') → '{matched_key}'")
            except Exception as e:
                logger.warning(f"  '{ds}' → ERROR: {e}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Group MLPerf log entries by dataset
    logger.info("Grouping MLPerf log entries by dataset...")
    dataset_entries = defaultdict(list)

    for entry in mlperf_log:
        qsl_idx = entry['qsl_idx']

        if qsl_idx >= len(reference_df):
            logger.warning(
                f"qsl_idx {qsl_idx} out of range (max: {len(reference_df)-1})")
            continue

        ref_row = reference_df.iloc[qsl_idx]
        dataset_name = validate_dataset_name(ref_row['dataset'])
        dataset_entries[dataset_name].append(entry)

    logger.info(f"Grouped entries by dataset:")
    total_entries = 0
    for ds_name, entries in sorted(dataset_entries.items()):
        logger.info(f"  {ds_name}: {len(entries)} entries")
        total_entries += len(entries)
    logger.info(f"Total entries: {total_entries}")

    # Pre-load LiveCodeBench benchmark if needed
    lcb_executor = None
    if any('livecodebench' in ds for ds in dataset_entries.keys()):
        try:
            logger.info(
                "Pre-loading LiveCodeBench benchmark for parallel evaluation...")
            os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm in workers
            _ = load_lcb_benchmark()
            logger.info("LiveCodeBench benchmark loaded successfully")

            # Create shared ProcessPoolExecutor for all LCB evaluations
            max_workers = min(
                multiprocessing.cpu_count(),
                args.num_lcb_workers)
            lcb_executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(
                f"Created ProcessPoolExecutor with {max_workers} workers for LiveCodeBench")
        except Exception as e:
            logger.error(f"Failed to pre-load LiveCodeBench benchmark: {e}")
            logger.error(
                f"Please make sure LiveCodeBench submodule is initalized at submodules/LiveCodeBench")
            raise RuntimeError(f"LiveCodeBench benchmark failed to load: {e}.")

    # Process each dataset separately with its own progress bar
    logger.info("\nProcessing MLPerf log entries by dataset...")

    results = []
    # Track stats per dataset (simple correct/total)
    dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    outputs_data = []  # For saving detokenized outputs

    try:
        for dataset_name in sorted(dataset_entries.keys()):
            entries = dataset_entries[dataset_name]
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing {dataset_name}: {len(entries)} samples")
            logger.info(f"{'=' * 80}")

            evaluator = get_evaluator(dataset_name)
            is_livecodebench = 'livecodebench' in dataset_name.lower()

            if is_livecodebench and lcb_executor is not None:
                # Batched LiveCodeBench evaluation
                results_batch, outputs_batch = process_livecodebench_batch(
                    entries, reference_df, tokenizer, evaluator,
                    lcb_executor, dataset_name, args
                )
                results.extend(results_batch)
                if args.save_outputs:
                    outputs_data.extend(outputs_batch)

                # Update stats
                for res in results_batch:
                    dataset_stats[dataset_name]["total"] += 1
                    if res['is_correct']:
                        dataset_stats[dataset_name]["correct"] += 1
            else:
                # Sequential evaluation for non-LCB datasets
                for entry in tqdm(
                        entries, desc=f"Evaluating {dataset_name}", unit="entry"):
                    seq_id = entry['seq_id']
                    qsl_idx = entry['qsl_idx']
                    hex_data = entry['data']

                    ref_row = reference_df.iloc[qsl_idx]
                    ground_truth = ref_row.get('ground_truth', None)

                    # Decode tokens to text
                    token_ids = decode_hex_to_tokens(hex_data)
                    model_output = detokenize(token_ids, tokenizer)

                    # Evaluate
                    try:
                        is_correct, extracted, eval_details = evaluate_single_entry(
                            model_output, ground_truth, dataset_name
                        )
                    except Exception as e:
                        logger.warning(
                            f"Evaluation error for qsl_idx={qsl_idx}, dataset={dataset_name}: {e}")
                        is_correct = False
                        extracted = None
                        eval_details = f"Evaluation error: {e}"

                    # Record result
                    result = {
                        'seq_id': seq_id,
                        'qsl_idx': qsl_idx,
                        'dataset': dataset_name,
                        'is_correct': is_correct,
                        'extracted_answer': str(extracted) if extracted is not None else None,
                        'ground_truth': str(ground_truth) if not pd.isna(ground_truth) else None,
                        'evaluation_details': eval_details,
                        'token_count': len(token_ids),
                        'model_output_preview': model_output[:200] if args.verbose else None
                    }
                    results.append(result)

                    # Store output data for pickle export
                    if args.save_outputs:
                        output_record = {
                            'qsl_idx': qsl_idx,
                            'seq_id': seq_id,
                            'dataset': dataset_name,
                            'ground_truth': ground_truth,
                            'model_output': model_output,
                            'output_token_ids': token_ids,
                            'extracted_answer': extracted,
                            'is_correct': is_correct,
                            'evaluation_details': eval_details
                        }
                        outputs_data.append(output_record)

                    # Update stats
                    dataset_stats[dataset_name]["total"] += 1
                    if is_correct:
                        dataset_stats[dataset_name]["correct"] += 1

    finally:
        # Clean up LiveCodeBench executor
        if lcb_executor is not None:
            logger.info("Shutting down LiveCodeBench ProcessPoolExecutor")
            lcb_executor.shutdown(wait=True)
            os.environ.pop('TQDM_DISABLE', None)

    # Calculate per-dataset scores and final score
    # Final score = sum(dataset_correct / dataset_repeats)
    logger.info("\nCalculating final scores...")

    total_correct = sum(stats["correct"] for stats in dataset_stats.values())
    total_samples = sum(stats["total"] for stats in dataset_stats.values())
    overall_accuracy = (
        total_correct /
        total_samples *
        100) if total_samples > 0 else 0.0

    # Calculate weighted final score
    final_score = 0.0
    max_score = 0.0
    final_score_components = {}
    for dataset_name, stats in dataset_stats.items():
        repeats = DATASET_REPEATS.get(dataset_name, 1)
        component_score = stats["correct"] / repeats
        max_component_score = stats["total"] / repeats
        final_score += component_score
        max_score += max_component_score
        final_score_components[dataset_name] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "repeats": repeats,
            "component_score": component_score,
            "max_component_score": max_component_score
        }

    final_score_percentage = (
        final_score /
        max_score *
        100) if max_score > 0 else 0.0

    # Print results
    print("\n" + "=" * 80)
    print("MLPerf Accuracy Evaluation Results")
    print("=" * 80)
    print(f"Total samples evaluated: {total_samples}")
    print(
        f"Overall raw accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    print("=" * 80)

    print("\nPer-Dataset Breakdown:")
    print("-" * 80)
    print(f"{'Dataset':25s} {'Correct':>8s} {'Total':>8s} {'Repeats':>8s} {'Score':>10s} {'Accuracy':>10s}")
    print("-" * 80)
    for dataset_name in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset_name]
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"] * 100)
            repeats = DATASET_REPEATS.get(dataset_name, 1)
            component_score = stats["correct"] / repeats
            print(
                f"{dataset_name:25s} {stats['correct']:8d} {stats['total']:8d} {repeats:8d} {component_score:10.2f} {accuracy:9.2f}%")

    print("=" * 80)
    print(f"\nFinal Score Calculation:")
    print("-" * 80)
    score_parts = []
    value_parts = []
    result_parts = []
    max_parts = []
    for dataset_name in sorted(final_score_components.keys()):
        comp = final_score_components[dataset_name]
        score_parts.append(f"{dataset_name}/{comp['repeats']}")
        value_parts.append(f"{comp['correct']}/{comp['repeats']}")
        result_parts.append(f"{comp['component_score']:.2f}")
        max_parts.append(f"{comp['total']}/{comp['repeats']}")
    print(f"Formula: {' + '.join(score_parts)}")
    print(f"Score:   = {' + '.join(value_parts)}")
    print(f"         = {' + '.join(result_parts)}")
    print(f"         = {final_score:.2f}")
    print(f"Max:     = {' + '.join(max_parts)}")
    print(f"         = {max_score:.2f}")
    print(
        f"\nFINAL SCORE: {final_score_percentage:.2f}% ({final_score:.2f}/{max_score:.2f})")
    print("=" * 80)

    print("\n\nPrinting for submission_checker:")
    print(f"\n'exact_match': {final_score_percentage:.3f}")

    # Save detokenized outputs to pickle if requested
    if args.save_outputs:
        logger.info(f"Saving detokenized outputs to {args.save_outputs}...")

        # Sort by qsl_idx for ordered output
        outputs_data_sorted = sorted(outputs_data, key=lambda x: x['qsl_idx'])

        # Convert to DataFrame for easier inspection
        outputs_df = pd.DataFrame(outputs_data_sorted)

        output_path = Path(args.save_outputs)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(outputs_df, f)

        logger.info(
            f"Saved {len(outputs_df)} detokenized outputs (ordered by qsl_idx) to: {output_path}")
        logger.info(f"Columns: {list(outputs_df.columns)}")

    # Save detailed results if requested
    if args.output_file:
        # Build per-dataset stats
        per_dataset_stats = {}
        for dataset_name, stats in dataset_stats.items():
            repeats = DATASET_REPEATS.get(dataset_name, 1)
            component_score = stats["correct"] / repeats
            max_component_score = stats["total"] / repeats
            per_dataset_stats[dataset_name] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0,
                "repeats": repeats,
                "component_score": component_score,
                "max_component_score": max_component_score
            }

        summary = {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "final_score": final_score,
            "max_score": max_score,
            "final_score_percentage": final_score_percentage,
            "dataset_repeats": DATASET_REPEATS,
            "per_dataset": per_dataset_stats
        }

        output_data = {
            "summary": summary,
            "detailed_results": results if args.verbose else None
        }

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
