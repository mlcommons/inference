#!/usr/bin/env python3
"""Evaluate MLPerf performance logs and analyze output token lengths.

This script reads MLPerf accuracy logs (mlperf_log_accuracy.json) and
detokenizes the hex-encoded token IDs to produce human-readable text output.
Optionally includes input prompts and reference data from a pickle file,
and generates histogram plots for token length analysis.

Usage:
    # Basic usage (outputs only)
    python eval_mlperf_performance.py \
        --mlperf-log mlperf_logs/offline/accuracy/mlperf_log_accuracy.json \
        --output-file detokenized_outputs.json \
        --tokenizer openai/gpt-oss-120b

    # With reference data (includes inputs and metadata)
    python eval_mlperf_performance.py \
        --mlperf-log mlperf_logs/offline/accuracy/mlperf_log_accuracy.json \
        --output-file detokenized_outputs.json \
        --reference-data data/accuracy_eval_tokenized_filtered.pkl \
        --tokenizer openai/gpt-oss-120b

    # With histogram plots (enables plotting when --plot-dir is specified)
    python eval_mlperf_performance.py \
        --mlperf-log mlperf_logs/offline/accuracy/mlperf_log_accuracy.json \
        --output-file detokenized_outputs.json \
        --reference-data data/accuracy_eval_tokenized_filtered.pkl \
        --plot-dir plots

The output JSON format (with reference data):
    [
        {
            "qsl_idx": 0,
            "token_ids": [1, 2, 3, ...],
            "text": "detokenized response text",
            "num_tokens": 150,
            "dataset": "gpqa",
            "input_prompt": "Question: ...",
            "input_token_ids": [...],
            "num_input_tokens": 1024,
            "ground_truth": "Answer"
        },
        ...
    ]
"""

from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parse MLPerf accuracy JSON and detokenize responses"
    )

    parser.add_argument(
        "--mlperf-log",
        type=str,
        required=True,
        help="Path to mlperf_log_accuracy.json file"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output JSON file with detokenized responses"
    )

    parser.add_argument(
        "--reference-data",
        type=str,
        default=None,
        help="Path to reference parquet or pickle file (DataFrame with prompts, dataset, etc.) - optional"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="openai/gpt-oss-120b",
        help="Tokenizer to use for detokenization (default: openai/gpt-oss-120b)"
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON with indentation"
    )

    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save histogram plots (enables plotting if specified)"
    )

    return parser.parse_args()


def decode_hex_to_tokens(hex_string: str) -> List[int]:
    """Decode hex-encoded byte array to list of token IDs.

    MLPerf stores token IDs as hex-encoded bytes where each token is a 4-byte
    little-endian integer.

    Args:
        hex_string: Hex-encoded string from MLPerf log

    Returns:
        List of token IDs
    """
    # Remove any whitespace
    hex_string = hex_string.strip()

    # Convert hex string to bytes
    byte_data = bytes.fromhex(hex_string)

    # Each token is stored as 4 bytes (int32) in little-endian format
    token_ids = []
    for i in range(0, len(byte_data), 4):
        if i + 4 <= len(byte_data):
            # Unpack 4 bytes as little-endian int32
            token_id = int.from_bytes(
                byte_data[i:i + 4], byteorder='little', signed=True)
            token_ids.append(token_id)

    return token_ids


def parse_mlperf_log(log_path: str) -> List[Dict[str, Any]]:
    """Parse MLPerf accuracy log file.

    Handles multiple formats:
    - JSON array: [{"qsl_idx": 0, ...}, ...]
    - JSONL: one JSON object per line
    - Concatenated JSON: multiple JSON objects on same line

    Args:
        log_path: Path to mlperf_log_accuracy.json

    Returns:
        List of entries with qsl_idx and hex-encoded data
    """
    logger.info(f"Reading MLPerf log: {log_path}")

    entries = []

    # First try to load as a single JSON array
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        if isinstance(log_data, list):
            logger.info(f"Loaded {len(log_data)} entries as JSON array")
            return log_data
    except json.JSONDecodeError:
        pass  # Not a valid JSON array, try line-by-line parsing

    # Parse line by line (JSONL or concatenated JSON)
    decoder = json.JSONDecoder()
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Try to parse as single JSON object first
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                # Line might have multiple concatenated JSON objects
                # Extract them one by one using raw_decode
                remaining = line
                parsed_count = 0
                while remaining:
                    remaining = remaining.lstrip()
                    if not remaining:
                        break
                    try:
                        obj, end_idx = decoder.raw_decode(remaining)
                        entries.append(obj)
                        remaining = remaining[end_idx:]
                        parsed_count += 1
                    except json.JSONDecodeError as e:
                        if parsed_count == 0:
                            logger.warning(
                                f"Line {line_num}: Could not parse JSON: {e}")
                        break

    logger.info(f"Loaded {len(entries)} entries from MLPerf log")
    return entries


def plot_histograms(
    results: List[Dict[str, Any]],
    output_dir: str,
    has_reference: bool = False
) -> None:
    """Generate histogram plots for output token lengths and differences.

    Args:
        results: List of parsed results with token lengths
        output_dir: Directory to save plots
        has_reference: Whether reference data is available for difference plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating histogram plots in {output_dir}...")

    # Extract output token lengths
    output_lengths = [r['num_tokens'] for r in results]

    # Plot 1: Output Sequence Length (OSL) Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(
        output_lengths,
        bins=50,
        edgecolor='black',
        alpha=0.7,
        color='steelblue')
    plt.xlabel('Output Token Length (OSL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(
        f'Distribution of Output Token Lengths\n(n={len(output_lengths)}, mean={sum(output_lengths)/len(output_lengths):.1f}, median={sorted(output_lengths)[len(output_lengths)//2]})',
        fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    # Add statistics box
    stats_text = f'Min: {min(output_lengths)}\nMax: {max(output_lengths)}\nStd: {(sum((x - sum(output_lengths)/len(output_lengths))**2 for x in output_lengths) / len(output_lengths))**0.5:.1f}'
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    osl_plot_path = output_path / 'output_token_length_histogram.png'
    plt.tight_layout()
    plt.savefig(osl_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved OSL histogram: {osl_plot_path}")

    # Plot 2: Token Length Difference Histogram (if reference data available)
    if has_reference:
        results_with_diff = [
            r for r in results if 'output_token_len_diff' in r]
        if results_with_diff:
            differences = [r['output_token_len_diff']
                           for r in results_with_diff]

            plt.figure(figsize=(12, 6))
            plt.hist(
                differences,
                bins=50,
                edgecolor='black',
                alpha=0.7,
                color='coral')
            plt.xlabel(
                'Token Length Difference (Actual - Reference)',
                fontsize=12)
            plt.ylabel('Frequency', fontsize=12)

            mean_diff = sum(differences) / len(differences)
            median_diff = sorted(differences)[len(differences) // 2]
            plt.title(
                f'Distribution of Output Token Length Differences\n(n={len(differences)}, mean={mean_diff:.1f}, median={median_diff})',
                fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.axvline(
                x=0,
                color='red',
                linestyle='--',
                linewidth=2,
                label='Zero difference')

            # Add statistics box
            longer = sum(1 for d in differences if d > 0)
            shorter = sum(1 for d in differences if d < 0)
            exact = sum(1 for d in differences if d == 0)
            stats_text = f'Min: {min(differences)}\nMax: {max(differences)}\nStd: {(sum((x - mean_diff)**2 for x in differences) / len(differences))**0.5:.1f}\n\nLonger: {longer} ({longer/len(differences)*100:.1f}%)\nShorter: {shorter} ({shorter/len(differences)*100:.1f}%)\nExact: {exact} ({exact/len(differences)*100:.1f}%)'
            plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            plt.legend()

            diff_plot_path = output_path / 'token_length_difference_histogram.png'
            plt.tight_layout()
            plt.savefig(diff_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved difference histogram: {diff_plot_path}")

            # Plot 3: Combined comparison (side by side)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Reference vs Actual
            ref_lengths = [r['ref_num_output_tokens']
                           for r in results_with_diff]
            actual_lengths = [r['actual_num_output_tokens']
                              for r in results_with_diff]

            ax1.hist([ref_lengths, actual_lengths], bins=50, label=['Reference', 'Actual'],
                     alpha=0.6, edgecolor='black', color=['steelblue', 'coral'])
            ax1.set_xlabel('Output Token Length', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title(
                f'Reference vs Actual Output Token Lengths\n(n={len(results_with_diff)})',
                fontsize=13)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)

            # Scatter plot: Reference vs Actual
            ax2.scatter(
                ref_lengths,
                actual_lengths,
                alpha=0.4,
                s=10,
                color='purple')
            ax2.plot([min(ref_lengths), max(ref_lengths)], [min(ref_lengths), max(ref_lengths)],
                     'r--', linewidth=2, label='y=x (perfect match)')
            ax2.set_xlabel('Reference Token Length', fontsize=12)
            ax2.set_ylabel('Actual Token Length', fontsize=12)
            ax2.set_title(
                'Reference vs Actual Token Lengths (Scatter)',
                fontsize=13)
            ax2.legend()
            ax2.grid(alpha=0.3)

            comparison_plot_path = output_path / 'token_length_comparison.png'
            plt.tight_layout()
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved comparison plot: {comparison_plot_path}")
        else:
            logger.warning("No samples with token length differences found")

    logger.info(f"✓ All plots saved to {output_dir}/")


def detokenize_responses(
    entries: List[Dict[str, Any]],
    tokenizer: Any,
    reference_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """Detokenize responses from MLPerf log entries.

    When reference data is provided, input_prompt is generated by detokenizing
    input token IDs from the reference data (checks: tok_input, input_token_ids,
    input_tokens, tokenized_input). This shows exactly what was sent to the model
    (after tokenization), not the original text prompt.

    Args:
        entries: List of MLPerf log entries with hex-encoded token IDs
        tokenizer: HuggingFace tokenizer instance
        reference_df: Optional reference DataFrame with input prompts and metadata

    Returns:
        List of dictionaries with qsl_idx, token_ids, and detokenized text
    """
    logger.info("Detokenizing responses...")

    results = []
    for entry in tqdm(entries, desc="Detokenizing", unit="response"):
        qsl_idx = entry.get("qsl_idx")
        hex_data = entry.get("data", "")

        # Decode hex to token IDs
        try:
            token_ids = decode_hex_to_tokens(hex_data)
        except Exception as e:
            logger.error(f"Error decoding tokens for qsl_idx={qsl_idx}: {e}")
            token_ids = []

        # Detokenize to text
        try:
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error detokenizing qsl_idx={qsl_idx}: {e}")
            text = ""

        # Build result record
        result = {
            "qsl_idx": qsl_idx,
            "token_ids": token_ids,
            "text": text,
            "num_tokens": len(token_ids)
        }

        # Add reference data if available
        if reference_df is not None and qsl_idx < len(reference_df):
            ref_row = reference_df.iloc[qsl_idx]

            # Add common fields from reference data
            if 'dataset' in ref_row:
                result['dataset'] = ref_row['dataset']

            # Get input token IDs and detokenize to see what was actually sent to the model
            # Check multiple possible field names for input tokens
            input_token_ids = None
            for field in ['tok_input', 'input_token_ids',
                          'input_tokens', 'tokenized_input']:
                if field in ref_row:
                    input_token_ids = ref_row[field]
                    break

            if input_token_ids is not None:
                result['input_token_ids'] = input_token_ids
                if isinstance(input_token_ids, list):
                    result['num_input_tokens'] = len(input_token_ids)
                    # Detokenize input tokens to show what was actually sent to
                    # the model
                    try:
                        result['input_prompt'] = tokenizer.decode(
                            input_token_ids, skip_special_tokens=False)
                    except Exception as e:
                        logger.warning(
                            f"Error detokenizing input tokens for qsl_idx={qsl_idx}: {e}")
                        result['input_prompt'] = None
                else:
                    result['num_input_tokens'] = None
                    result['input_prompt'] = None
            else:
                # Fallback to raw prompt field if input token IDs not available
                if 'prompt' in ref_row:
                    result['input_prompt'] = ref_row['prompt']
                elif 'input_text' in ref_row:
                    result['input_prompt'] = ref_row['input_text']
                elif 'text' in ref_row:
                    result['input_prompt'] = ref_row['text']

            if 'ground_truth' in ref_row:
                result['ground_truth'] = ref_row['ground_truth']

            # Compute output token length difference
            # Check for reference output token length in various possible field
            # names
            ref_output_len = None
            for field in ['output_token_ids', 'target_token_ids',
                          'output_tokens', 'expected_output_token_ids']:
                if field in ref_row:
                    ref_tokens = ref_row[field]
                    if isinstance(ref_tokens, list):
                        ref_output_len = len(ref_tokens)
                        result['ref_output_token_ids'] = ref_tokens
                        break
                    elif isinstance(ref_tokens, (int, float)) and not pd.isna(ref_tokens):
                        ref_output_len = int(ref_tokens)
                        break

            # Also check for direct length field
            if ref_output_len is None:
                for field in ['output_len', 'output_length',
                              'num_output_tokens', 'target_len']:
                    if field in ref_row and not pd.isna(ref_row[field]):
                        ref_output_len = int(ref_row[field])
                        break

            if ref_output_len is not None:
                actual_output_len = len(token_ids)
                result['ref_num_output_tokens'] = ref_output_len
                result['actual_num_output_tokens'] = actual_output_len
                result['output_token_len_diff'] = actual_output_len - \
                    ref_output_len
                result['output_token_len_ratio'] = actual_output_len / \
                    ref_output_len if ref_output_len > 0 else None

            # Add any other columns that might be useful
            for col in ['question_id', 'difficulty', 'subject', 'category']:
                if col in ref_row:
                    result[col] = ref_row[col]

        results.append(result)

    return results


def main():
    """Main function."""
    args = parse_args()

    # Validate input file exists
    log_path = Path(args.mlperf_log)
    if not log_path.exists():
        logger.error(f"MLPerf log file not found: {args.mlperf_log}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("MLPerf Accuracy Log Parser")
    logger.info("=" * 80)
    logger.info(f"Input log: {args.mlperf_log}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(
        f"Reference data: {args.reference_data if args.reference_data else 'None (outputs only)'}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info("=" * 80)

    # Load reference data if provided
    reference_df = None
    if args.reference_data:
        logger.info(f"Loading reference data from {args.reference_data}")
        try:
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

            logger.info(f"✓ Reference data loaded: {reference_df.shape}")
            logger.info(f"  Columns: {list(reference_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            sys.exit(1)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        logger.info("✓ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Parse MLPerf log
    try:
        entries = parse_mlperf_log(args.mlperf_log)
    except Exception as e:
        logger.error(f"Failed to parse MLPerf log: {e}")
        sys.exit(1)

    if not entries:
        logger.error("No entries found in MLPerf log")
        sys.exit(1)

    # Detokenize responses
    try:
        results = detokenize_responses(entries, tokenizer, reference_df)
    except Exception as e:
        logger.error(f"Failed to detokenize responses: {e}")
        sys.exit(1)

    # Write output JSON
    logger.info(f"Writing detokenized outputs to: {args.output_file}")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        if args.pretty:
            json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            json.dump(results, f, ensure_ascii=False)

    logger.info("=" * 80)
    logger.info("✓ Parsing completed successfully")
    logger.info("=" * 80)
    logger.info(f"Total responses parsed: {len(results)}")

    # Print statistics
    total_tokens = sum(r["num_tokens"] for r in results)
    avg_tokens = total_tokens / len(results) if results else 0
    logger.info(f"Total output tokens: {total_tokens:,}")
    logger.info(f"Average tokens per response: {avg_tokens:.1f}")

    # Print token length difference statistics if reference data was provided
    if reference_df is not None:
        results_with_diff = [
            r for r in results if 'output_token_len_diff' in r]
        if results_with_diff:
            diffs = [r['output_token_len_diff'] for r in results_with_diff]
            ratios = [r['output_token_len_ratio']
                      for r in results_with_diff if r['output_token_len_ratio'] is not None]

            logger.info(
                f"\nOutput Token Length Analysis ({len(results_with_diff)} samples with reference):")
            logger.info(
                f"  Mean difference (actual - ref): {sum(diffs) / len(diffs):.2f} tokens")
            logger.info(f"  Min difference: {min(diffs)} tokens")
            logger.info(f"  Max difference: {max(diffs)} tokens")
            if ratios:
                logger.info(
                    f"  Mean ratio (actual / ref): {sum(ratios) / len(ratios):.3f}x")

            # Count samples that are longer/shorter
            longer = sum(1 for d in diffs if d > 0)
            shorter = sum(1 for d in diffs if d < 0)
            exact = sum(1 for d in diffs if d == 0)
            logger.info(
                f"  Longer than reference: {longer} ({longer/len(diffs)*100:.1f}%)")
            logger.info(
                f"  Shorter than reference: {shorter} ({shorter/len(diffs)*100:.1f}%)")
            logger.info(
                f"  Exact match: {exact} ({exact/len(diffs)*100:.1f}%)")

    logger.info("=" * 80)

    # Show sample output
    if results:
        logger.info("Sample output (first entry):")
        sample = results[0]
        logger.info(f"  qsl_idx: {sample['qsl_idx']}")
        logger.info(f"  num_tokens: {sample['num_tokens']}")
        logger.info(f"  text preview: {sample['text'][:200]}...")
        logger.info("=" * 80)

    # Generate histogram plots if plot directory is specified
    if args.plot_dir:
        logger.info("\n" + "=" * 80)
        logger.info("Generating Histogram Plots")
        logger.info("=" * 80)
        plot_histograms(
            results, args.plot_dir, has_reference=(
                reference_df is not None))
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
