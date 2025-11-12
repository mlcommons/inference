#!/usr/bin/env python3
"""
Summarize evaluation results from eval_accuracy.py output.

Reads an evaluated pickle file and prints a summary of results by dataset,
including per-pass statistics and aggregated pass@k results.
"""

import argparse
import pickle
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def detect_pass_k(df: pd.DataFrame) -> int:
    """Detect if DataFrame has pass@k format and return k.

    Returns:
        Number of passes (k) if pass@k format detected, otherwise 1
    """
    # Check for model_output_0, model_output_1, etc.
    pass_k = 0
    while f'model_output_{pass_k}' in df.columns:
        pass_k += 1

    # If no _0 suffix found, check for single model_output column
    if pass_k == 0 and 'model_output' in df.columns:
        return 1

    return pass_k


def calculate_dataset_stats(df: pd.DataFrame, dataset_name: str,
                            pass_num: int = None, pass_k: int = 1) -> Dict[str, Any]:
    """Calculate statistics for a specific dataset and pass.

    Args:
        df: DataFrame with evaluation results
        dataset_name: Name of the dataset to filter
        pass_num: Pass number (None for aggregated results)
        pass_k: Total number of passes (for aggregated results)

    Returns:
        Dictionary with statistics
    """
    # Filter to this dataset
    dataset_df = df[df['dataset'] == dataset_name]

    # Determine column suffixes
    if pass_num is None:
        # Aggregated results
        accuracy_col = 'prompt_accuracy' if 'prompt_accuracy' in dataset_df.columns else 'prompt_accuracy_0'

        # For aggregated pass@k, count answered as any sample with at least one
        # extracted answer
        if pass_k > 1:
            # Check if any pass has an extracted answer
            answered_mask = pd.Series(
                [False] * len(dataset_df),
                index=dataset_df.index)
            for i in range(pass_k):
                col = f'extracted_answer_{i}'
                if col in dataset_df.columns:
                    answered_mask |= dataset_df[col].notna()
            answered = answered_mask.sum()
        else:
            extracted_col = 'extracted_answer' if 'extracted_answer' in dataset_df.columns else 'extracted_answer_0'
            answered = dataset_df[extracted_col].notna().sum()
    else:
        # Specific pass
        suffix = f'_{pass_num}'
        extracted_col = f'extracted_answer{suffix}'
        accuracy_col = f'prompt_accuracy{suffix}'
        answered = dataset_df[extracted_col].notna().sum()

    # Calculate statistics
    total = len(dataset_df)
    correct = (dataset_df[accuracy_col] > 0).sum()

    # Calculate percentage (correct / total)
    if total > 0:
        pct_correct = (correct / total) * 100
    else:
        pct_correct = 0.0

    # Calculate mean accuracy (handles HealthBench partial scores)
    mean_accuracy = dataset_df[accuracy_col].mean()

    return {
        'dataset': dataset_name,
        'total': int(total),
        'answered': int(answered),
        'correct': int(correct),
        'pct_correct': float(pct_correct),
        'mean_accuracy': float(mean_accuracy),
    }


def print_summary_table(
        stats_list: List[Dict[str, Any]], title: str = "Summary"):
    """Print a formatted summary table.

    Args:
        stats_list: List of statistics dictionaries
        title: Title for the table
    """
    print(f"\n{'=' * 85}")
    print(f"{title}")
    print('=' * 85)
    print(f"{'Dataset':<20} {'Total':>8} {'Answered':>10} {'Correct':>10} {'Accuracy':>12}")
    print('-' * 85)

    for stats in stats_list:
        dataset_name = stats['dataset']
        total = stats['total']
        answered = stats['answered']
        correct = stats['correct']
        pct_correct = stats['pct_correct']

        # Format the row
        print(
            f"{dataset_name:<20} {total:>8} {answered:>10} {correct:>10} {pct_correct:>11.2f}%")

    # Print totals
    if len(stats_list) > 1:
        total_samples = sum(s['total'] for s in stats_list)
        total_answered = sum(s['answered'] for s in stats_list)
        total_correct = sum(s['correct'] for s in stats_list)
        overall_pct = (
            total_correct /
            total_samples *
            100) if total_samples > 0 else 0.0

        print('-' * 85)
        print(f"{'OVERALL':<20} {total_samples:>8} {total_answered:>10} {total_correct:>10} {overall_pct:>11.2f}%")

    print('=' * 85)


def summarize_evaluation(pickle_path: str, json_output: bool = False) -> str:
    """Load and summarize evaluation results.

    Args:
        pickle_path: Path to evaluated pickle file
        json_output: If True, save results to JSON file instead of printing

    Returns:
        Path to JSON file if json_output=True, otherwise empty string
    """
    # Load the pickle file
    print(f"Loading evaluation results from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded {len(df)} samples")

    # Detect pass@k format
    pass_k = detect_pass_k(df)
    print(f"Detected format: pass@{pass_k}" if pass_k >
            1 else "Detected format: single-pass")

    # Get list of datasets
    datasets = sorted(df['dataset'].unique())
    print(f"Datasets found: {', '.join(datasets)}")

    # Structure to hold all results
    results_data = {
        'input_file': pickle_path,
        'total_samples': len(df),
        'pass_k': pass_k,
        'datasets': list(datasets),
    }

    # Calculate statistics for each dataset
    if pass_k > 1:
        # Collect per-pass statistics
        per_pass_results = []
        for pass_num in range(pass_k):
            stats_list = []
            for dataset in datasets:
                stats = calculate_dataset_stats(
                    df, dataset, pass_num=pass_num, pass_k=pass_k)
                stats_list.append(stats)

            print_summary_table(stats_list, title=f"Pass {pass_num} Results")

            per_pass_results.append({
                'pass_number': pass_num,
                'datasets': stats_list,
                'overall': {
                    'total': sum(s['total'] for s in stats_list),
                    'answered': sum(s['answered'] for s in stats_list),
                    'correct': sum(s['correct'] for s in stats_list),
                    'accuracy': (sum(s['correct'] for s in stats_list) / sum(s['total'] for s in stats_list) * 100) if sum(s['total'] for s in stats_list) > 0 else 0.0
                }
            })

        results_data['per_pass_results'] = per_pass_results

        # Show aggregated (pass@k) statistics
        print("\n")
        stats_list = []
        for dataset in datasets:
            stats = calculate_dataset_stats(
                df, dataset, pass_num=None, pass_k=pass_k)
            stats_list.append(stats)

        aggregated_results = {
            'datasets': stats_list,
            'overall': {
                'total': sum(s['total'] for s in stats_list),
                'answered': sum(s['answered'] for s in stats_list),
                'correct': sum(s['correct'] for s in stats_list),
                'accuracy': (sum(s['correct'] for s in stats_list) / sum(s['total'] for s in stats_list) * 100) if sum(s['total'] for s in stats_list) > 0 else 0.0
            }
        }
        results_data['aggregated_results'] = aggregated_results

        # Always print summary table
        print_summary_table(
            stats_list,
            title=f"Aggregated Pass@{pass_k} Results (Max Across Passes)")
    else:
        # Single pass - just show the results
        stats_list = []
        for dataset in datasets:
            stats = calculate_dataset_stats(
                df, dataset, pass_num=None, pass_k=pass_k)
            stats_list.append(stats)

        single_pass_results = {
            'datasets': stats_list,
            'overall': {
                'total': sum(s['total'] for s in stats_list),
                'answered': sum(s['answered'] for s in stats_list),
                'correct': sum(s['correct'] for s in stats_list),
                'accuracy': (sum(s['correct'] for s in stats_list) / sum(s['total'] for s in stats_list) * 100) if sum(s['total'] for s in stats_list) > 0 else 0.0
            }
        }
        results_data['results'] = single_pass_results

        # Always print summary table
        print_summary_table(stats_list, title="Evaluation Results")

    # Print column information for reference
    print("\nColumn Information:")
    print(f"  - Total: Total number of samples in the dataset")
    if pass_k > 1:
        print(f"  - Answered: Number of samples with at least one extracted answer across all passes")
    else:
        print(f"  - Answered: Number of samples with extracted answers")
    print(f"  - Correct: Number of correct answers (accuracy > 0)")
    print(f"  - Accuracy: Percentage of total samples that were correct (correct / total)")

    if pass_k > 1:
        print(f"\nPass@{pass_k} Note:")
        print(f"  - Per-pass results show individual pass performance")
        print(
            f"  - Aggregated results show the maximum accuracy achieved across all {pass_k} passes")
        print(
            f"  - A sample is considered correct if ANY of the {pass_k} attempts were correct")
        print(
            f"  - A sample is considered answered if ANY of the {pass_k} attempts extracted an answer")

    # Save to JSON if requested
    if json_output:
        # Generate output filename: input_file_summarize.json
        input_path = Path(pickle_path)
        output_filename = input_path.stem + "_summarize.json"
        output_path = input_path.parent / output_filename

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nSummary saved to: {output_path}")
        return str(output_path)

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Summarize evaluation results by dataset")
    parser.add_argument("input_file",
                        help="Path to evaluated pickle file from eval_accuracy.py")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format (for programmatic use)")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(
            f"Error: Input file not found: {args.input_file}",
            file=sys.stderr)
        sys.exit(1)

    # Check if file has _evaluated suffix (warn if not)
    if "_evaluated" not in args.input_file:
        print(f"Warning: Input file does not contain '_evaluated' suffix. "
              f"Make sure this is an evaluated pickle file from eval_accuracy.py",
              file=sys.stderr)

    try:
        summarize_evaluation(args.input_file, json_output=args.json)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
