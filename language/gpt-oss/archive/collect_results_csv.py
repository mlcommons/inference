#!/usr/bin/env python3
"""
Collect results from multiple summarize_eval.py JSON outputs into a CSV.

The CSV format shows:
- Each row represents one dataset from one JSON file
- Columns: run_1, run_2, ..., run_k, pass@k
- Values are the "correct" counts (number of correct answers)
"""

import argparse
import json
import csv
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def expand_glob_patterns(patterns: List[str]) -> List[str]:
    """Expand glob patterns to actual file paths.

    Args:
        patterns: List of file paths or glob patterns (e.g., '*.json', 'results_*_summarize.json')

    Returns:
        List of actual file paths (sorted)
    """
    expanded_files = []

    for pattern in patterns:
        # If it's a literal file path that exists, use it directly
        if Path(pattern).exists() and not any(
                c in pattern for c in ['*', '?', '[', ']']):
            expanded_files.append(pattern)
        else:
            # Try to expand as a glob pattern
            matches = glob.glob(pattern)
            if matches:
                expanded_files.extend(matches)
            else:
                # If no matches and it's not a glob pattern, report the file as
                # missing
                if not any(c in pattern for c in ['*', '?', '[', ']']):
                    print(
                        f"Warning: File not found: {pattern}",
                        file=sys.stderr)
                else:
                    print(
                        f"Warning: No files matched pattern: {pattern}",
                        file=sys.stderr)

    # Remove duplicates and sort
    return sorted(set(expanded_files))


def load_json_summary(json_path: str) -> Dict[str, Any]:
    """Load a JSON summary file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_results(json_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Extract per-pass and aggregated correct counts by dataset.

    Returns:
        Dictionary mapping dataset name to results:
        {
            'aime': {
                'run_1': 735,
                'run_2': 740,
                ...
                'pass@k': 875
            }
        }
    """
    pass_k = json_data['pass_k']
    results = defaultdict(dict)
    overall_results = {}

    # Extract per-pass results
    if 'per_pass_results' in json_data:
        for pass_result in json_data['per_pass_results']:
            pass_num = pass_result['pass_number']
            run_label = f"run_{pass_num + 1}"  # Convert 0-indexed to 1-indexed

            # Calculate sum of individual datasets for verification
            sum_correct = 0
            for dataset_stat in pass_result['datasets']:
                dataset_name = dataset_stat['dataset']
                correct = dataset_stat['correct']
                results[dataset_name][run_label] = correct
                sum_correct += correct

            # Extract overall from JSON
            if 'overall' in pass_result:
                overall_correct = pass_result['overall']['correct']
                overall_results[run_label] = overall_correct

                # Assert that the sum matches the overall
                assert sum_correct == overall_correct, (
                    f"Mismatch in {run_label}: sum of datasets ({sum_correct}) != "
                    f"overall ({overall_correct})"
                )

    # Extract aggregated results
    if 'aggregated_results' in json_data:
        # Calculate sum of individual datasets for verification
        sum_correct = 0
        for dataset_stat in json_data['aggregated_results']['datasets']:
            dataset_name = dataset_stat['dataset']
            correct = dataset_stat['correct']
            results[dataset_name][f'pass@{pass_k}'] = correct
            sum_correct += correct

        # Extract overall from JSON
        if 'overall' in json_data['aggregated_results']:
            overall_correct = json_data['aggregated_results']['overall']['correct']
            overall_results[f'pass@{pass_k}'] = overall_correct

            # Assert that the sum matches the overall
            assert sum_correct == overall_correct, (
                f"Mismatch in pass@{pass_k}: sum of datasets ({sum_correct}) != "
                f"overall ({overall_correct})"
            )

    # Handle single-pass results
    elif 'results' in json_data:
        # Calculate sum of individual datasets for verification
        sum_correct = 0
        for dataset_stat in json_data['results']['datasets']:
            dataset_name = dataset_stat['dataset']
            correct = dataset_stat['correct']
            results[dataset_name]['run_1'] = correct
            results[dataset_name]['pass@1'] = correct
            sum_correct += correct

        # Extract overall from JSON if available
        if 'overall' in json_data['results']:
            overall_correct = json_data['results']['overall']['correct']
            overall_results['run_1'] = overall_correct
            overall_results['pass@1'] = overall_correct

            # Assert that the sum matches the overall
            assert sum_correct == overall_correct, (
                f"Mismatch in run_1: sum of datasets ({sum_correct}) != "
                f"overall ({overall_correct})"
            )

    # Add overall results
    if overall_results:
        results['overall'] = overall_results

    return dict(results)


def collect_to_csv(json_files: List[str], output_csv: str,
                   dataset_order: List[str] = None):
    """Collect results from multiple JSON files into a CSV.

    Args:
        json_files: List of JSON file paths
        output_csv: Output CSV file path
        dataset_order: Optional list to specify dataset order (e.g., ['aime', 'gpqa', 'livecodebench'])
    """
    all_results = []
    pass_k = None

    # Load all JSON files
    for json_path in json_files:
        json_data = load_json_summary(json_path)

        # Determine pass@k value
        if pass_k is None:
            pass_k = json_data['pass_k']
        elif pass_k != json_data['pass_k']:
            print(f"Warning: {json_path} has pass@{json_data['pass_k']} but expected pass@{pass_k}",
                  file=sys.stderr)

        # Extract results
        results = extract_results(json_data)
        all_results.append({
            'source_file': json_path,
            'results': results
        })

    if not all_results:
        print("Error: No results to process", file=sys.stderr)
        return

    # Determine column order
    run_columns = [f"run_{i+1}" for i in range(pass_k)]
    pass_column = f"pass@{pass_k}"
    columns = ['dataset'] + run_columns + [pass_column]

    # Collect all unique datasets
    all_datasets = set()
    for result in all_results:
        all_datasets.update(result['results'].keys())

    # Sort datasets (use provided order or alphabetical)
    # Always put 'overall' at the end
    all_datasets_no_overall = all_datasets - {'overall'}

    if dataset_order:
        # Use provided order, put remaining datasets at the end
        sorted_datasets = []
        for ds in dataset_order:
            if ds.lower() in [d.lower() for d in all_datasets_no_overall]:
                # Find the actual dataset name (case-sensitive)
                actual_name = next(
                    d for d in all_datasets_no_overall if d.lower() == ds.lower())
                sorted_datasets.append(actual_name)
        # Add any datasets not in the order list (excluding 'overall')
        remaining = sorted(
            [d for d in all_datasets_no_overall if d not in sorted_datasets])
        sorted_datasets.extend(remaining)
    else:
        sorted_datasets = sorted(all_datasets_no_overall)

    # Add 'overall' at the end if it exists
    if 'overall' in all_datasets:
        sorted_datasets.append('overall')

    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(columns)

        # Write data rows
        for result in all_results:
            for dataset in sorted_datasets:
                if dataset in result['results']:
                    row = [dataset]
                    dataset_results = result['results'][dataset]

                    # Add run columns
                    for run_col in run_columns:
                        row.append(dataset_results.get(run_col, ''))

                    # Add pass@k column
                    row.append(dataset_results.get(pass_column, ''))

                    writer.writerow(row)

    print(f"CSV saved to: {output_csv}")
    print(
        f"Collected {len(all_results)} result sets across {len(sorted_datasets)} datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Collect multiple JSON summaries into a CSV. Supports glob patterns.",
        epilog="Examples:\n"
               "  %(prog)s results_*_summarize.json\n"
               "  %(prog)s data/*.json -o output.csv\n"
               "  %(prog)s run1.json run2.json run3.json --dataset-order aime gpqa livecodebench",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_files", nargs='+',
                        help="One or more JSON files or glob patterns (e.g., '*.json', 'results_*_summarize.json')")
    parser.add_argument("-o", "--output", default="collected_results.csv",
                        help="Output CSV file (default: collected_results.csv)")
    parser.add_argument("--dataset-order", nargs='*',
                        help="Optional dataset order (e.g., aime gpqa livecodebench)")

    args = parser.parse_args()

    # Expand glob patterns
    expanded_files = expand_glob_patterns(args.json_files)

    if not expanded_files:
        print(
            "Error: No JSON files found matching the provided patterns",
            file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(expanded_files)} JSON files:")
    for f in expanded_files:
        print(f"  - {f}")
    print()

    try:
        collect_to_csv(expanded_files, args.output, args.dataset_order)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
