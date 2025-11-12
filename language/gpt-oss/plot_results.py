#!/usr/bin/env python3
"""
Generate grouped box plots from collected results CSV.

Creates two plots:
1. Individual runs box plot (run_1, run_2, ..., run_k)
2. Pass@k box plot
"""

import argparse
import sys
import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def load_csv_data(csv_path: str) -> Dict[str, Dict[str, List[int]]]:
    """Load CSV data and organize by dataset.
    
    Returns:
        {
            'aime': {
                'run_1': [735, 752, 765, ...],
                'run_2': [740, 754, 765, ...],
                'pass@5': [875, 875, 885, ...]
            },
            'gpqa': {...},
            ...
        }
    """
    data = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']
            for key, value in row.items():
                if key != 'dataset' and value:
                    try:
                        data[dataset][key].append(int(value))
                    except ValueError:
                        continue
    
    return dict(data)


def create_combined_box_plot(dataset_name: str,
                             dataset_data: Dict[str, List[int]],
                             run_columns: List[str],
                             passk_columns: List[str],
                             output_file: str,
                             ylabel: str = "Correct Count"):
    """Create separate box plots for individual runs and pass@k in the same figure.
    
    Args:
        dataset_name: Name of the dataset
        dataset_data: Data for this dataset (column -> list of values)
        run_columns: Individual run columns to combine (e.g., ['run_1', 'run_2', ...])
        passk_columns: Pass@k columns (e.g., ['pass@5'])
        output_file: Output file path
        ylabel: Y-axis label
    """
    # Combine all individual runs into one list
    all_runs_data = []
    for col in run_columns:
        if col in dataset_data and dataset_data[col]:
            all_runs_data.extend(dataset_data[col])
    
    # Collect pass@k data
    passk_data = []
    for col in passk_columns:
        if col in dataset_data and dataset_data[col]:
            passk_data.extend(dataset_data[col])
    
    if not all_runs_data and not passk_data:
        print(f"Warning: No data to plot for {dataset_name}")
        return
    
    # Determine number of subplots needed
    num_plots = 0
    if all_runs_data:
        num_plots += 1
    if passk_data:
        num_plots += 1
    
    if num_plots == 0:
        print(f"Warning: No data to plot for {dataset_name}")
        return
    
    # Create figure with subplots side by side
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    # Make axes iterable even if there's only one subplot
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot individual runs
    if all_runs_data:
        ax = axes[plot_idx]
        plot_idx += 1
        
        bp = ax.boxplot([all_runs_data], positions=[0], widths=0.5,
                        patch_artist=True, showmeans=True,
                        whis=[0, 100], showfliers=False,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                      markeredgecolor='red', markersize=8))
        
        # Color the box
        bp['boxes'][0].set_facecolor(plt.cm.Set3(0.2))
        bp['boxes'][0].set_alpha(0.7)
        
        # Add scatter plot of individual points
        # Add small random jitter to x-position for visibility
        np.random.seed(42)  # For reproducibility
        x_jitter = np.random.normal(0, 0.04, size=len(all_runs_data))
        ax.scatter(x_jitter, all_runs_data, alpha=0.4, s=30, 
                  color='darkblue', zorder=3, edgecolors='black', linewidth=0.5)
        
        # Set labels
        ax.set_xticks([0])
        ax.set_xticklabels(['Individual Runs'], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset_name} - Individual Runs", fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics
        mean_val = np.mean(all_runs_data)
        std_val = np.std(all_runs_data)
        min_val = np.min(all_runs_data)
        max_val = np.max(all_runs_data)
        n_samples = len(all_runs_data)
        
        stats_text = f"n={n_samples}\nμ={mean_val:.1f}\nσ={std_val:.1f}\nmin={min_val}\nmax={max_val}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Plot pass@k
    if passk_data:
        ax = axes[plot_idx]
        
        passk_label = passk_columns[0] if len(passk_columns) == 1 else 'Pass@k'
        
        bp = ax.boxplot([passk_data], positions=[0], widths=0.5,
                        patch_artist=True, showmeans=True,
                        whis=[0, 100], showfliers=False,
                        meanprops=dict(marker='D', markerfacecolor='red', 
                                      markeredgecolor='red', markersize=8))
        
        # Color the box
        bp['boxes'][0].set_facecolor(plt.cm.Set3(0.6))
        bp['boxes'][0].set_alpha(0.7)
        
        # Add scatter plot of individual points
        # Add small random jitter to x-position for visibility
        np.random.seed(42)  # For reproducibility
        x_jitter = np.random.normal(0, 0.04, size=len(passk_data))
        ax.scatter(x_jitter, passk_data, alpha=0.4, s=30, 
                  color='darkorange', zorder=3, edgecolors='black', linewidth=0.5)
        
        # Set labels
        ax.set_xticks([0])
        ax.set_xticklabels([passk_label], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{dataset_name} - {passk_label}", fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add statistics
        mean_val = np.mean(passk_data)
        std_val = np.std(passk_data)
        min_val = np.min(passk_data)
        max_val = np.max(passk_data)
        n_samples = len(passk_data)
        
        stats_text = f"n={n_samples}\nμ={mean_val:.1f}\nσ={std_val:.1f}\nmin={min_val}\nmax={max_val}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate box plots from collected results CSV")
    parser.add_argument("csv_file",
                        help="Input CSV file from collect_results_csv.py")
    parser.add_argument("-o", "--output-dir", default=".",
                        help="Output directory for plots (default: current directory)")
    parser.add_argument("--prefix", default="boxplot",
                        help="Prefix for output files (default: boxplot)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.csv_file}")
    data = load_csv_data(args.csv_file)
    
    if not data:
        print("Error: No data loaded from CSV", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded data for {len(data)} datasets")
    
    # Determine which columns are runs vs pass@k
    all_columns = set()
    for dataset_data in data.values():
        all_columns.update(dataset_data.keys())
    
    # Separate run columns from pass@k columns
    run_columns = sorted([col for col in all_columns if col.startswith('run_')])
    passk_columns = sorted([col for col in all_columns if col.startswith('pass@')])
    
    if not run_columns and not passk_columns:
        print("Error: No run or pass@k columns found in CSV", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(run_columns)} run columns: {', '.join(run_columns)}")
    print(f"Found {len(passk_columns)} pass@k columns: {', '.join(passk_columns)}")
    print()
    
    # Generate plots for each dataset separately
    datasets = sorted(data.keys())
    
    print(f"Generating plots for {len(datasets)} datasets...\n")
    
    for dataset in datasets:
        dataset_data = data[dataset]
        
        # Create combined plot: Individual Runs (all combined) vs Pass@k
        if run_columns or passk_columns:
            output_file = output_dir / f"{args.prefix}_{dataset}.png"
            print(f"Creating combined box plot for {dataset}...")
            create_combined_box_plot(
                dataset_name=dataset,
                dataset_data=dataset_data,
                run_columns=run_columns,
                passk_columns=passk_columns,
                output_file=str(output_file),
                ylabel="Correct Count"
            )
        
        print()
    
    print("Done!")


if __name__ == "__main__":
    main()

