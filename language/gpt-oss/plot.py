#!/usr/bin/env python3
"""
Histogram analysis of token input length (ISL) and output length (OSL) across datasets.
Creates 8 histograms as specified.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


def load_data(pkl_path):
    """Load the pickle file and return the DataFrame."""
    print(f"Loading data from {pkl_path}...")
    df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def create_per_dataset_histogram(df, column_name, title, filename, output_dir):
    """Create individual histograms for each dataset."""
    datasets = sorted(df['dataset'].unique())
    print(f"Creating {filename}...")
    print(f"  Datasets: {datasets}")
    print(f"  Total samples: {len(df)}")

    # Determine grid layout
    n_datasets = len(datasets)
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_data = df[df['dataset'] == dataset][column_name]

        # Create histogram
        ax.hist(
            dataset_data,
            bins=30,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            color='skyblue' if 'OSL' in title else 'lightcoral')

        ax.set_title(
            f'{dataset}\n(n={len(dataset_data)})',
            fontsize=12,
            fontweight='bold')
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = dataset_data.mean()
        median_val = dataset_data.median()
        std_val = dataset_data.std()
        stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)

    # Hide unused subplots
    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_dir}/{filename}")
    plt.close()


def create_full_histogram(df, column_name, title, filename, output_dir):
    """Create a single histogram combining all datasets."""
    print(f"Creating {filename}...")
    print(f"  Total samples: {len(df)}")

    plt.figure(figsize=(12, 8))

    color = 'skyblue' if 'OSL' in title else 'lightcoral'
    plt.hist(
        df[column_name],
        bins=50,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        color=color)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(
        column_name.replace(
            'tok_',
            '').replace(
            '_len',
            '').upper(),
        fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_val = df[column_name].mean()
    median_val = df[column_name].median()
    std_val = df[column_name].std()
    min_val = df[column_name].min()
    max_val = df[column_name].max()

    stats_text = f'Total samples: {len(df)}\n'
    stats_text += f'Mean: {mean_val:.1f}\n'
    stats_text += f'Median: {median_val:.1f}\n'
    stats_text += f'Std: {std_val:.1f}\n'
    stats_text += f'Min: {min_val}\n'
    stats_text += f'Max: {max_val}'

    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue' if 'OSL' in title else 'lightcoral', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    print(f"  Saved to {output_dir}/{filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create histograms of token lengths (ISL and OSL)')
    parser.add_argument('pkl_path', help='Path to the pickle file')
    parser.add_argument(
        '--output-dir',
        default='histograms',
        help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Load data
    df = load_data(args.pkl_path)

    # Check if dataset column exists
    has_dataset = 'dataset' in df.columns
    if not has_dataset:
        print("\nNote: 'dataset' column not found - skipping per-dataset histograms")
        # Add a dummy dataset column for compatibility with existing code
        df['dataset'] = 'default'

    # Check if prompt_accuracy column exists
    has_accuracy = 'prompt_accuracy' in df.columns
    
    if has_accuracy:
        # Filter for 100% accuracy
        df_100 = df[df['prompt_accuracy'] == 100.0].copy()
        print(
            f"\nFiltered {len(df_100)} rows with prompt_accuracy == 100 (out of {len(df)} total)\n")
    else:
        print("\nNote: 'prompt_accuracy' column not found - skipping accuracy-based histograms\n")
        # Create empty dataframe with dataset column for consistency
        df_100 = pd.DataFrame(columns=df.columns)

    print("=" * 60)
    print("CREATING ISL HISTOGRAMS")
    print("=" * 60)

    # 1. Per dataset ISL histogram
    if has_dataset:
        create_per_dataset_histogram(
            df, 'tok_input_len',
            'Token Input Length (ISL)',
            '1_per_dataset_ISL.png',
            args.output_dir)
    else:
        print("Skipping per-dataset ISL: dataset column not found")

    # 2. Per dataset ISL histogram (accuracy == 100)
    if has_dataset and has_accuracy and len(df_100) > 0:
        create_per_dataset_histogram(
            df_100, 'tok_input_len',
            'Token Input Length (ISL) - 100% Accuracy',
            '2_per_dataset_ISL_acc100.png',
            args.output_dir)
    elif not has_dataset:
        print("Skipping per-dataset ISL (acc==100): dataset column not found")
    elif not has_accuracy:
        print("Skipping per-dataset ISL (acc==100): prompt_accuracy column not found")
    else:
        print("Skipping per-dataset ISL (acc==100): no data with 100% accuracy")

    # 3. Full ISL histogram
    create_full_histogram(
        df, 'tok_input_len',
        'Token Input Length (ISL) - All Data',
        '3_full_ISL.png',
        args.output_dir)

    # 4. Full ISL histogram (accuracy == 100)
    if has_accuracy and len(df_100) > 0:
        create_full_histogram(
            df_100, 'tok_input_len',
            'Token Input Length (ISL) - 100% Accuracy',
            '4_full_ISL_acc100.png',
            args.output_dir)
    elif has_accuracy:
        print("Skipping full ISL (acc==100): no data with 100% accuracy")
    else:
        print("Skipping full ISL (acc==100): prompt_accuracy column not found")

    print("\n" + "=" * 60)
    print("CREATING OSL HISTOGRAMS")
    print("=" * 60)

    # 5. Per dataset OSL histogram
    if has_dataset:
        create_per_dataset_histogram(
            df, 'tok_model_output_len',
            'Token Output Length (OSL)',
            '5_per_dataset_OSL.png',
            args.output_dir)
    else:
        print("Skipping per-dataset OSL: dataset column not found")

    # 6. Per dataset OSL histogram (accuracy == 100)
    if has_dataset and has_accuracy and len(df_100) > 0:
        create_per_dataset_histogram(
            df_100, 'tok_model_output_len',
            'Token Output Length (OSL) - 100% Accuracy',
            '6_per_dataset_OSL_acc100.png',
            args.output_dir)
    elif not has_dataset:
        print("Skipping per-dataset OSL (acc==100): dataset column not found")
    elif not has_accuracy:
        print("Skipping per-dataset OSL (acc==100): prompt_accuracy column not found")
    else:
        print("Skipping per-dataset OSL (acc==100): no data with 100% accuracy")

    # 7. Full OSL histogram
    create_full_histogram(
        df, 'tok_model_output_len',
        'Token Output Length (OSL) - All Data',
        '7_full_OSL.png',
        args.output_dir)

    # 8. Full OSL histogram (accuracy == 100)
    if has_accuracy and len(df_100) > 0:
        create_full_histogram(
            df_100, 'tok_model_output_len',
            'Token Output Length (OSL) - 100% Accuracy',
            '8_full_OSL_acc100.png',
            args.output_dir)
    elif has_accuracy:
        print("Skipping full OSL (acc==100): no data with 100% accuracy")
    else:
        print("Skipping full OSL (acc==100): prompt_accuracy column not found")

    print(f"\n{'=' * 60}")
    print(f"All histograms saved to {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
