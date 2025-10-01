#!/usr/bin/env python3
"""
Histogram analysis of token reference output length (OSL) across datasets.
Creates individual histograms per dataset and a master histogram.
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

def create_histograms(df, output_dir="histograms"):
    """Create histograms of tok_model_output_len for each dataset and a master histogram."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create histograms for all data
    _create_histogram_plots(df, output_dir, "individual_datasets_histograms.png", "Token Reference Output Length (OSL) - All Data", "tok_model_output_len")
    
    # Create histograms for data with prompt_accuracy = 100
    df_100_accuracy = df[df['prompt_accuracy'] == 100]
    if len(df_100_accuracy) > 0:
        print(f"\nFiltering for prompt_accuracy = 100: {len(df_100_accuracy)} rows out of {len(df)} total")
        _create_histogram_plots(df_100_accuracy, output_dir, "individual_datasets_histograms_100_accuracy.png", "Token Reference Output Length (OSL) - 100% Accuracy Only", "tok_model_output_len")
    else:
        print("\nNo rows found with prompt_accuracy = 100")

def create_input_histograms(df, output_dir="histograms"):
    """Create histograms of tok_input_len for each dataset and a master histogram."""
    
    # Create histograms for all data
    _create_histogram_plots(df, output_dir, "individual_datasets_input_histograms.png", "Token Input Length (ISL) - All Data", "tok_input_len")
    
    # Create histograms for data with prompt_accuracy = 100
    df_100_accuracy = df[df['prompt_accuracy'] == 100]
    if len(df_100_accuracy) > 0:
        print(f"\nFiltering for prompt_accuracy = 100 (input lengths): {len(df_100_accuracy)} rows out of {len(df)} total")
        _create_histogram_plots(df_100_accuracy, output_dir, "individual_datasets_input_histograms_100_accuracy.png", "Token Input Length (ISL) - 100% Accuracy Only", "tok_input_len")
    else:
        print("\nNo rows found with prompt_accuracy = 100 for input lengths")

def _create_histogram_plots(df, output_dir, filename, title_prefix, column_name):
    """Helper function to create histogram plots."""
    
    # Get unique datasets
    datasets = df['dataset'].unique()
    print(f"Found {len(datasets)} unique datasets: {datasets}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create individual histograms for each dataset
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, dataset in enumerate(datasets):
        if i >= 6:  # Limit to 6 subplots for readability
            break
            
        ax = axes[i]
        dataset_data = df[df['dataset'] == dataset][column_name]
        
        # Create histogram
        ax.hist(dataset_data, bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_title(f'Dataset: {dataset}\n(n={len(dataset_data)})', fontsize=12, fontweight='bold')
        
        # Set appropriate labels based on column type
        if column_name == 'tok_model_output_len':
            ax.set_xlabel('Token Reference Output Length (OSL)', fontsize=10)
        elif column_name == 'tok_input_len':
            ax.set_xlabel('Token Input Length (ISL)', fontsize=10)
        else:
            ax.set_xlabel(column_name, fontsize=10)
            
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = dataset_data.mean()
        median_val = dataset_data.median()
        std_val = dataset_data.std()
        ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(datasets), 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

def create_master_histograms(df, output_dir="histograms"):
    """Create master histogram and overlaid histograms."""
    
    # Create master histogram with all datasets
    plt.figure(figsize=(12, 8))
    
    # Create histogram for all data
    plt.hist(df['tok_model_output_len'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5, color='skyblue')
    plt.title('Master Histogram: Token Reference Output Length (OSL) - All Datasets', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Token Reference Output Length (OSL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add overall statistics
    mean_val = df['tok_model_output_len'].mean()
    median_val = df['tok_model_output_len'].median()
    std_val = df['tok_model_output_len'].std()
    min_val = df['tok_model_output_len'].min()
    max_val = df['tok_model_output_len'].max()
    
    stats_text = f'Total samples: {len(df)}\n'
    stats_text += f'Mean: {mean_val:.1f}\n'
    stats_text += f'Median: {median_val:.1f}\n'
    stats_text += f'Std: {std_val:.1f}\n'
    stats_text += f'Min: {min_val}\n'
    stats_text += f'Max: {max_val}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/master_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed comparison plot with all datasets overlaid
    datasets = df['dataset'].unique()
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]['tok_model_output_len']
        plt.hist(dataset_data, bins=30, alpha=0.6, label=f'{dataset} (n={len(dataset_data)})', 
                color=colors[i], edgecolor='black', linewidth=0.3)
    
    plt.title('Overlaid Histograms: Token Reference Output Length (OSL) by Dataset', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Token Reference Output Length (OSL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overlaid_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Overall statistics for tok_model_output_len:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Std: {std_val:.2f}")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    print(f"  Total samples: {len(df)}")
    
    print(f"\nPer-dataset statistics:")
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]['tok_model_output_len']
        print(f"  {dataset}:")
        print(f"    Count: {len(dataset_data)}")
        print(f"    Mean: {dataset_data.mean():.2f}")
        print(f"    Median: {dataset_data.median():.2f}")
        print(f"    Std: {dataset_data.std():.2f}")
        print(f"    Min: {dataset_data.min()}")
        print(f"    Max: {dataset_data.max()}")
        print()

def create_input_master_histograms(df, output_dir="histograms"):
    """Create master histogram and overlaid histograms for input token lengths."""
    
    # Create master histogram with all datasets
    plt.figure(figsize=(12, 8))
    
    # Create histogram for all data
    plt.hist(df['tok_input_len'], bins=50, alpha=0.7, edgecolor='black', linewidth=0.5, color='lightcoral')
    plt.title('Master Histogram: Token Input Length (ISL) - All Datasets', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Token Input Length (ISL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add overall statistics
    mean_val = df['tok_input_len'].mean()
    median_val = df['tok_input_len'].median()
    std_val = df['tok_input_len'].std()
    min_val = df['tok_input_len'].min()
    max_val = df['tok_input_len'].max()
    
    stats_text = f'Total samples: {len(df)}\n'
    stats_text += f'Mean: {mean_val:.1f}\n'
    stats_text += f'Median: {median_val:.1f}\n'
    stats_text += f'Std: {std_val:.1f}\n'
    stats_text += f'Min: {min_val}\n'
    stats_text += f'Max: {max_val}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/master_input_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed comparison plot with all datasets overlaid
    datasets = df['dataset'].unique()
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_data = df[df['dataset'] == dataset]['tok_input_len']
        plt.hist(dataset_data, bins=30, alpha=0.6, label=f'{dataset} (n={len(dataset_data)})', 
                color=colors[i], edgecolor='black', linewidth=0.3)
    
    plt.title('Overlaid Histograms: Token Input Length (ISL) by Dataset', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Token Input Length (ISL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overlaid_input_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("INPUT LENGTH SUMMARY STATISTICS")
    print("="*60)
    print(f"Overall statistics for tok_input_len:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Std: {std_val:.2f}")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    print(f"  Total samples: {len(df)}")
    
    print(f"\nPer-dataset statistics for input lengths:")
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]['tok_input_len']
        print(f"  {dataset}:")
        print(f"    Count: {len(dataset_data)}")
        print(f"    Mean: {dataset_data.mean():.2f}")
        print(f"    Median: {dataset_data.median():.2f}")
        print(f"    Std: {dataset_data.std():.2f}")
        print(f"    Min: {dataset_data.min()}")
        print(f"    Max: {dataset_data.max()}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Create histograms of token lengths (output and input)')
    parser.add_argument('pkl_path', help='Path to the pickle file')
    parser.add_argument('--output-dir', default='histograms', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.pkl_path)
    
    # Create output length histograms
    print("\n" + "="*60)
    print("CREATING OUTPUT LENGTH HISTOGRAMS")
    print("="*60)
    create_histograms(df, args.output_dir)
    
    # Create master output histograms
    create_master_histograms(df, args.output_dir)
    
    # Create input length histograms
    print("\n" + "="*60)
    print("CREATING INPUT LENGTH HISTOGRAMS")
    print("="*60)
    create_input_histograms(df, args.output_dir)
    
    # Create master input histograms
    create_input_master_histograms(df, args.output_dir)
    
    print(f"\nAll histograms saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
