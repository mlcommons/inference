#!/usr/bin/env python3
"""
vLLM Metrics Visualizer

A comprehensive visualization tool for vLLM metrics data collected by the metrics collector.
Supports multiple storage formats and provides plotting capabilities for analysis and comparison.
"""

import json
import csv
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
import os
import logging
from pathlib import Path
import numpy as np


class MetricsReader:
    """Base class for reading metrics from different storage formats."""
    
    def read_metrics(self, file_path: str) -> pd.DataFrame:
        """Read metrics from storage and return as DataFrame."""
        raise NotImplementedError
    
    def get_available_metrics(self, file_path: str) -> List[str]:
        """Get list of available metric names in the storage."""
        raise NotImplementedError


class JSONMetricsReader(MetricsReader):
    """Reader for JSON storage format."""
    
    def read_metrics(self, file_path: str) -> pd.DataFrame:
        """Read metrics from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_available_metrics(self, file_path: str) -> List[str]:
        """Get available metric names from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            return []
        
        return list(set([item['metric_name'] for item in data]))


class CSVMetricsReader(MetricsReader):
    """Reader for CSV storage format."""
    
    def read_metrics(self, file_path: str) -> pd.DataFrame:
        """Read metrics from CSV file."""
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Parse labels from JSON string
        df['labels'] = df['labels'].apply(lambda x: json.loads(x) if x else {})
        
        return df
    
    def get_available_metrics(self, file_path: str) -> List[str]:
        """Get available metric names from CSV file."""
        df = pd.read_csv(file_path)
        return df['metric_name'].unique().tolist()


class SQLiteMetricsReader(MetricsReader):
    """Reader for SQLite storage format."""
    
    def read_metrics(self, file_path: str) -> pd.DataFrame:
        """Read metrics from SQLite database."""
        conn = sqlite3.connect(file_path)
        df = pd.read_sql_query("SELECT * FROM metrics", conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['labels'] = df['labels'].apply(lambda x: json.loads(x) if x else {})
        
        return df
    
    def get_available_metrics(self, file_path: str) -> List[str]:
        """Get available metric names from SQLite database."""
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT metric_name FROM metrics")
        metrics = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return metrics


class PrometheusMetricsReader(MetricsReader):
    """Reader for Prometheus format files."""
    
    def read_metrics(self, file_path: str) -> pd.DataFrame:
        """Read metrics from Prometheus format file."""
        metrics = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                # Parse Prometheus format: metric_name{labels} value [timestamp]
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                metric_part = parts[0]
                value = float(parts[1])
                timestamp = None
                
                if len(parts) > 2:
                    try:
                        timestamp = datetime.fromtimestamp(int(parts[2]) / 1000)
                    except:
                        pass
                
                # Parse metric name and labels
                if '{' in metric_part and '}' in metric_part:
                    metric_name = metric_part.split('{')[0]
                    labels_str = metric_part.split('{')[1].rstrip('}')
                    labels = {}
                    
                    # Simple label parsing
                    for label_pair in labels_str.split(','):
                        if '=' in label_pair:
                            key, val = label_pair.split('=', 1)
                            labels[key.strip()] = val.strip().strip('"')
                else:
                    metric_name = metric_part
                    labels = {}
                
                metrics.append({
                    'timestamp': timestamp or datetime.now(),
                    'metric_name': metric_name,
                    'value': value,
                    'labels': labels
                })
        
        df = pd.DataFrame(metrics)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_available_metrics(self, file_path: str) -> List[str]:
        """Get available metric names from Prometheus file."""
        metrics = set()
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                metric_part = parts[0]
                if '{' in metric_part:
                    metric_name = metric_part.split('{')[0]
                else:
                    metric_name = metric_part
                
                metrics.add(metric_name)
        
        return list(metrics)


class VLLMMetricsVisualizer:
    """Main visualizer class for vLLM metrics."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        self.readers = {
            'json': JSONMetricsReader(),
            'csv': CSVMetricsReader(),
            'sqlite': SQLiteMetricsReader(),
            'prometheus': PrometheusMetricsReader()
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _detect_format(self, file_path: str) -> str:
        """Detect storage format from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.json':
            return 'json'
        elif ext == '.csv':
            return 'csv'
        elif ext == '.db':
            return 'sqlite'
        elif ext == '.prom':
            return 'prometheus'
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def load_metrics(self, file_path: str, format_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load metrics from storage file.
        
        Args:
            file_path: Path to metrics file
            format_type: Storage format (auto-detected if None)
            
        Returns:
            DataFrame with metrics data
        """
        if format_type is None:
            format_type = self._detect_format(file_path)
        
        if format_type not in self.readers:
            raise ValueError(f"Unsupported format: {format_type}")
        
        self.logger.info(f"Loading metrics from {file_path} (format: {format_type})")
        return self.readers[format_type].read_metrics(file_path)
    
    def get_available_metrics(self, file_path: str, format_type: Optional[str] = None) -> List[str]:
        """Get list of available metrics in the file."""
        if format_type is None:
            format_type = self._detect_format(file_path)
        
        return self.readers[format_type].get_available_metrics(file_path)
    
    def plot_metric(self, 
                   file_path: str, 
                   metric_name: str, 
                   format_type: Optional[str] = None,
                   title: Optional[str] = None,
                   save_path: Optional[str] = None,
                   show_labels: bool = True) -> None:
        """
        Plot a single metric over time.
        
        Args:
            file_path: Path to metrics file
            metric_name: Name of metric to plot
            format_type: Storage format
            title: Plot title
            save_path: Path to save plot
            show_labels: Whether to show metric labels
        """
        df = self.load_metrics(file_path, format_type)
        
        if df.empty:
            self.logger.warning("No metrics data found")
            return
        
        # Filter for the specific metric
        metric_data = df[df['metric_name'] == metric_name].copy()
        
        if metric_data.empty:
            self.logger.warning(f"Metric '{metric_name}' not found")
            return
        
        # Sort by timestamp
        metric_data = metric_data.sort_values('timestamp')
        
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if show_labels and 'labels' in metric_data.columns:
            # Plot different label combinations separately
            for labels, group in metric_data.groupby('labels'):
                label_str = ', '.join([f"{k}={v}" for k, v in labels.items()]) if labels else 'default'
                ax.plot(group['timestamp'], group['value'], label=label_str, marker='o', markersize=3)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot all data points
            ax.plot(metric_data['timestamp'], metric_data['value'], marker='o', markersize=3)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title or f'{metric_name} Over Time')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_multiple_metrics(self, 
                            file_path: str, 
                            metric_names: List[str], 
                            format_type: Optional[str] = None,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None,
                            subplot_layout: Tuple[int, int] = None) -> None:
        """
        Plot multiple metrics in subplots.
        
        Args:
            file_path: Path to metrics file
            metric_names: List of metric names to plot
            format_type: Storage format
            title: Overall title
            save_path: Path to save plot
            subplot_layout: Layout for subplots (rows, cols)
        """
        df = self.load_metrics(file_path, format_type)
        
        if df.empty:
            self.logger.warning("No metrics data found")
            return
        
        # Calculate subplot layout if not provided
        if subplot_layout is None:
            n_metrics = len(metric_names)
            cols = min(2, n_metrics)
            rows = (n_metrics + cols - 1) // cols
            subplot_layout = (rows, cols)
        
        plt.style.use(self.style)
        fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], 
                                figsize=(self.figsize[0], self.figsize[1] * subplot_layout[0]))
        
        if subplot_layout[0] == 1 and subplot_layout[1] == 1:
            axes = [axes]
        elif subplot_layout[0] == 1 or subplot_layout[1] == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
            
            ax = axes[i]
            metric_data = df[df['metric_name'] == metric_name].copy()
            
            if metric_data.empty:
                ax.text(0.5, 0.5, f'Metric "{metric_name}" not found', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            metric_data = metric_data.sort_values('timestamp')
            
            # Plot with labels if available
            if 'labels' in metric_data.columns:
                for labels, group in metric_data.groupby('labels'):
                    label_str = ', '.join([f"{k}={v}" for k, v in labels.items()]) if labels else 'default'
                    ax.plot(group['timestamp'], group['value'], label=label_str, marker='o', markersize=2)
                
                if len(metric_data['labels'].unique()) > 1:
                    ax.legend(fontsize=8)
            else:
                ax.plot(metric_data['timestamp'], metric_data['value'], marker='o', markersize=2)
            
            ax.set_title(metric_name)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metric_names), len(axes)):
            axes[i].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def compare_metrics(self, 
                       file_path1: str, 
                       file_path2: str, 
                       metric_name: str,
                       format_type1: Optional[str] = None,
                       format_type2: Optional[str] = None,
                       label1: str = "Run 1",
                       label2: str = "Run 2",
                       title: Optional[str] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Compare the same metric from two different runs.
        
        Args:
            file_path1: Path to first metrics file
            file_path2: Path to second metrics file
            metric_name: Name of metric to compare
            format_type1: Format of first file
            format_type2: Format of second file
            label1: Label for first run
            label2: Label for second run
            title: Plot title
            save_path: Path to save plot
        """
        # Load metrics from both files
        df1 = self.load_metrics(file_path1, format_type1)
        df2 = self.load_metrics(file_path2, format_type2)
        
        if df1.empty or df2.empty:
            self.logger.warning("One or both files contain no data")
            return
        
        # Filter for the specific metric
        metric_data1 = df1[df1['metric_name'] == metric_name].copy()
        metric_data2 = df2[df2['metric_name'] == metric_name].copy()
        
        if metric_data1.empty or metric_data2.empty:
            self.logger.warning(f"Metric '{metric_name}' not found in one or both files")
            return
        
        # Sort by timestamp
        metric_data1 = metric_data1.sort_values('timestamp')
        metric_data2 = metric_data2.sort_values('timestamp')
        
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot both runs
        ax.plot(metric_data1['timestamp'], metric_data1['value'], 
               label=label1, marker='o', markersize=3, alpha=0.8)
        ax.plot(metric_data2['timestamp'], metric_data2['value'], 
               label=label2, marker='s', markersize=3, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title or f'{metric_name} Comparison: {label1} vs {label2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def compare_multiple_metrics(self, 
                               file_path1: str, 
                               file_path2: str, 
                               metric_names: List[str],
                               format_type1: Optional[str] = None,
                               format_type2: Optional[str] = None,
                               label1: str = "Run 1",
                               label2: str = "Run 2",
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Compare multiple metrics from two different runs.
        
        Args:
            file_path1: Path to first metrics file
            file_path2: Path to second metrics file
            metric_names: List of metric names to compare
            format_type1: Format of first file
            format_type2: Format of second file
            label1: Label for first run
            label2: Label for second run
            title: Overall title
            save_path: Path to save plot
        """
        # Calculate subplot layout
        n_metrics = len(metric_names)
        cols = min(2, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        plt.style.use(self.style)
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0], self.figsize[1] * rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Load metrics from both files
        df1 = self.load_metrics(file_path1, format_type1)
        df2 = self.load_metrics(file_path2, format_type2)
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Filter for the specific metric
            metric_data1 = df1[df1['metric_name'] == metric_name].copy()
            metric_data2 = df2[df2['metric_name'] == metric_name].copy()
            
            if metric_data1.empty or metric_data2.empty:
                ax.text(0.5, 0.5, f'Metric "{metric_name}" not found', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Sort by timestamp
            metric_data1 = metric_data1.sort_values('timestamp')
            metric_data2 = metric_data2.sort_values('timestamp')
            
            # Plot both runs
            ax.plot(metric_data1['timestamp'], metric_data1['value'], 
                   label=label1, marker='o', markersize=2, alpha=0.8)
            ax.plot(metric_data2['timestamp'], metric_data2['value'], 
                   label=label2, marker='s', markersize=2, alpha=0.8)
            
            ax.set_title(metric_name)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(metric_names), len(axes)):
            axes[i].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, 
                              file_path: str, 
                              format_type: Optional[str] = None,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a summary report of metrics.
        
        Args:
            file_path: Path to metrics file
            format_type: Storage format
            save_path: Path to save report (JSON format)
            
        Returns:
            Dictionary with summary statistics
        """
        df = self.load_metrics(file_path, format_type)
        
        if df.empty:
            return {"error": "No metrics data found"}
        
        summary = {
            "file_path": file_path,
            "total_records": len(df),
            "time_range": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "metrics": {}
        }
        
        # Calculate statistics for each metric
        for metric_name in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric_name]
            
            summary["metrics"][metric_name] = {
                "count": len(metric_data),
                "mean": float(metric_data['value'].mean()),
                "std": float(metric_data['value'].std()),
                "min": float(metric_data['value'].min()),
                "max": float(metric_data['value'].max()),
                "median": float(metric_data['value'].median())
            }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Summary report saved to {save_path}")
        
        return summary


def main():
    """Command line interface for the visualizer."""
    parser = argparse.ArgumentParser(description='vLLM Metrics Visualizer')
    parser.add_argument('--file', required=True, help='Path to metrics file')
    parser.add_argument('--format', choices=['json', 'csv', 'sqlite', 'prometheus'], 
                       help='Storage format (auto-detected if not specified)')
    parser.add_argument('--metric', help='Single metric to plot')
    parser.add_argument('--metrics', nargs='+', help='Multiple metrics to plot')
    parser.add_argument('--compare-file', help='Second file for comparison')
    parser.add_argument('--compare-format', choices=['json', 'csv', 'sqlite', 'prometheus'],
                       help='Format of comparison file')
    parser.add_argument('--label1', default='Run 1', help='Label for first run')
    parser.add_argument('--label2', default='Run 2', help='Label for second run')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--save', help='Path to save plot')
    parser.add_argument('--summary', help='Generate summary report')
    parser.add_argument('--list-metrics', action='store_true', help='List available metrics')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = VLLMMetricsVisualizer()
    
    # List available metrics
    if args.list_metrics:
        metrics = visualizer.get_available_metrics(args.file, args.format)
        print("Available metrics:")
        for metric in sorted(metrics):
            print(f"  - {metric}")
        return
    
    # Generate summary report
    if args.summary:
        summary = visualizer.generate_summary_report(args.file, args.format, args.summary)
        print("Summary Report:")
        print(json.dumps(summary, indent=2))
        return
    
    # Single metric plot
    if args.metric:
        visualizer.plot_metric(
            args.file, args.metric, args.format, args.title, args.save
        )
    
    # Multiple metrics plot
    elif args.metrics:
        if args.compare_file:
            visualizer.compare_multiple_metrics(
                args.file, args.compare_file, args.metrics,
                args.format, args.compare_format, args.label1, args.label2,
                args.title, args.save
            )
        else:
            visualizer.plot_multiple_metrics(
                args.file, args.metrics, args.format, args.title, args.save
            )
    
    # Comparison plot
    elif args.compare_file and args.metric:
        visualizer.compare_metrics(
            args.file, args.compare_file, args.metric,
            args.format, args.compare_format, args.label1, args.label2,
            args.title, args.save
        )
    
    else:
        print("Please specify --metric, --metrics, or --compare-file")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
