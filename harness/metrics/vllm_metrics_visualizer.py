#!/usr/bin/env python3
"""
vLLM Metrics Visualizer

A comprehensive visualization tool for vLLM metrics data collected by the metrics collector.
Supports multiple storage formats and provides plotting capabilities for analysis and comparison.

ARCHITECTURE OVERVIEW:
=====================

The visualizer follows a modular, plugin-based architecture designed for extensibility
and maintainability:

1. STORAGE READERS (Plugin Pattern):
   - JSONMetricsReader: Handles JSON format with list of metric objects
   - CSVMetricsReader: Handles CSV format with columns for each field
   - SQLiteMetricsReader: Handles SQLite database with metrics table
   - PrometheusMetricsReader: Handles Prometheus text format
   
   Each reader implements the MetricsReader interface:
   - read_metrics(file_path) -> pd.DataFrame
   - get_available_metrics(file_path) -> List[str]

2. DATA PROCESSING:
   - All readers return normalized DataFrames with consistent structure:
     * timestamp: pandas datetime column
     * metric_name: string column with metric names
     * value: float column with metric values
     * labels: dict column with metric labels (for histogram buckets)
   
   - Special handling for histogram metrics:
     * Detects _bucket, _count, _sum suffixes
     * Groups related histogram components
     * Handles 'le' (less than or equal) labels for buckets

3. VISUALIZATION ENGINE:
   - Single metric plotting with label support
   - Multiple metrics in subplot grids
   - Run comparison capabilities
   - Customizable styling and figure sizing
   - High-resolution output (300 DPI)

4. COMPARISON SYSTEM:
   - Side-by-side metric comparison
   - Support for different file formats in same comparison
   - Customizable run labels
   - Statistical analysis capabilities

KEY ALGORITHMS:
==============

1. FORMAT DETECTION:
   - Auto-detects format from file extension
   - Maps extensions: .json->json, .csv->csv, .db->sqlite, .prom->prometheus
   - Falls back to explicit format specification

2. HISTOGRAM HANDLING:
   - Converts dict labels to string representation for pandas grouping
   - Groups histogram components (buckets, count, sum) by base metric name
   - Preserves label information for proper legend generation

3. SUBPLOT LAYOUT:
   - Calculates optimal grid layout (max 2 columns)
   - Handles single and multiple subplot configurations
   - Automatically hides unused subplots

4. DATA NORMALIZATION:
   - Converts all timestamps to pandas datetime
   - Ensures consistent column names across formats
   - Handles missing or malformed data gracefully

USAGE PATTERNS:
===============

1. BASIC VISUALIZATION:
   visualizer = VLLMMetricsVisualizer()
   visualizer.plot_metric("metrics.json", "vllm:gpu_utilization")

2. COMPARISON ANALYSIS:
   visualizer.compare_metrics("run1.json", "run2.json", "vllm:request_latency")

3. MULTIPLE METRICS:
   visualizer.plot_multiple_metrics("metrics.csv", ["metric1", "metric2"])

4. PROGRAMMATIC ACCESS:
   df = visualizer.load_metrics("metrics.json")
   # Work with pandas DataFrame directly

ERROR HANDLING:
==============

- Graceful handling of missing metrics
- File format validation
- Pandas grouping errors (fixed for dict labels)
- Empty dataset detection
- Invalid file path handling

EXTENSIBILITY:
==============

- Easy to add new storage formats by implementing MetricsReader
- Pluggable styling system via matplotlib styles
- Configurable figure sizing and layout
- Command-line and programmatic interfaces
"""

import json
import csv
import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import argparse
import os
import logging
from pathlib import Path
import numpy as np

# Configure matplotlib for better compatibility
# Try to use a GUI backend, fall back to Agg if needed
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')  # Non-interactive backend

# Set matplotlib to non-interactive mode by default
plt.ioff()


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
    """
    Main visualizer class for vLLM metrics.
    
    This class provides comprehensive visualization capabilities for vLLM metrics data
    collected by the metrics collector. It supports multiple storage formats and offers
    various plotting options including single metrics, multiple metrics, and run comparisons.
    
    The visualizer follows a modular design:
    1. Storage Readers: Handle different file formats (JSON, CSV, SQLite, Prometheus)
    2. Data Processing: Parse and normalize metrics data into pandas DataFrames
    3. Visualization: Create matplotlib plots with customizable styling
    4. Comparison: Support for comparing metrics between different runs
    
    Key Features:
    - Multi-format support: JSON, CSV, SQLite, Prometheus
    - Single and multiple metric plotting
    - Run comparison capabilities
    - Histogram metrics support (buckets, count, sum)
    - Customizable styling and figure sizing
    - High-resolution output (300 DPI)
    - Command-line and programmatic interfaces
    
    Usage:
        # Basic usage
        visualizer = VLLMMetricsVisualizer()
        visualizer.plot_metric("metrics.json", "vllm:gpu_utilization")
        
        # Comparison
        visualizer.compare_metrics("run1.json", "run2.json", "vllm:request_latency")
        
        # Multiple metrics
        visualizer.plot_multiple_metrics("metrics.csv", ["metric1", "metric2"])
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer with custom styling options.
        
        This constructor sets up the visualizer with default styling and creates
        the necessary storage readers for different file formats. The visualizer
        uses a plugin-like architecture where each storage format has its own reader.
        
        Args:
            style: Matplotlib style to use for plots (default: 'seaborn-v0_8')
                  Available styles: 'default', 'seaborn-v0_8', 'seaborn-v0_8-darkgrid', etc.
            figsize: Default figure size as (width, height) tuple (default: (12, 8))
        
        Attributes:
            style (str): Matplotlib style for plots
            figsize (Tuple[int, int]): Default figure dimensions
            readers (Dict[str, MetricsReader]): Storage format readers
            logger (Logger): Logger instance for debugging and info messages
        """
        self.style = style
        self.figsize = figsize
        
        # Initialize storage readers for different formats
        # Each reader implements the MetricsReader interface
        self.readers = {
            'json': JSONMetricsReader(),        # JSON format with list of metric objects
            'csv': CSVMetricsReader(),          # CSV format with columns for each field
            'sqlite': SQLiteMetricsReader(),    # SQLite database with metrics table
            'prometheus': PrometheusMetricsReader()  # Prometheus format with text-based metrics
        }
        
        # Setup logging for debugging and user feedback
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Check if we're in a headless environment
        self.headless = self._is_headless_environment()
        if self.headless:
            self.logger.info("Running in headless environment - plots will be saved to files")
        
        # Load metrics type information
        self.metrics_types = self._load_metrics_types()
    
    def _is_headless_environment(self) -> bool:
        """
        Check if we're running in a headless environment.
        
        Returns:
            bool: True if headless environment detected
        """
        # Check for common headless indicators
        headless_indicators = [
            'DISPLAY' not in os.environ,
            os.environ.get('SSH_CLIENT') is not None,
            os.environ.get('SSH_TTY') is not None,
            matplotlib.get_backend() == 'Agg'
        ]
        return any(headless_indicators)
    
    def _load_metrics_types(self) -> Dict[str, str]:
        """
        Load metrics type information from metrics_info.txt file.
        
        Returns:
            Dict[str, str]: Mapping of metric names to their types (GAUGE, COUNTER, HISTOGRAM, SUMMARY)
        """
        metrics_types = {}
        metrics_info_file = os.path.join(os.path.dirname(__file__), 'metrics_info.txt')
        
        try:
            with open(metrics_info_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#') and ':' in line:
                        # Split on the last colon to handle metric names with colons
                        parts = line.rsplit(':', 1)
                        if len(parts) == 2:
                            metric_name = parts[0].strip()
                            metric_type = parts[1].strip()
                            metrics_types[metric_name] = metric_type
                        else:
                            print(f"Line {line_num}: ERROR - Invalid format: {line}")
            
            self.logger.info(f"Loaded {len(metrics_types)} metric type definitions")
        except FileNotFoundError:
            self.logger.warning(f"Metrics info file not found: {metrics_info_file}")
        except Exception as e:
            self.logger.error(f"Error loading metrics types: {e}")
        
        return metrics_types
    
    def _get_metric_type(self, metric_name: str) -> str:
        """
        Get the type of a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            str: Metric type (GAUGE, COUNTER, HISTOGRAM, SUMMARY)
        """
        # Check for histogram components first
        if metric_name.endswith('_bucket') or metric_name.endswith('_count') or metric_name.endswith('_sum'):
            return 'HISTOGRAM'
        
        # Check in loaded types
        if metric_name in self.metrics_types:
            return self.metrics_types[metric_name]
        
        # Default to GAUGE if not specified
        return 'GAUGE'
    
    
    def _process_histogram_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process histogram metrics (placeholder for future implementation).
        
        Args:
            df: DataFrame with histogram metric data
            
        Returns:
            DataFrame with processed histogram values
        """
        # TODO: Implement histogram processing
        # For now, return as-is
        self.logger.info("Histogram processing not yet implemented - returning raw values")
        return df
    
    def _process_metric_by_type(self, df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        """
        Process metric data based on its type.
        
        Note: Counter delta processing is now handled in the metrics collector,
        so this method only handles histogram processing.
        
        Args:
            df: DataFrame with metric data
            metric_name: Name of the metric
            
        Returns:
            DataFrame with processed metric data
        """
        metric_type = self._get_metric_type(metric_name)
        
        if metric_type == 'HISTOGRAM':
            return self._process_histogram_metric(df)
        else:  # GAUGE, COUNTER, SUMMARY, or unknown - no processing needed
            return df
    
    def _detect_format(self, file_path: str) -> str:
        """
        Detect storage format from file extension.
        
        This helper method automatically determines the storage format based on the
        file extension. This allows users to omit the format_type parameter in most cases.
        
        Algorithm:
        1. Extract file extension using pathlib.Path
        2. Convert to lowercase for case-insensitive matching
        3. Map extension to format type
        4. Raise error for unsupported formats
        
        Args:
            file_path: Path to the metrics file
            
        Returns:
            str: Format type ('json', 'csv', 'sqlite', 'prometheus')
            
        Raises:
            ValueError: If file extension is not supported
        """
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
        Load metrics from storage file and return as pandas DataFrame.
        
        This is the core data loading method that handles all supported storage formats.
        It uses the appropriate reader based on the format type and returns a normalized
        DataFrame with consistent columns: timestamp, metric_name, value, labels.
        
        Algorithm:
        1. Auto-detect format if not specified
        2. Validate format is supported
        3. Use appropriate reader to load data
        4. Return normalized DataFrame
        
        The returned DataFrame has the following structure:
        - timestamp: pandas datetime column
        - metric_name: string column with metric names
        - value: float column with metric values
        - labels: dict column with metric labels (for histogram buckets, etc.)
        
        Args:
            file_path: Path to the metrics file
            format_type: Storage format ('json', 'csv', 'sqlite', 'prometheus')
                        Auto-detected from file extension if None
            
        Returns:
            pd.DataFrame: Normalized metrics data with columns:
                         - timestamp: datetime64[ns]
                         - metric_name: object (string)
                         - value: float64
                         - labels: object (dict)
            
        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
            Exception: If file format is invalid or corrupted
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
        Plot a single metric over time with optional label support.
        
        This method creates a line plot showing how a specific metric changes over time.
        It's particularly useful for monitoring individual metrics like GPU utilization,
        request counts, or latency measurements.
        
        Algorithm:
        1. Load metrics data from specified file
        2. Filter data for the specific metric name
        3. Sort data by timestamp for proper time series display
        4. If labels are present and show_labels=True:
           - Group data by label combinations (e.g., histogram buckets)
           - Plot each label group as separate line with legend
        5. If no labels or show_labels=False:
           - Plot all data points as single line
        6. Apply styling, grid, and labels
        7. Save to file if save_path provided
        
        Special handling for histogram metrics:
        - Histogram metrics (ending in _bucket, _count, _sum) are automatically
          detected and their labels (like 'le' for buckets) are used to create
          separate lines for each bucket threshold
        
        Args:
            file_path: Path to metrics file
            metric_name: Name of metric to plot (e.g., 'vllm:gpu_utilization')
            format_type: Storage format (auto-detected if None)
            title: Custom title for the plot (default: '{metric_name} Over Time')
            save_path: Path to save plot as PNG (optional, 300 DPI)
            show_labels: Whether to show different label combinations as separate lines
                        (useful for histogram buckets with 'le' labels)
        
        Returns:
            None: Displays plot and optionally saves to file
            
        Raises:
            ValueError: If metric not found in data
            FileNotFoundError: If metrics file doesn't exist
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
        
        # Process metric based on its type (counter deltas handled in collector)
        metric_data = self._process_metric_by_type(metric_data, metric_name)
        
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create sample numbers for x-axis (0, 1, 2, ...)
        sample_numbers = range(len(metric_data))
        
        if show_labels and 'labels' in metric_data.columns:
            # Plot different label combinations separately
            for labels, group in metric_data.groupby('labels'):
                label_str = ', '.join([f"{k}={v}" for k, v in labels.items()]) if labels else 'default'
                group_sample_numbers = range(len(group))
                ax.plot(group_sample_numbers, group['value'], label=label_str, marker='o', markersize=3)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Plot all data points
            ax.plot(sample_numbers, metric_data['value'], marker='o', markersize=3)
        
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Value')
        ax.set_title(title or f'{metric_name} Over Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot (default behavior)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            # Generate default filename if no save_path provided
            default_filename = f"{metric_name.replace(':', '_').replace('/', '_')}_plot.png"
            plt.savefig(default_filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {default_filename}")
        
        plt.close()  # Close figure to free memory
    
    def plot_multiple_metrics(self, 
                            file_path: str, 
                            metric_names: List[str], 
                            format_type: Optional[str] = None,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None,
                            subplot_layout: Tuple[int, int] = None) -> None:
        """
        Plot multiple metrics in subplots.
        
        This function creates a grid of subplots, one for each metric specified.
        Each subplot shows the metric's values over time, with support for labeled metrics
        (like histogram buckets) displayed as separate lines.
        
        Algorithm:
        1. Load metrics data from specified file format
        2. Calculate optimal subplot layout (2 columns max)
        3. Create matplotlib subplot grid
        4. For each metric:
           - Filter data for the specific metric
           - Sort by timestamp
           - Group by labels if present (for histogram buckets)
           - Plot each label group as separate line
        5. Apply styling and save if requested
        
        Args:
            file_path: Path to metrics file
            metric_names: List of metric names to plot
            format_type: Storage format (auto-detected if None)
            title: Overall title for the plot
            save_path: Path to save plot (optional)
            subplot_layout: Layout for subplots (rows, cols) - auto-calculated if None
        """
        df = self.load_metrics(file_path, format_type)
        
        if df.empty:
            self.logger.warning("No metrics data found")
            return
        
        # Calculate subplot layout if not provided
        # Algorithm: Use 2 columns max, calculate rows needed
        if subplot_layout is None:
            n_metrics = len(metric_names)
            cols = min(2, n_metrics)
            rows = (n_metrics + cols - 1) // cols
            subplot_layout = (rows, cols)
        
        # Create matplotlib figure with subplots
        plt.style.use(self.style)
        fig, axes = plt.subplots(subplot_layout[0], subplot_layout[1], 
                                figsize=(self.figsize[0], self.figsize[1] * subplot_layout[0]))
        
        # Handle different subplot configurations
        if subplot_layout[0] == 1 and subplot_layout[1] == 1:
            axes = [axes]
        elif subplot_layout[0] == 1 or subplot_layout[1] == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each metric in its own subplot
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
            
            # Create sample numbers for x-axis
            sample_numbers = range(len(metric_data))
            
            # Handle labeled metrics (like histogram buckets with 'le' labels)
            if 'labels' in metric_data.columns and not metric_data['labels'].isna().all():
                # Convert labels to string representation for grouping
                # This fixes the "unhashable type: 'dict'" error
                metric_data['labels_str'] = metric_data['labels'].apply(
                    lambda x: str(sorted(x.items())) if isinstance(x, dict) and x else 'default'
                )
                
                # Group by string representation of labels
                for labels_str, group in metric_data.groupby('labels_str'):
                    # Get original labels for display
                    original_labels = group['labels'].iloc[0] if not group['labels'].isna().iloc[0] else {}
                    label_str = ', '.join([f"{k}={v}" for k, v in original_labels.items()]) if original_labels else 'default'
                    group_sample_numbers = range(len(group))
                    ax.plot(group_sample_numbers, group['value'], label=label_str, marker='o', markersize=2)
                
                # Add legend if multiple label groups exist
                if len(metric_data['labels_str'].unique()) > 1:
                    ax.legend(fontsize=8)
            else:
                # Plot all data points as single line
                ax.plot(sample_numbers, metric_data['value'], marker='o', markersize=2)
            
            # Configure subplot appearance
            ax.set_title(metric_name)
            ax.set_xlabel('Sample Number')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metric_names), len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Adjust layout and save plot
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            # Generate default filename if no save_path provided
            default_filename = f"multiple_metrics_{len(metric_names)}_plots.png"
            plt.savefig(default_filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {default_filename}")
        
        plt.close()  # Close figure to free memory
    
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
        Compare the same metric from two different runs side-by-side.
        
        This method is essential for A/B testing, performance optimization, and
        configuration comparison. It loads metrics from two different files and
        plots them on the same chart with different colors and markers.
        
        Algorithm:
        1. Load metrics from both files (can be different formats)
        2. Filter both datasets for the specified metric
        3. Sort both datasets by timestamp
        4. Plot first run with default styling (blue line, circle markers)
        5. Plot second run with different styling (orange line, square markers)
        6. Add legend to distinguish between runs
        7. Apply grid, labels, and title
        8. Save to file if requested
        
        Use cases:
        - Compare baseline vs optimized configurations
        - Analyze performance before/after changes
        - Compare different model versions
        - Validate performance improvements
        
        Args:
            file_path1: Path to first metrics file
            file_path2: Path to second metrics file
            metric_name: Name of metric to compare (must exist in both files)
            format_type1: Format of first file (auto-detected if None)
            format_type2: Format of second file (auto-detected if None)
            label1: Legend label for first run (default: "Run 1")
            label2: Legend label for second run (default: "Run 2")
            title: Custom plot title (default: '{metric_name} Comparison: {label1} vs {label2}')
            save_path: Path to save comparison plot (optional, 300 DPI)
        
        Returns:
            None: Displays comparison plot and optionally saves to file
            
        Raises:
            ValueError: If metric not found in one or both files
            FileNotFoundError: If either metrics file doesn't exist
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
        
        # Process metrics based on their type (counter deltas handled in collector)
        metric_data1 = self._process_metric_by_type(metric_data1, metric_name)
        metric_data2 = self._process_metric_by_type(metric_data2, metric_name)
        
        # Create sample numbers for x-axis (same range for both)
        max_samples = max(len(metric_data1), len(metric_data2))
        sample_numbers1 = range(len(metric_data1))
        sample_numbers2 = range(len(metric_data2))
        
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot both runs as line graphs
        ax.plot(sample_numbers1, metric_data1['value'], 
               label=label1, marker='o', markersize=3, alpha=0.8, linewidth=2)
        ax.plot(sample_numbers2, metric_data2['value'], 
               label=label2, marker='s', markersize=3, alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Sample Number')
        ax.set_ylabel('Value')
        ax.set_title(title or f'{metric_name} Comparison: {label1} vs {label2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        else:
            # Generate default filename if no save_path provided
            default_filename = f"{metric_name.replace(':', '_').replace('/', '_')}_comparison.png"
            plt.savefig(default_filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {default_filename}")
        
        plt.close()  # Close figure to free memory
    
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
            
            # Process metrics based on their type (counter deltas handled in collector)
            metric_data1 = self._process_metric_by_type(metric_data1, metric_name)
            metric_data2 = self._process_metric_by_type(metric_data2, metric_name)
            
            # Create sample numbers for x-axis
            sample_numbers1 = range(len(metric_data1))
            sample_numbers2 = range(len(metric_data2))
            
            # Plot both runs as line graphs
            ax.plot(sample_numbers1, metric_data1['value'], 
                   label=label1, marker='o', markersize=2, alpha=0.8, linewidth=1.5)
            ax.plot(sample_numbers2, metric_data2['value'], 
                   label=label2, marker='s', markersize=2, alpha=0.8, linewidth=1.5)
            
            ax.set_title(metric_name)
            ax.set_xlabel('Sample Number')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metric_names), len(axes)):
            axes[i].set_visible(False)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        else:
            # Generate default filename if no save_path provided
            default_filename = f"multiple_metrics_comparison_{len(metric_names)}_plots.png"
            plt.savefig(default_filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {default_filename}")
        
        plt.close()  # Close figure to free memory
    
    def generate_summary_report(self, 
                              file_path: str, 
                              format_type: Optional[str] = None,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of metrics with statistical analysis.
        
        This method provides a statistical overview of all metrics in the dataset,
        including basic statistics (mean, std, min, max, median) and time range
        information. It's useful for quick analysis and reporting.
        
        Algorithm:
        1. Load metrics data from specified file
        2. Calculate overall statistics (total records, time range)
        3. For each unique metric:
           - Calculate count, mean, standard deviation
           - Find minimum, maximum, and median values
           - Store statistics in structured format
        4. Optionally save report to JSON file
        5. Return comprehensive summary dictionary
        
        The summary report includes:
        - File information and total record count
        - Time range (start and end timestamps)
        - Per-metric statistics (count, mean, std, min, max, median)
        
        Args:
            file_path: Path to metrics file
            format_type: Storage format (auto-detected if None)
            save_path: Path to save JSON report (optional)
            
        Returns:
            Dict[str, Any]: Summary report with structure:
                {
                    "file_path": str,
                    "total_records": int,
                    "time_range": {
                        "start": str (ISO format),
                        "end": str (ISO format)
                    },
                    "metrics": {
                        "metric_name": {
                            "count": int,
                            "mean": float,
                            "std": float,
                            "min": float,
                            "max": float,
                            "median": float
                        }
                    }
                }
        
        Raises:
            FileNotFoundError: If metrics file doesn't exist
            Exception: If file format is invalid
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
    parser.add_argument('--save', help='Path to save plot (auto-generated if not specified)')
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
