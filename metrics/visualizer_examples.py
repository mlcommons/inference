#!/usr/bin/env python3
"""
Examples for using the vLLM Metrics Visualizer

This script demonstrates various ways to use the VLLMMetricsVisualizer class
for plotting and analyzing vLLM metrics data.
"""

from vllm_metrics_visualizer import VLLMMetricsVisualizer
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_single_metric_plot():
    """Example: Plot a single metric over time."""
    print("=== Single Metric Plot Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Plot a single metric
    visualizer.plot_metric(
        file_path="vllm_metrics.json",
        metric_name="vllm:num_requests_running",
        title="Number of Running Requests Over Time",
        save_path="single_metric_plot.png"
    )


def example_multiple_metrics_plot():
    """Example: Plot multiple metrics in subplots."""
    print("\n=== Multiple Metrics Plot Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Plot multiple metrics
    metrics_to_plot = [
        "vllm:num_requests_running",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:gpu_utilization"
    ]
    
    visualizer.plot_multiple_metrics(
        file_path="vllm_metrics.csv",
        metric_names=metrics_to_plot,
        title="vLLM Performance Metrics",
        save_path="multiple_metrics_plot.png"
    )


def example_compare_two_runs():
    """Example: Compare metrics from two different runs."""
    print("\n=== Compare Two Runs Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Compare a single metric between two runs
    visualizer.compare_metrics(
        file_path1="run1_metrics.json",
        file_path2="run2_metrics.json",
        metric_name="vllm:request_latency",
        label1="Baseline Run",
        label2="Optimized Run",
        title="Request Latency Comparison",
        save_path="latency_comparison.png"
    )


def example_compare_multiple_metrics():
    """Example: Compare multiple metrics from two runs."""
    print("\n=== Compare Multiple Metrics Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Compare multiple metrics between two runs
    metrics_to_compare = [
        "vllm:num_requests_running",
        "vllm:generation_tokens_total",
        "vllm:gpu_utilization",
        "vllm:gpu_memory_used"
    ]
    
    visualizer.compare_multiple_metrics(
        file_path1="baseline_metrics.csv",
        file_path2="optimized_metrics.csv",
        metric_names=metrics_to_compare,
        label1="Baseline Configuration",
        label2="Optimized Configuration",
        title="Performance Comparison: Baseline vs Optimized",
        save_path="performance_comparison.png"
    )


def example_prometheus_metrics():
    """Example: Work with Prometheus format metrics."""
    print("\n=== Prometheus Metrics Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Plot histogram metrics from Prometheus format
    visualizer.plot_metric(
        file_path="vllm_metrics.prom",
        metric_name="vllm_request_duration_seconds_bucket",
        format_type="prometheus",
        title="Request Duration Histogram Buckets",
        save_path="histogram_buckets.png"
    )


def example_sqlite_metrics():
    """Example: Work with SQLite database metrics."""
    print("\n=== SQLite Metrics Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Plot metrics from SQLite database
    visualizer.plot_multiple_metrics(
        file_path="metrics.db",
        metric_names=["vllm:num_requests_running", "vllm:gpu_utilization"],
        format_type="sqlite",
        title="Metrics from SQLite Database",
        save_path="sqlite_metrics.png"
    )


def example_generate_summary():
    """Example: Generate summary report of metrics."""
    print("\n=== Summary Report Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Generate summary report
    summary = visualizer.generate_summary_report(
        file_path="vllm_metrics.json",
        save_path="metrics_summary.json"
    )
    
    print("Summary Report:")
    for metric, stats in summary.get("metrics", {}).items():
        print(f"\n{metric}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Std: {stats['std']:.2f}")


def example_list_available_metrics():
    """Example: List available metrics in a file."""
    print("\n=== List Available Metrics Example ===")
    
    visualizer = VLLMMetricsVisualizer()
    
    # List available metrics
    metrics = visualizer.get_available_metrics("vllm_metrics.json")
    print("Available metrics:")
    for metric in sorted(metrics):
        print(f"  - {metric}")


def example_custom_styling():
    """Example: Use custom styling and figure size."""
    print("\n=== Custom Styling Example ===")
    
    # Create visualizer with custom styling
    visualizer = VLLMMetricsVisualizer(
        style='seaborn-v0_8-darkgrid',
        figsize=(15, 10)
    )
    
    visualizer.plot_metric(
        file_path="vllm_metrics.csv",
        metric_name="vllm:gpu_utilization",
        title="GPU Utilization with Custom Styling",
        save_path="custom_styled_plot.png"
    )


def example_programmatic_usage():
    """Example: Programmatic usage in your own applications."""
    print("\n=== Programmatic Usage Example ===")
    
    # Create visualizer instance
    visualizer = VLLMMetricsVisualizer()
    
    # Load metrics data
    df = visualizer.load_metrics("vllm_metrics.json")
    
    if not df.empty:
        print(f"Loaded {len(df)} metric records")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Available metrics: {df['metric_name'].unique().tolist()}")
        
        # You can now work with the DataFrame directly
        # For example, filter for specific metrics
        gpu_metrics = df[df['metric_name'] == 'vllm:gpu_utilization']
        if not gpu_metrics.empty:
            avg_gpu_util = gpu_metrics['value'].mean()
            print(f"Average GPU utilization: {avg_gpu_util:.2f}%")


if __name__ == "__main__":
    print("vLLM Metrics Visualizer - Examples")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to test)
    
    # Basic plotting examples
    # example_single_metric_plot()
    # example_multiple_metrics_plot()
    
    # Comparison examples
    # example_compare_two_runs()
    # example_compare_multiple_metrics()
    
    # Different storage formats
    # example_prometheus_metrics()
    # example_sqlite_metrics()
    
    # Analysis examples
    # example_generate_summary()
    # example_list_available_metrics()
    
    # Customization examples
    # example_custom_styling()
    # example_programmatic_usage()
    
    print("\nAll examples completed!")
    print("\nKey features demonstrated:")
    print("- Single and multiple metric plotting")
    print("- Comparison between different runs")
    print("- Support for all storage formats (JSON, CSV, SQLite, Prometheus)")
    print("- Summary report generation")
    print("- Custom styling and figure sizing")
    print("- Programmatic DataFrame access")
