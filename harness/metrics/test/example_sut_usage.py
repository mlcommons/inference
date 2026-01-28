#!/usr/bin/env python3
"""
Example usage script for SUT with integrated metrics collection and visualization.

This script demonstrates how to use the enhanced SUT with metrics collection:
1. Basic usage with metrics collection
2. Advanced configuration options
3. Custom metrics and visualizations
4. Integration with MLPerf Loadgen
"""

import os
import sys
import time
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def example_basic_usage():
    """Example: Basic SUT usage with metrics collection"""
    print("=" * 60)
    print("EXAMPLE: Basic SUT Usage with Metrics Collection")
    print("=" * 60)
    
    print("""
# Basic usage with metrics collection enabled
python SUT_VLLM_SingleReplica.py \\
    --model meta-llama/Llama-3.1-8B \\
    --dataset-path /path/to/dataset.pkl \\
    --enable-metrics-collection \\
    --metrics-output-dir ./metrics_output \\
    --metrics-collection-interval 5
    """)
    
    print("This will:")
    print("• Collect metrics every 5 seconds during execution")
    print("• Save metrics to ./metrics_output/ directory")
    print("• Automatically generate visualizations after completion")
    print("• Create summary reports")

def example_advanced_usage():
    """Example: Advanced SUT usage with custom configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Advanced SUT Usage with Custom Configuration")
    print("=" * 60)
    
    print("""
# Advanced usage with profiling and metrics collection
python SUT_VLLM_SingleReplica.py \\
    --model meta-llama/Llama-3.1-8B \\
    --dataset-path /path/to/dataset.pkl \\
    --scenario Offline \\
    --test-mode performance \\
    --batch-size 32 \\
    --enable-metrics-collection \\
    --metrics-output-dir ./detailed_metrics \\
    --metrics-collection-interval 2 \\
    --enable-profiler \\
    --profiler-dir ./profiler_logs \\
    --enable-nvtx \\
    --print-timing \\
    --log-level DEBUG
    """)
    
    print("This will:")
    print("• Collect metrics every 2 seconds for detailed analysis")
    print("• Enable PyTorch profiling for performance analysis")
    print("• Enable NVTX profiling for GPU analysis")
    print("• Print detailed timing information")
    print("• Generate comprehensive metrics and visualizations")

def example_mlperf_integration():
    """Example: MLPerf Loadgen integration"""
    print("\n" + "=" * 60)
    print("EXAMPLE: MLPerf Loadgen Integration")
    print("=" * 60)
    
    print("""
# MLPerf Loadgen integration with metrics collection
python SUT_VLLM_SingleReplica.py \\
    --model meta-llama/Llama-3.1-8B \\
    --dataset-path /path/to/dataset.pkl \\
    --scenario Offline \\
    --test-mode performance \\
    --num-samples 1000 \\
    --batch-size 32 \\
    --enable-metrics-collection \\
    --metrics-output-dir ./mlperf_metrics \\
    --metrics-collection-interval 1 \\
    --user-conf user.conf \\
    --lg-model-name llama3_1-8b \\
    --output-log-dir ./mlperf_logs
    """)
    
    print("This will:")
    print("• Run MLPerf Loadgen test with metrics collection")
    print("• Collect metrics every second for high-resolution data")
    print("• Generate MLPerf-compliant results")
    print("• Create detailed performance visualizations")

def example_custom_metrics():
    """Example: Custom metrics configuration"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Custom Metrics Configuration")
    print("=" * 60)
    
    print("""
# Custom metrics collection with specific metrics
# (This would require modifying the SUT code to accept custom metrics)

# In the SUT code, you can customize the metrics to collect:
metrics_to_collect = [
    'vllm:num_requests_running',
    'vllm:gpu_utilization', 
    'vllm:gpu_memory_used',
    'vllm:kv_cache_usage_ratio',
    'vllm:request_latency',
    'vllm:generation_tokens_total'
]
    """)
    
    print("Custom metrics can include:")
    print("• GPU utilization and memory usage")
    print("• Request latency and throughput")
    print("• Token generation rates")
    print("• Cache usage statistics")
    print("• Custom application metrics")

def example_visualization_outputs():
    """Example: Expected visualization outputs"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Expected Visualization Outputs")
    print("=" * 60)
    
    print("""
After running with --enable-metrics-collection, you'll get:

./metrics_output/
├── vllm_metrics_20241201_143022.json          # Raw metrics data
├── vllm_metrics_20241201_143022_processed.json # Processed metrics
└── visualizations/
    ├── gpu_utilization_20241201_143022.png    # GPU utilization plot
    ├── requests_running_20241201_143022.png   # Request count plot
    ├── performance_overview_20241201_143022.png # Multi-metric plot
    └── metrics_summary_20241201_143022.json   # Statistical summary
    """)
    
    print("Visualizations include:")
    print("• Time-series plots of key metrics")
    print("• Multi-metric overview charts")
    print("• Statistical summaries in JSON format")
    print("• High-resolution PNG images for reports")

def example_programmatic_usage():
    """Example: Programmatic usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Programmatic Usage")
    print("=" * 60)
    
    print("""
# Programmatic usage in Python scripts
from SUT_VLLM_SingleReplica import VLLMSingleSUT

# Create SUT with metrics collection
sut = VLLMSingleSUT(
    model_name="meta-llama/Llama-3.1-8B",
    dataset_path="/path/to/dataset.pkl",
    test_mode="performance",
    enable_metrics_collection=True,
    metrics_output_dir="./custom_metrics",
    metrics_collection_interval=5
)

# Use with MLPerf Loadgen
import mlperf_loadgen as lg

# Configure test settings
settings = lg.TestSettings()
settings.scenario = lg.TestScenario.Offline
settings.mode = lg.TestMode.PerformanceOnly

# Run test with metrics collection
lg.StartTest(sut.issue_query, qsl, settings)
    """)
    
    print("Programmatic usage allows:")
    print("• Custom integration with existing workflows")
    print("• Automated testing with metrics collection")
    print("• Custom visualization generation")
    print("• Integration with CI/CD pipelines")

def example_troubleshooting():
    """Example: Troubleshooting and debugging"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Troubleshooting and Debugging")
    print("=" * 60)
    
    print("""
# Debug mode for troubleshooting
python SUT_VLLM_SingleReplica.py \\
    --model meta-llama/Llama-3.1-8B \\
    --dataset-path /path/to/dataset.pkl \\
    --enable-metrics-collection \\
    --metrics-output-dir ./debug_metrics \\
    --log-level DEBUG \\
    --print-timing \\
    --print-histogram
    """)
    
    print("Debug features include:")
    print("• Detailed logging of metrics collection")
    print("• Timing information for performance analysis")
    print("• Histogram of input lengths for optimization")
    print("• Error handling and recovery")
    print("• Validation of metrics endpoint connectivity")

def example_performance_analysis():
    """Example: Performance analysis workflow"""
    print("\n" + "=" * 60)
    print("EXAMPLE: Performance Analysis Workflow")
    print("=" * 60)
    
    print("""
# Complete performance analysis workflow

# 1. Run baseline test
python SUT_VLLM_SingleReplica.py \\
    --enable-metrics-collection \\
    --metrics-output-dir ./baseline_metrics \\
    --scenario Offline --test-mode performance

# 2. Run optimized test  
python SUT_VLLM_SingleReplica.py \\
    --enable-metrics-collection \\
    --metrics-output-dir ./optimized_metrics \\
    --scenario Offline --test-mode performance \\
    --batch-size 64  # Different configuration

# 3. Compare results using visualizer
python vllm_metrics_visualizer.py \\
    --file baseline_metrics/vllm_metrics_*.json \\
    --compare-file optimized_metrics/vllm_metrics_*.json \\
    --metrics "vllm:gpu_utilization" "vllm:request_latency" \\
    --label1 "Baseline" --label2 "Optimized" \\
    --save performance_comparison.png
    """)
    
    print("Performance analysis includes:")
    print("• Baseline vs optimized comparisons")
    print("• GPU utilization analysis")
    print("• Latency and throughput metrics")
    print("• Memory usage patterns")
    print("• Automated report generation")

def main():
    """Display all usage examples"""
    print("vLLM SUT METRICS COLLECTION - USAGE EXAMPLES")
    print("=" * 80)
    
    example_basic_usage()
    example_advanced_usage()
    example_mlperf_integration()
    example_custom_metrics()
    example_visualization_outputs()
    example_programmatic_usage()
    example_troubleshooting()
    example_performance_analysis()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The enhanced SUT now includes:")
    print("• Automatic metrics collection during execution")
    print("• Real-time visualization generation")
    print("• Integration with MLPerf Loadgen")
    print("• Comprehensive performance analysis tools")
    print("• Debug and troubleshooting capabilities")
    print("\nFor more information, see the README files in the metrics directory.")

if __name__ == "__main__":
    main()
