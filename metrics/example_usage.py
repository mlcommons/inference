#!/usr/bin/env python3
"""
Example usage of the vLLM Metrics Collector

This script demonstrates how to use the VLLMMetricsCollector class
programmatically in your own applications.
"""

from vllm_metrics_collector import VLLMMetricsCollector, JSONStorage, CSVStorage, SQLiteStorage, PrometheusStorage, PrometheusFileStorage
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example with JSON storage."""
    print("=== Basic Usage Example ===")
    
    # Create storage backend
    storage = JSONStorage("example_metrics.json")
    
    # Define metrics to collect
    metrics_to_collect = [
        'vllm:num_requests_running',
        'vllm:generation_tokens_total',
        'vllm:request_success_total'
    ]
    
    # Create collector
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=metrics_to_collect,
        collection_interval=5,  # Collect every 5 seconds
        timeout=10
    )
    
    try:
        # Start collection
        collector.start()
        print("Metrics collection started. Press Ctrl+C to stop.")
        
        # Let it run for 30 seconds
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("Stopping collection...")
    finally:
        collector.stop()
        print("Collection stopped. Check example_metrics.json for results.")


def example_custom_metrics():
    """Example with custom metrics and CSV storage."""
    print("\n=== Custom Metrics Example ===")
    
    # Create CSV storage
    storage = CSVStorage("custom_metrics.csv")
    
    # Custom metrics list
    custom_metrics = [
        'vllm:request_latency',
        'vllm:request_input_tokens',
        'vllm:request_output_tokens',
        'vllm:gpu_utilization',
        'vllm:gpu_memory_used'
    ]
    
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=custom_metrics,
        collection_interval=2,  # Collect every 2 seconds
        timeout=15
    )
    
    try:
        collector.start()
        print("Custom metrics collection started. Press Ctrl+C to stop.")
        time.sleep(20)
    except KeyboardInterrupt:
        print("Stopping custom collection...")
    finally:
        collector.stop()
        print("Custom collection stopped. Check custom_metrics.csv for results.")


def example_database_storage():
    """Example with SQLite database storage."""
    print("\n=== Database Storage Example ===")
    
    # Create SQLite storage
    storage = SQLiteStorage("metrics_database.db")
    
    # All available vLLM metrics
    all_metrics = [
        'vllm:num_requests_running',
        'vllm:generation_tokens_total',
        'vllm:request_success_total',
        'vllm:request_failure_total',
        'vllm:request_latency',
        'vllm:request_input_tokens',
        'vllm:request_output_tokens',
        'vllm:gpu_utilization',
        'vllm:gpu_memory_used',
        'vllm:gpu_memory_total',
        'vllm:kv_cache_usage_ratio',
        'vllm:num_requests_waiting',
        'vllm:num_requests_finished',
        'vllm:num_requests_cancelled'
    ]
    
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=all_metrics,
        collection_interval=3,
        timeout=20
    )
    
    try:
        collector.start()
        print("Database storage collection started. Press Ctrl+C to stop.")
        time.sleep(25)
    except KeyboardInterrupt:
        print("Stopping database collection...")
    finally:
        collector.stop()
        print("Database collection stopped. Check metrics_database.db for results.")


def example_prometheus_file_storage():
    """Example with Prometheus file storage."""
    print("\n=== Prometheus File Storage Example ===")
    
    # Create Prometheus file storage
    storage = PrometheusFileStorage("prometheus_metrics.prom")
    
    # Define metrics to collect
    prometheus_metrics = [
        'vllm:num_requests_running',
        'vllm:generation_tokens_total',
        'vllm:request_success_total',
        'vllm:gpu_utilization',
        'vllm:gpu_memory_used'
    ]
    
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=prometheus_metrics,
        collection_interval=5,
        timeout=15
    )
    
    try:
        collector.start()
        print("Prometheus file collection started. Press Ctrl+C to stop.")
        time.sleep(25)
    except KeyboardInterrupt:
        print("Stopping Prometheus file collection...")
    finally:
        collector.stop()
        print("Prometheus file collection stopped. Check prometheus_metrics.prom for results.")


def example_prometheus_pushgateway():
    """Example with Prometheus Pushgateway."""
    print("\n=== Prometheus Pushgateway Example ===")
    
    # Create Prometheus storage with Pushgateway
    storage = PrometheusStorage(
        output_path="prometheus_metrics.prom",  # Also save to file
        pushgateway_url="http://localhost:9091",  # Pushgateway URL
        job_name="vllm_metrics_collector",
        instance="instance_001"
    )
    
    # All vLLM metrics
    all_metrics = [
        'vllm:num_requests_running',
        'vllm:generation_tokens_total',
        'vllm:request_success_total',
        'vllm:request_failure_total',
        'vllm:request_latency',
        'vllm:gpu_utilization',
        'vllm:gpu_memory_used'
    ]
    
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=all_metrics,
        collection_interval=10,
        timeout=20
    )
    
    try:
        collector.start()
        print("Prometheus Pushgateway collection started. Press Ctrl+C to stop.")
        print("Metrics will be pushed to Pushgateway and saved to file.")
        time.sleep(30)
    except KeyboardInterrupt:
        print("Stopping Prometheus Pushgateway collection...")
    finally:
        collector.stop()
        print("Prometheus Pushgateway collection stopped. Check Pushgateway and prometheus_metrics.prom for results.")


if __name__ == "__main__":
    print("vLLM Metrics Collector - Example Usage")
    print("=" * 50)
    
    # Run examples (uncomment the ones you want to test)
    
    # Basic usage
    example_basic_usage()
    
    # Custom metrics
    # example_custom_metrics()
    
    # Database storage
    # example_database_storage()
    
    # Prometheus file storage
    # example_prometheus_file_storage()
    
    # Prometheus Pushgateway
    # example_prometheus_pushgateway()
    
    print("\nAll examples completed!")
