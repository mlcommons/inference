#!/usr/bin/env python3
"""
Test script to demonstrate histogram metrics handling in vLLM Metrics Collector.

This script shows how the collector properly handles Prometheus histogram metrics
including buckets, count, and sum components.
"""

from vllm_metrics_collector import VLLMMetricsCollector, PrometheusFileStorage
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_histogram_parsing():
    """Test histogram metrics parsing with sample data."""
    print("=== Testing Histogram Metrics Parsing ===")
    
    # Sample Prometheus metrics text with histogram data
    sample_metrics = """
# HELP vllm_request_duration_seconds Histogram of request durations
# TYPE vllm_request_duration_seconds histogram
vllm_request_duration_seconds_bucket{le="0.1"} 0
vllm_request_duration_seconds_bucket{le="0.5"} 5
vllm_request_duration_seconds_bucket{le="1.0"} 10
vllm_request_duration_seconds_bucket{le="2.5"} 15
vllm_request_duration_seconds_bucket{le="5.0"} 18
vllm_request_duration_seconds_bucket{le="+Inf"} 20
vllm_request_duration_seconds_count 20
vllm_request_duration_seconds_sum 25.5

# HELP vllm_request_ttft_seconds Histogram of time to first token
# TYPE vllm_request_ttft_seconds histogram
vllm_request_ttft_seconds_bucket{le="0.01"} 0
vllm_request_ttft_seconds_bucket{le="0.05"} 3
vllm_request_ttft_seconds_bucket{le="0.1"} 8
vllm_request_ttft_seconds_bucket{le="0.5"} 12
vllm_request_ttft_seconds_bucket{le="1.0"} 15
vllm_request_ttft_seconds_bucket{le="+Inf"} 18
vllm_request_ttft_seconds_count 18
vllm_request_ttft_seconds_sum 2.1

# Regular counter metric
vllm_num_requests_running 5
vllm_generation_tokens_total 1500
"""
    
    # Create a test collector
    storage = PrometheusFileStorage("test_histogram_metrics.prom")
    
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=[
            'vllm:request_duration',
            'vllm:request_ttft',
            'vllm:num_requests_running',
            'vllm:generation_tokens_total'
        ],
        collection_interval=1,
        timeout=5
    )
    
    # Test the parsing directly
    parsed_metrics = collector.parse_metrics(sample_metrics)
    
    print(f"Parsed {len(parsed_metrics)} metrics:")
    for metric in parsed_metrics:
        print(f"  {metric.metric_name}: {metric.value} (labels: {metric.labels})")
    
    # Store metrics to see the output format
    for metric in parsed_metrics:
        storage.store_metric(metric)
    
    storage.close()
    print("\nHistogram metrics written to test_histogram_metrics.prom")
    print("Check the file to see proper Prometheus histogram format!")


def test_histogram_collection():
    """Test histogram metrics collection from real vLLM endpoint."""
    print("\n=== Testing Real Histogram Collection ===")
    
    # Create Prometheus file storage
    storage = PrometheusFileStorage("real_histogram_metrics.prom")
    
    # Define histogram metrics to collect
    histogram_metrics = [
        'vllm:request_duration',
        'vllm:request_ttft',
        'vllm:request_itl',
        'vllm:num_requests_running',
        'vllm:generation_tokens_total'
    ]
    
    collector = VLLMMetricsCollector(
        metrics_endpoint='http://localhost:8000/metrics',
        storage=storage,
        metrics_to_collect=histogram_metrics,
        collection_interval=5,
        timeout=15
    )
    
    try:
        collector.start()
        print("Histogram metrics collection started. Press Ctrl+C to stop.")
        print("This will collect histogram metrics including buckets, count, and sum.")
        time.sleep(30)
    except KeyboardInterrupt:
        print("Stopping histogram collection...")
    finally:
        collector.stop()
        print("Histogram collection stopped. Check real_histogram_metrics.prom for results.")


def demonstrate_histogram_queries():
    """Demonstrate PromQL queries for histogram metrics."""
    print("\n=== PromQL Queries for Histogram Metrics ===")
    
    queries = {
        "95th percentile request duration": 
            "histogram_quantile(0.95, sum(rate(vllm_request_duration_seconds_bucket[5m])) by (le))",
        
        "Average request duration":
            "sum(rate(vllm_request_duration_seconds_sum[5m])) / sum(rate(vllm_request_duration_seconds_count[5m]))",
        
        "Request rate":
            "sum(rate(vllm_request_duration_seconds_count[5m]))",
        
        "95th percentile TTFT":
            "histogram_quantile(0.95, sum(rate(vllm_request_ttft_seconds_bucket[5m])) by (le))",
        
        "Average TTFT":
            "sum(rate(vllm_request_ttft_seconds_sum[5m])) / sum(rate(vllm_request_ttft_seconds_count[5m]))"
    }
    
    print("Use these PromQL queries in Grafana or Prometheus:")
    for description, query in queries.items():
        print(f"\n{description}:")
        print(f"  {query}")


if __name__ == "__main__":
    print("vLLM Histogram Metrics Test")
    print("=" * 50)
    
    # Test histogram parsing with sample data
    test_histogram_parsing()
    
    # Demonstrate PromQL queries
    demonstrate_histogram_queries()
    
    # Test real collection (uncomment if vLLM is running)
    # test_histogram_collection()
    
    print("\nHistogram metrics test completed!")
    print("\nKey benefits of histogram support:")
    print("- Proper bucket, count, and sum metric handling")
    print("- Accurate percentile calculations")
    print("- Standard Prometheus histogram format")
    print("- Compatible with Grafana and Prometheus tooling")
