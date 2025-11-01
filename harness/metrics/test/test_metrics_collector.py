#!/usr/bin/env python3
"""
Test script for vLLM Metrics Collector functionality.

This script tests the metrics collector with various configurations including:
- Basic metrics collection
- CSV and JSON storage formats
- Auto-postprocessing
- Debug mode
- Error handling
"""

import os
import sys
import time
import threading
import requests
from unittest.mock import Mock, patch
import tempfile
import json
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_metrics_collector import (
    VLLMMetricsCollector, 
    JSONStorage, 
    CSVStorage, 
    SQLiteStorage,
    PrometheusFileStorage,
    MetricData
)

class MockMetricsServer:
    """Mock vLLM metrics server for testing"""
    
    def __init__(self, port=8000):
        self.port = port
        self.url = f"http://localhost:{port}/metrics"
        self.server = None
        self.thread = None
        self.running = False
        
    def start(self):
        """Start the mock server"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    
                    # Mock Prometheus metrics
                    metrics = """# HELP vllm_num_requests_running Number of requests currently running
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running 5.0
# HELP vllm_generation_tokens_total Total number of tokens generated
# TYPE vllm_generation_tokens_total counter
vllm_generation_tokens_total 1234.0
# HELP vllm_request_latency Request latency
# TYPE vllm_request_latency histogram
vllm_request_latency_bucket{le="0.1"} 10.0
vllm_request_latency_bucket{le="0.5"} 25.0
vllm_request_latency_bucket{le="1.0"} 30.0
vllm_request_latency_bucket{le="+Inf"} 30.0
vllm_request_latency_count 30.0
vllm_request_latency_sum 15.0
# HELP vllm_gpu_utilization GPU utilization percentage
# TYPE vllm_gpu_utilization gauge
vllm_gpu_utilization 85.5
"""
                    self.wfile.write(metrics.encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        self.server = HTTPServer(('localhost', self.port), MetricsHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        self.running = True
        
        # Wait for server to start
        time.sleep(0.5)
        
    def stop(self):
        """Stop the mock server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self.running = False

def test_basic_metrics_collection():
    """Test basic metrics collection functionality"""
    print("Testing basic metrics collection...")
    
    # Create temporary storage
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        storage_file = f.name
    
    try:
        # Create storage and collector
        storage = JSONStorage(storage_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:8000/metrics",
            storage=storage,
            metrics_to_collect=['vllm:num_requests_running', 'vllm:gpu_utilization'],
            collection_interval=1,
            timeout=5
        )
        
        # Start mock server
        mock_server = MockMetricsServer()
        mock_server.start()
        
        # Start collection
        collector.start()
        time.sleep(3)  # Collect for 3 seconds
        
        # Stop collection
        collector.stop()
        mock_server.stop()
        
        # Verify data was collected
        with open(storage_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) > 0, "No metrics collected"
        assert any('vllm:num_requests_running' in item['metric_name'] for item in data), "Target metric not found"
        
        print("✓ Basic metrics collection test passed")
        
    finally:
        # Cleanup
        if os.path.exists(storage_file):
            os.unlink(storage_file)

def test_csv_storage():
    """Test CSV storage functionality"""
    print("Testing CSV storage...")
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        storage_file = f.name
    
    try:
        storage = CSVStorage(storage_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:8000/metrics",
            storage=storage,
            metrics_to_collect=['vllm:gpu_utilization'],
            collection_interval=1,
            timeout=5
        )
        
        mock_server = MockMetricsServer()
        mock_server.start()
        
        collector.start()
        time.sleep(2)
        collector.stop()
        mock_server.stop()
        
        # Verify CSV data
        with open(storage_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) > 0, "No CSV data collected"
        assert 'metric_name' in rows[0], "CSV headers missing"
        
        print("✓ CSV storage test passed")
        
    finally:
        if os.path.exists(storage_file):
            os.unlink(storage_file)

def test_auto_postprocessing():
    """Test auto-postprocessing functionality"""
    print("Testing auto-postprocessing...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        storage_file = f.name
    
    try:
        storage = JSONStorage(storage_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:8000/metrics",
            storage=storage,
            metrics_to_collect=['vllm:generation_tokens_total'],
            collection_interval=1,
            timeout=5,
            auto_postprocess=True
        )
        
        mock_server = MockMetricsServer()
        mock_server.start()
        
        collector.start()
        time.sleep(2)
        collector.stop()  # This should trigger auto-postprocessing
        mock_server.stop()
        
        # Check if processed file was created
        base_name = os.path.splitext(storage_file)[0]
        processed_file = f"{base_name}_processed.json"
        
        assert os.path.exists(processed_file), "Processed file not created"
        
        # Verify processed data
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)
        
        assert len(processed_data) > 0, "No processed data"
        
        print("✓ Auto-postprocessing test passed")
        
    finally:
        # Cleanup
        for file_path in [storage_file, f"{os.path.splitext(storage_file)[0]}_processed.json"]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_debug_mode():
    """Test debug mode functionality"""
    print("Testing debug mode...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        storage_file = f.name
    
    try:
        storage = JSONStorage(storage_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:8000/metrics",
            storage=storage,
            metrics_to_collect=['vllm:num_requests_running'],
            collection_interval=1,
            timeout=5,
            debug_mode=True
        )
        
        mock_server = MockMetricsServer()
        mock_server.start()
        
        collector.start()
        time.sleep(2)
        collector.stop()
        mock_server.stop()
        
        print("✓ Debug mode test passed")
        
    finally:
        if os.path.exists(storage_file):
            os.unlink(storage_file)

def test_csv_postprocessing():
    """Test CSV postprocessing functionality"""
    print("Testing CSV postprocessing...")
    
    # Create sample CSV data
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'metric_name', 'value', 'labels'])
        writer.writerow(['2024-01-01T00:00:00', 'vllm:generation_tokens_total', '100', '{}'])
        writer.writerow(['2024-01-01T00:00:01', 'vllm:generation_tokens_total', '200', '{}'])
        writer.writerow(['2024-01-01T00:00:02', 'vllm:generation_tokens_total', '300', '{}'])
        csv_file = f.name
    
    try:
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:8000/metrics",
            storage=JSONStorage("dummy.json"),
            metrics_to_collect=[],
            debug_mode=True
        )
        
        # Test CSV postprocessing
        output_file = f"{os.path.splitext(csv_file)[0]}_processed.csv"
        collector.postprocess_metrics(csv_file, output_file)
        
        # Verify processed file
        assert os.path.exists(output_file), "Processed CSV file not created"
        
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3, "Incorrect number of processed rows"
        
        print("✓ CSV postprocessing test passed")
        
    finally:
        # Cleanup
        for file_path in [csv_file, output_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_error_handling():
    """Test error handling with invalid endpoint"""
    print("Testing error handling...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        storage_file = f.name
    
    try:
        storage = JSONStorage(storage_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:9999/metrics",  # Invalid port
            storage=storage,
            metrics_to_collect=['vllm:test_metric'],
            collection_interval=1,
            timeout=2
        )
        
        collector.start()
        time.sleep(3)  # Let it try to collect
        collector.stop()
        
        # Should handle errors gracefully
        print("✓ Error handling test passed")
        
    finally:
        if os.path.exists(storage_file):
            os.unlink(storage_file)

def main():
    """Run all tests"""
    print("=" * 60)
    print("vLLM METRICS COLLECTOR TESTS")
    print("=" * 60)
    
    try:
        test_basic_metrics_collection()
        test_csv_storage()
        test_auto_postprocessing()
        test_debug_mode()
        test_csv_postprocessing()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
