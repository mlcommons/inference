#!/usr/bin/env python3
"""
Integration test script for vLLM Metrics Collector and Visualizer.

This script tests the complete workflow:
1. Collect metrics using the collector
2. Postprocess the collected metrics
3. Visualize the processed metrics
4. Generate summary reports
"""

import os
import sys
import time
import tempfile
import json
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_metrics_collector import VLLMMetricsCollector, JSONStorage, CSVStorage
from vllm_metrics_visualizer import VLLMMetricsVisualizer

class MockMetricsServer:
    """Mock vLLM metrics server for integration testing"""
    
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
                    
                    # Generate dynamic metrics
                    current_time = datetime.now().isoformat()
                    metrics = f"""# HELP vllm_num_requests_running Number of requests currently running
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running {5 + (int(time.time()) % 10)}
# HELP vllm_generation_tokens_total Total number of tokens generated
# TYPE vllm_generation_tokens_total counter
vllm_generation_tokens_total {1000 + (int(time.time()) % 1000)}
# HELP vllm_gpu_utilization GPU utilization percentage
# TYPE vllm_gpu_utilization gauge
vllm_gpu_utilization {70.0 + (int(time.time()) % 30)}
# HELP vllm_request_latency Request latency
# TYPE vllm_request_latency histogram
vllm_request_latency_bucket{{le="0.1"}} 10.0
vllm_request_latency_bucket{{le="0.5"}} 25.0
vllm_request_latency_bucket{{le="1.0"}} 30.0
vllm_request_latency_bucket{{le="+Inf"}} 30.0
vllm_request_latency_count 30.0
vllm_request_latency_sum 15.0
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

def test_complete_workflow():
    """Test complete workflow: collect -> postprocess -> visualize"""
    print("Testing complete workflow...")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        metrics_file = f.name
    
    try:
        # Start mock server
        mock_server = MockMetricsServer()
        mock_server.start()
        
        # Step 1: Collect metrics
        print("  Step 1: Collecting metrics...")
        storage = JSONStorage(metrics_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint=mock_server.url,
            storage=storage,
            metrics_to_collect=[
                'vllm:num_requests_running',
                'vllm:gpu_utilization',
                'vllm:generation_tokens_total'
            ],
            collection_interval=1,
            timeout=5,
            debug_mode=True
        )
        
        collector.start()
        time.sleep(5)  # Collect for 5 seconds
        collector.stop()
        
        # Verify metrics were collected
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
        
        assert len(metrics_data) > 0, "No metrics collected"
        print(f"  ✓ Collected {len(metrics_data)} metric entries")
        
        # Step 2: Postprocess metrics
        print("  Step 2: Postprocessing metrics...")
        processed_file = f"{os.path.splitext(metrics_file)[0]}_processed.json"
        collector.postprocess_metrics(metrics_file, processed_file)
        
        # Verify processed file
        assert os.path.exists(processed_file), "Processed file not created"
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)
        
        assert len(processed_data) > 0, "No processed metrics"
        print(f"  ✓ Processed {len(processed_data)} metric entries")
        
        # Step 3: Visualize metrics
        print("  Step 3: Visualizing metrics...")
        visualizer = VLLMMetricsVisualizer()
        
        # Test single metric plotting
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png1:
            visualizer.plot_metric(
                file_path=processed_file,
                metric_name='vllm:gpu_utilization',
                title='GPU Utilization Over Time',
                save_path=png1.name
            )
            assert os.path.exists(png1.name), "Single metric plot not created"
        
        # Test multiple metrics plotting
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png2:
            visualizer.plot_multiple_metrics(
                file_path=processed_file,
                metric_names=['vllm:num_requests_running', 'vllm:gpu_utilization'],
                title='System Metrics Overview',
                save_path=png2.name
            )
            assert os.path.exists(png2.name), "Multi-metric plot not created"
        
        # Test summary report
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as summary_file:
            summary = visualizer.generate_summary_report(processed_file, summary_file.name)
            assert os.path.exists(summary_file.name), "Summary report not created"
        
        print("  ✓ Generated visualizations and summary report")
        
        # Step 4: Test CSV workflow
        print("  Step 4: Testing CSV workflow...")
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as csv_file:
            csv_storage = CSVStorage(csv_file.name)
            csv_collector = VLLMMetricsCollector(
                metrics_endpoint=mock_server.url,
                storage=csv_storage,
                metrics_to_collect=['vllm:gpu_utilization'],
                collection_interval=1,
                timeout=5
            )
            
            csv_collector.start()
            time.sleep(3)
            csv_collector.stop()
            
            # Test CSV postprocessing
            csv_processed = f"{os.path.splitext(csv_file.name)[0]}_processed.csv"
            csv_collector.postprocess_metrics(csv_file.name, csv_processed)
            
            assert os.path.exists(csv_processed), "CSV processed file not created"
            
            # Test CSV visualization
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as csv_png:
                visualizer.plot_metric(
                    file_path=csv_processed,
                    metric_name='vllm:gpu_utilization',
                    title='CSV GPU Utilization',
                    save_path=csv_png.name
                )
                assert os.path.exists(csv_png.name), "CSV plot not created"
        
        print("  ✓ CSV workflow completed successfully")
        
        # Cleanup
        mock_server.stop()
        
        print("✓ Complete workflow test passed")
        
    finally:
        # Cleanup all temporary files
        for file_path in [metrics_file, processed_file, png1.name, png2.name, 
                         summary_file.name, csv_file.name, csv_processed, csv_png.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_auto_postprocessing_workflow():
    """Test workflow with auto-postprocessing"""
    print("Testing auto-postprocessing workflow...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        metrics_file = f.name
    
    try:
        # Start mock server
        mock_server = MockMetricsServer()
        mock_server.start()
        
        # Test auto-postprocessing
        storage = JSONStorage(metrics_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint=mock_server.url,
            storage=storage,
            metrics_to_collect=['vllm:gpu_utilization'],
            collection_interval=1,
            timeout=5,
            auto_postprocess=True  # Enable auto-postprocessing
        )
        
        collector.start()
        time.sleep(3)
        collector.stop()  # This should trigger auto-postprocessing
        
        # Check if processed file was created automatically
        processed_file = f"{os.path.splitext(metrics_file)[0]}_processed.json"
        assert os.path.exists(processed_file), "Auto-processed file not created"
        
        # Test visualization of auto-processed data
        visualizer = VLLMMetricsVisualizer()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
            visualizer.plot_metric(
                file_path=processed_file,
                metric_name='vllm:gpu_utilization',
                title='Auto-Processed GPU Utilization',
                save_path=png_file.name
            )
            assert os.path.exists(png_file.name), "Auto-processed plot not created"
        
        print("✓ Auto-postprocessing workflow test passed")
        
    finally:
        # Cleanup
        mock_server.stop()
        for file_path in [metrics_file, processed_file, png_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_error_recovery():
    """Test error recovery and handling"""
    print("Testing error recovery...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        metrics_file = f.name
    
    try:
        # Test with invalid endpoint
        storage = JSONStorage(metrics_file)
        collector = VLLMMetricsCollector(
            metrics_endpoint="http://localhost:9999/metrics",  # Invalid port
            storage=storage,
            metrics_to_collect=['vllm:test_metric'],
            collection_interval=1,
            timeout=2,
            debug_mode=True
        )
        
        collector.start()
        time.sleep(3)  # Let it try to collect
        collector.stop()
        
        # Should handle errors gracefully
        print("✓ Error recovery test passed")
        
    finally:
        if os.path.exists(metrics_file):
            os.unlink(metrics_file)

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("vLLM METRICS INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_complete_workflow()
        test_auto_postprocessing_workflow()
        test_error_recovery()
        
        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
