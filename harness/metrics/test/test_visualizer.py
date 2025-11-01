#!/usr/bin/env python3
"""
Test script for vLLM Metrics Visualizer functionality.

This script tests the visualizer with various configurations including:
- Single metric plotting
- Multiple metrics plotting
- Run comparison
- Summary report generation
- Different storage formats
- Error handling
"""

import os
import sys
import tempfile
import json
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_metrics_visualizer import VLLMMetricsVisualizer

def create_sample_json_data(file_path, num_samples=100):
    """Create sample JSON metrics data"""
    data = []
    base_time = datetime.now() - timedelta(minutes=10)
    
    for i in range(num_samples):
        timestamp = base_time + timedelta(seconds=i*6)
        data.extend([
            {
                'timestamp': timestamp.isoformat(),
                'metric_name': 'vllm:num_requests_running',
                'value': np.random.randint(1, 10),
                'labels': {}
            },
            {
                'timestamp': timestamp.isoformat(),
                'metric_name': 'vllm:gpu_utilization',
                'value': np.random.uniform(50, 95),
                'labels': {}
            },
            {
                'timestamp': timestamp.isoformat(),
                'metric_name': 'vllm:request_latency',
                'value': np.random.uniform(0.1, 2.0),
                'labels': {}
            }
        ])
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def create_sample_csv_data(file_path, num_samples=100):
    """Create sample CSV metrics data"""
    data = []
    base_time = datetime.now() - timedelta(minutes=10)
    
    for i in range(num_samples):
        timestamp = base_time + timedelta(seconds=i*6)
        data.extend([
            {
                'timestamp': timestamp.isoformat(),
                'metric_name': 'vllm:num_requests_running',
                'value': np.random.randint(1, 10),
                'labels': '{}'
            },
            {
                'timestamp': timestamp.isoformat(),
                'metric_name': 'vllm:gpu_utilization',
                'value': np.random.uniform(50, 95),
                'labels': '{}'
            }
        ])
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def test_single_metric_plotting():
    """Test single metric plotting functionality"""
    print("Testing single metric plotting...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        # Create sample data
        create_sample_json_data(json_file, 50)
        
        # Test visualizer
        visualizer = VLLMMetricsVisualizer()
        
        # Test plotting single metric
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
            visualizer.plot_metric(
                file_path=json_file,
                metric_name='vllm:gpu_utilization',
                title='GPU Utilization Test',
                save_path=png_file.name
            )
            
            assert os.path.exists(png_file.name), "Plot file not created"
            assert os.path.getsize(png_file.name) > 0, "Plot file is empty"
        
        print("✓ Single metric plotting test passed")
        
    finally:
        # Cleanup
        for file_path in [json_file, png_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_multiple_metrics_plotting():
    """Test multiple metrics plotting functionality"""
    print("Testing multiple metrics plotting...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        # Create sample data
        create_sample_json_data(json_file, 30)
        
        visualizer = VLLMMetricsVisualizer()
        
        # Test plotting multiple metrics
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
            visualizer.plot_multiple_metrics(
                file_path=json_file,
                metric_names=['vllm:num_requests_running', 'vllm:gpu_utilization'],
                title='Multiple Metrics Test',
                save_path=png_file.name
            )
            
            assert os.path.exists(png_file.name), "Multi-metric plot file not created"
            assert os.path.getsize(png_file.name) > 0, "Multi-metric plot file is empty"
        
        print("✓ Multiple metrics plotting test passed")
        
    finally:
        # Cleanup
        for file_path in [json_file, png_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_run_comparison():
    """Test run comparison functionality"""
    print("Testing run comparison...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f1:
        file1 = f1.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f2:
        file2 = f2.name
    
    try:
        # Create two different datasets
        create_sample_json_data(file1, 20)
        create_sample_json_data(file2, 20)
        
        visualizer = VLLMMetricsVisualizer()
        
        # Test comparison plotting
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
            visualizer.compare_metrics(
                file_path1=file1,
                file_path2=file2,
                metric_name='vllm:gpu_utilization',
                label1='Run 1',
                label2='Run 2',
                title='GPU Utilization Comparison',
                save_path=png_file.name
            )
            
            assert os.path.exists(png_file.name), "Comparison plot file not created"
            assert os.path.getsize(png_file.name) > 0, "Comparison plot file is empty"
        
        print("✓ Run comparison test passed")
        
    finally:
        # Cleanup
        for file_path in [file1, file2, png_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_csv_format():
    """Test CSV format support"""
    print("Testing CSV format...")
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        csv_file = f.name
    
    try:
        # Create sample CSV data
        create_sample_csv_data(csv_file, 25)
        
        visualizer = VLLMMetricsVisualizer()
        
        # Test CSV plotting
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
            visualizer.plot_metric(
                file_path=csv_file,
                metric_name='vllm:gpu_utilization',
                title='CSV Format Test',
                save_path=png_file.name
            )
            
            assert os.path.exists(png_file.name), "CSV plot file not created"
        
        print("✓ CSV format test passed")
        
    finally:
        # Cleanup
        for file_path in [csv_file, png_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_summary_report():
    """Test summary report generation"""
    print("Testing summary report...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        summary_file = f.name
    
    try:
        # Create sample data
        create_sample_json_data(json_file, 40)
        
        visualizer = VLLMMetricsVisualizer()
        
        # Generate summary report
        summary = visualizer.generate_summary_report(json_file, summary_file)
        
        # Verify summary file
        assert os.path.exists(summary_file), "Summary file not created"
        
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        assert 'file_path' in summary_data, "Summary missing file_path"
        assert 'total_records' in summary_data, "Summary missing total_records"
        assert 'metrics' in summary_data, "Summary missing metrics"
        
        print("✓ Summary report test passed")
        
    finally:
        # Cleanup
        for file_path in [json_file, summary_file]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def test_available_metrics():
    """Test available metrics detection"""
    print("Testing available metrics...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        # Create sample data
        create_sample_json_data(json_file, 20)
        
        visualizer = VLLMMetricsVisualizer()
        
        # Get available metrics
        metrics = visualizer.get_available_metrics(json_file)
        
        assert len(metrics) > 0, "No metrics found"
        assert 'vllm:gpu_utilization' in metrics, "Expected metric not found"
        
        print("✓ Available metrics test passed")
        
    finally:
        # Cleanup
        if os.path.exists(json_file):
            os.unlink(json_file)

def test_error_handling():
    """Test error handling with invalid files"""
    print("Testing error handling...")
    
    visualizer = VLLMMetricsVisualizer()
    
    # Test with non-existent file
    try:
        visualizer.plot_metric("nonexistent.json", "test_metric")
        assert False, "Should have raised an exception"
    except Exception:
        pass  # Expected
    
    # Test with invalid metric
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        create_sample_json_data(json_file, 10)
        
        # This should handle gracefully
        try:
            visualizer.plot_metric(json_file, "nonexistent_metric")
        except Exception:
            pass  # Expected for non-existent metric
        
        print("✓ Error handling test passed")
        
    finally:
        if os.path.exists(json_file):
            os.unlink(json_file)

def test_custom_styling():
    """Test custom styling functionality"""
    print("Testing custom styling...")
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        json_file = f.name
    
    try:
        # Create sample data
        create_sample_json_data(json_file, 15)
        
        # Test with custom styling
        visualizer = VLLMMetricsVisualizer(
            style='seaborn-v0_8-whitegrid',
            figsize=(10, 6)
        )
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
            visualizer.plot_metric(
                file_path=json_file,
                metric_name='vllm:gpu_utilization',
                title='Custom Styling Test',
                save_path=png_file.name
            )
            
            assert os.path.exists(png_file.name), "Styled plot file not created"
        
        print("✓ Custom styling test passed")
        
    finally:
        # Cleanup
        for file_path in [json_file, png_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

def main():
    """Run all visualizer tests"""
    print("=" * 60)
    print("vLLM METRICS VISUALIZER TESTS")
    print("=" * 60)
    
    try:
        test_single_metric_plotting()
        test_multiple_metrics_plotting()
        test_run_comparison()
        test_csv_format()
        test_summary_report()
        test_available_metrics()
        test_error_handling()
        test_custom_styling()
        
        print("\n" + "=" * 60)
        print("ALL VISUALIZER TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
