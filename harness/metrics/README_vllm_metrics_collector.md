# vLLM Metrics Collector

A comprehensive Python program that continuously queries vLLM metrics endpoint and stores specified metrics using threading for non-blocking operation. Supports multiple storage formats, automatic postprocessing, and debug verification.

## Features

- **Threading-based collection**: Non-blocking metrics collection using Python threads
- **Multiple storage backends**: JSON, CSV, SQLite, and Prometheus support
- **Prometheus integration**: Direct Pushgateway support and Prometheus format output
- **Configurable metrics**: Choose which metrics to collect
- **Robust error handling**: Graceful handling of network errors and timeouts
- **Command-line interface**: Easy to use from command line
- **Programmatic API**: Use in your own Python applications
- **CSV postprocessing**: Support for postprocessing both JSON and CSV files
- **Auto-postprocessing**: Automatically postprocess metrics after collection stops
- **Debug mode**: Additional verification and logging for troubleshooting

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure vLLM server is running with metrics enabled:
```bash
vllm serve <model_name> --enable-metrics
```

## Usage

### Command Line Usage

#### Basic Usage
```bash
python vllm_metrics_collector.py --endpoint http://localhost:8000/metrics
```

#### Advanced Usage
```bash
python vllm_metrics_collector.py \
    --endpoint http://localhost:8000/metrics \
    --interval 5 \
    --storage-type sqlite \
    --output my_metrics \
    --metrics vllm:num_requests_running vllm:generation_tokens_total \
    --verbose
```

#### Prometheus Usage
```bash
# Prometheus file output
python vllm_metrics_collector.py \
    --storage-type prometheus-file \
    --output vllm_metrics

# Prometheus Pushgateway
python vllm_metrics_collector.py \
    --storage-type prometheus \
    --pushgateway-url http://localhost:9091 \
    --job-name vllm_metrics \
    --instance instance_001
```

#### Auto-postprocessing Usage
```bash
# Collect metrics and automatically postprocess when done
python vllm_metrics_collector.py \
    --storage-type csv \
    --output metrics \
    --auto-postprocess
```

#### Debug Mode Usage
```bash
# Enable debug mode for additional verification
python vllm_metrics_collector.py \
    --endpoint http://localhost:8000/metrics \
    --debug \
    --verbose
```

#### Postprocessing Existing Files
```bash
# Postprocess a JSON file
python vllm_metrics_collector.py \
    --postprocess metrics.json \
    --output-processed metrics_processed.json

# Postprocess a CSV file
python vllm_metrics_collector.py \
    --postprocess metrics.csv \
    --output-processed metrics_processed.csv
```

#### Command Line Options

- `--endpoint`: vLLM metrics endpoint URL (default: http://localhost:8000/metrics)
- `--interval`: Collection interval in seconds (default: 10)
- `--timeout`: Request timeout in seconds (default: 30)
- `--storage-type`: Storage backend (json/csv/sqlite/prometheus/prometheus-file, default: csv)
- `--output`: Output file/database name without extension (default: vllm_metrics)
- `--metrics`: Space-separated list of metrics to collect
- `--verbose`: Enable verbose logging
- `--pushgateway-url`: Prometheus Pushgateway URL (for prometheus storage type)
- `--job-name`: Job name for Prometheus Pushgateway (default: vllm_metrics)
- `--instance`: Instance name for Prometheus Pushgateway
- `--postprocess`: Postprocess an existing metrics file
- `--output-processed`: Output file for processed metrics
- `--auto-postprocess`: Automatically postprocess metrics after collection stops
- `--debug`: Enable debug mode for additional verification and logging

### Programmatic Usage

#### Basic Usage
```python
from vllm_metrics_collector import VLLMMetricsCollector, JSONStorage

# Create storage backend
storage = JSONStorage("metrics.json")

# Create collector
collector = VLLMMetricsCollector(
    metrics_endpoint='http://localhost:8000/metrics',
    storage=storage,
    metrics_to_collect=['vllm:num_requests_running', 'vllm:generation_tokens_total'],
    collection_interval=10
)

# Start collection
collector.start()

# Your application logic here...

# Stop collection
collector.stop()
```

#### Advanced Usage with Auto-postprocessing and Debug Mode
```python
from vllm_metrics_collector import VLLMMetricsCollector, CSVStorage

# Create storage backend
storage = CSVStorage("metrics.csv")

# Create collector with auto-postprocessing and debug mode
collector = VLLMMetricsCollector(
    metrics_endpoint='http://localhost:8000/metrics',
    storage=storage,
    metrics_to_collect=['vllm:num_requests_running', 'vllm:gpu_utilization'],
    collection_interval=5,
    auto_postprocess=True,  # Automatically postprocess when collection stops
    debug_mode=True         # Enable debug verification
)

# Start collection
collector.start()

# Your application logic here...

# Stop collection (will automatically postprocess if auto_postprocess=True)
collector.stop()
```

#### Manual Postprocessing
```python
# Postprocess existing metrics file
collector = VLLMMetricsCollector(
    metrics_endpoint='http://localhost:8000/metrics',
    storage=storage,
    metrics_to_collect=[],
    debug_mode=True
)

# Postprocess JSON file
collector.postprocess_metrics('metrics.json', 'metrics_processed.json')

# Postprocess CSV file
collector.postprocess_metrics('metrics.csv', 'metrics_processed.csv')
```

## Available Metrics

The following vLLM metrics are commonly available:

- `vllm:num_requests_running` - Number of requests currently running
- `vllm:generation_tokens_total` - Total number of tokens generated
- `vllm:request_success_total` - Total number of successful requests
- `vllm:request_failure_total` - Total number of failed requests
- `vllm:request_latency` - Request latency metrics
- `vllm:request_input_tokens` - Number of input tokens
- `vllm:request_output_tokens` - Number of output tokens
- `vllm:gpu_utilization` - GPU utilization percentage
- `vllm:gpu_memory_used` - GPU memory used
- `vllm:gpu_memory_total` - Total GPU memory
- `vllm:kv_cache_usage_ratio` - KV cache usage ratio
- `vllm:num_requests_waiting` - Number of requests waiting
- `vllm:num_requests_finished` - Number of finished requests
- `vllm:num_requests_cancelled` - Number of cancelled requests

## Storage Backends

### JSON Storage
Stores metrics in JSON format:
```python
storage = JSONStorage("metrics.json")
```

### CSV Storage
Stores metrics in CSV format:
```python
storage = CSVStorage("metrics.csv")
```

### SQLite Storage
Stores metrics in SQLite database:
```python
storage = SQLiteStorage("metrics.db")
```

### Prometheus Storage
Stores metrics in Prometheus format with optional Pushgateway support:
```python
# Prometheus file output only
storage = PrometheusFileStorage("metrics.prom")

# Prometheus with Pushgateway
storage = PrometheusStorage(
    output_path="metrics.prom",  # Optional file output
    pushgateway_url="http://localhost:9091",
    job_name="vllm_metrics",
    instance="instance_001"
)
```

## Postprocessing

The metrics collector supports postprocessing of collected metrics to convert counter metrics to deltas and process histogram metrics. This is useful for analysis and visualization.

### Automatic Postprocessing
Enable automatic postprocessing when collection stops:
```bash
python vllm_metrics_collector.py --auto-postprocess
```

### Manual Postprocessing
Postprocess existing files:
```bash
# JSON to JSON
python vllm_metrics_collector.py --postprocess metrics.json

# CSV to CSV
python vllm_metrics_collector.py --postprocess metrics.csv

# JSON to CSV
python vllm_metrics_collector.py --postprocess metrics.json --output-processed metrics_processed.csv
```

### Postprocessing Features
- **Counter Processing**: Converts counter metrics to deltas (rate of change)
- **Histogram Processing**: Processes histogram metrics (buckets, count, sum)
- **Format Support**: Supports both JSON and CSV input/output formats
- **Backup Creation**: Automatically creates .raw backup of original files
- **Metrics Type Detection**: Uses metrics_info.txt for metric type information

## Debug Mode

Debug mode provides additional verification and logging to help troubleshoot issues:

### Features
- **Endpoint Verification**: Checks if the metrics endpoint is accessible and contains expected metrics
- **Metrics Validation**: Verifies parsed metrics for validity and target metric presence
- **Enhanced Logging**: Provides detailed debug information about the collection process
- **Error Detection**: Identifies potential issues with metric parsing and storage

### Usage
```bash
# Enable debug mode
python vllm_metrics_collector.py --debug --verbose
```

### Debug Output
Debug mode provides information about:
- Endpoint accessibility and response content
- Target metrics found in the response
- Parsed metrics validation
- Storage operations
- Error conditions and warnings

## Examples

### Complete Workflow Example
```bash
# 1. Collect metrics with auto-postprocessing
python vllm_metrics_collector.py \
    --storage-type csv \
    --output run1_metrics \
    --auto-postprocess \
    --debug

# 2. Postprocess existing file manually
python vllm_metrics_collector.py \
    --postprocess run1_metrics.csv \
    --output-processed run1_processed.csv

# 3. Compare with another run
python vllm_metrics_visualizer.py \
    --file run1_processed.csv \
    --compare-file run2_processed.csv \
    --metrics "vllm:gpu_utilization" "vllm:request_latency"
```

### Programmatic Example
```python
from vllm_metrics_collector import VLLMMetricsCollector, CSVStorage
import time

# Create collector with all features
collector = VLLMMetricsCollector(
    metrics_endpoint='http://localhost:8000/metrics',
    storage=CSVStorage("metrics.csv"),
    metrics_to_collect=[
        'vllm:num_requests_running',
        'vllm:gpu_utilization',
        'vllm:request_latency'
    ],
    collection_interval=5,
    auto_postprocess=True,
    debug_mode=True
)

try:
    # Start collection
    collector.start()
    
    # Run for 5 minutes
    time.sleep(300)
    
finally:
    # Stop collection (will auto-postprocess)
    collector.stop()
```

## Error Handling

The collector includes robust error handling for:
- Network timeouts and connection errors
- Invalid metric endpoints
- Storage backend errors
- Graceful shutdown on SIGINT/SIGTERM
- Debug mode verification failures

## Logging

The collector uses Python's logging module with multiple levels:
- **INFO**: Standard operation information
- **DEBUG**: Detailed debug information (use with `--debug` or `--verbose`)
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

## Requirements

- Python 3.7+
- requests>=2.28.0
- pandas>=1.5.0
- sqlite3 (built-in)

## License

This project is open source and available under the MIT License.
