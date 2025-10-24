# vLLM Metrics Collector

A Python program that continuously queries vLLM metrics endpoint and stores specified metrics using threading for non-blocking operation.

## Features

- **Threading-based collection**: Non-blocking metrics collection using Python threads
- **Multiple storage backends**: JSON, CSV, SQLite, and Prometheus support
- **Prometheus integration**: Direct Pushgateway support and Prometheus format output
- **Configurable metrics**: Choose which metrics to collect
- **Robust error handling**: Graceful handling of network errors and timeouts
- **Command-line interface**: Easy to use from command line
- **Programmatic API**: Use in your own Python applications

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

### Programmatic Usage

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

## Examples

See `example_usage.py` for comprehensive usage examples.

## Error Handling

The collector includes robust error handling for:
- Network timeouts and connection errors
- Invalid metric endpoints
- Storage backend errors
- Graceful shutdown on SIGINT/SIGTERM

## Logging

The collector uses Python's logging module. Set `--verbose` for debug-level logging.

## Requirements

- Python 3.7+
- requests>=2.28.0

## License

This project is open source and available under the MIT License.
