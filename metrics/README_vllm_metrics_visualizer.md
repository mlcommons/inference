# vLLM Metrics Visualizer

A comprehensive visualization tool for vLLM metrics data collected by the metrics collector. Supports multiple storage formats and provides plotting capabilities for analysis and comparison.

## Features

- **Multiple Storage Formats**: Support for JSON, CSV, SQLite, and Prometheus formats
- **Single Metric Plotting**: Plot individual metrics over time
- **Multiple Metrics Plotting**: Plot multiple metrics in subplots
- **Run Comparison**: Compare metrics between different runs
- **Summary Reports**: Generate statistical summaries of metrics
- **Command Line Interface**: Easy-to-use CLI for quick visualization
- **Programmatic API**: Use in your own Python applications
- **Custom Styling**: Customizable plots with different themes and sizes

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have metrics data collected using the vLLM Metrics Collector.

## Usage

### Command Line Interface

#### Basic Usage
```bash
# List available metrics
python vllm_metrics_visualizer.py --file metrics.json --list-metrics

# Plot a single metric
python vllm_metrics_visualizer.py --file metrics.json --metric "vllm:num_requests_running"

# Plot multiple metrics
python vllm_metrics_visualizer.py --file metrics.json --metrics "vllm:num_requests_running" "vllm:gpu_utilization"
```

#### Comparison Usage
```bash
# Compare single metric between two runs
python vllm_metrics_visualizer.py \
    --file run1_metrics.json \
    --compare-file run2_metrics.json \
    --metric "vllm:request_latency" \
    --label1 "Baseline" \
    --label2 "Optimized"

# Compare multiple metrics
python vllm_metrics_visualizer.py \
    --file run1_metrics.json \
    --compare-file run2_metrics.json \
    --metrics "vllm:num_requests_running" "vllm:gpu_utilization" \
    --label1 "Baseline" \
    --label2 "Optimized"
```

#### Advanced Usage
```bash
# Generate summary report
python vllm_metrics_visualizer.py --file metrics.json --summary report.json

# Save plots to file
python vllm_metrics_visualizer.py \
    --file metrics.json \
    --metric "vllm:gpu_utilization" \
    --save "gpu_utilization.png"

# Use different storage format
python vllm_metrics_visualizer.py \
    --file metrics.db \
    --format sqlite \
    --metric "vllm:num_requests_running"
```

### Programmatic Usage

#### Basic Plotting
```python
from vllm_metrics_visualizer import VLLMMetricsVisualizer

# Create visualizer
visualizer = VLLMMetricsVisualizer()

# Plot single metric
visualizer.plot_metric(
    file_path="metrics.json",
    metric_name="vllm:num_requests_running",
    title="Running Requests Over Time"
)

# Plot multiple metrics
visualizer.plot_multiple_metrics(
    file_path="metrics.csv",
    metric_names=["vllm:num_requests_running", "vllm:gpu_utilization"],
    title="Performance Metrics"
)
```

#### Comparison Plotting
```python
# Compare single metric
visualizer.compare_metrics(
    file_path1="run1_metrics.json",
    file_path2="run2_metrics.json",
    metric_name="vllm:request_latency",
    label1="Baseline Run",
    label2="Optimized Run"
)

# Compare multiple metrics
visualizer.compare_multiple_metrics(
    file_path1="baseline_metrics.csv",
    file_path2="optimized_metrics.csv",
    metric_names=["vllm:num_requests_running", "vllm:gpu_utilization"],
    label1="Baseline",
    label2="Optimized"
)
```

#### Data Analysis
```python
# Load metrics data
df = visualizer.load_metrics("metrics.json")

# Get available metrics
metrics = visualizer.get_available_metrics("metrics.json")

# Generate summary report
summary = visualizer.generate_summary_report("metrics.json", "summary.json")
```

### Custom Styling

```python
# Create visualizer with custom styling
visualizer = VLLMMetricsVisualizer(
    style='seaborn-v0_8-darkgrid',
    figsize=(15, 10)
)

# Use custom styling for plots
visualizer.plot_metric(
    file_path="metrics.json",
    metric_name="vllm:gpu_utilization",
    title="GPU Utilization with Custom Styling"
)
```

## Supported Storage Formats

### JSON Format
```python
# JSON metrics file
visualizer.plot_metric("metrics.json", "vllm:num_requests_running")
```

### CSV Format
```python
# CSV metrics file
visualizer.plot_metric("metrics.csv", "vllm:gpu_utilization", format_type="csv")
```

### SQLite Format
```python
# SQLite database
visualizer.plot_metric("metrics.db", "vllm:request_latency", format_type="sqlite")
```

### Prometheus Format
```python
# Prometheus format file
visualizer.plot_metric("metrics.prom", "vllm_request_duration_seconds_bucket", format_type="prometheus")
```

## Command Line Options

- `--file`: Path to metrics file (required)
- `--format`: Storage format (json/csv/sqlite/prometheus, auto-detected if not specified)
- `--metric`: Single metric to plot
- `--metrics`: Multiple metrics to plot (space-separated)
- `--compare-file`: Second file for comparison
- `--compare-format`: Format of comparison file
- `--label1`: Label for first run (default: "Run 1")
- `--label2`: Label for second run (default: "Run 2")
- `--title`: Plot title
- `--save`: Path to save plot
- `--summary`: Generate summary report (JSON format)
- `--list-metrics`: List available metrics in file

## Examples

### Complete Workflow Example
```bash
# 1. Collect metrics
python vllm_metrics_collector.py --storage-type csv --output run1_metrics

# 2. Visualize single metric
python vllm_metrics_visualizer.py \
    --file run1_metrics.csv \
    --metric "vllm:gpu_utilization" \
    --save "gpu_utilization.png"

# 3. Compare two runs
python vllm_metrics_visualizer.py \
    --file run1_metrics.csv \
    --compare-file run2_metrics.csv \
    --metrics "vllm:gpu_utilization" "vllm:request_latency" \
    --label1 "Baseline" \
    --label2 "Optimized" \
    --save "comparison.png"

# 4. Generate summary report
python vllm_metrics_visualizer.py \
    --file run1_metrics.csv \
    --summary "run1_summary.json"
```

### Programmatic Example
```python
from vllm_metrics_visualizer import VLLMMetricsVisualizer

# Create visualizer with custom styling
visualizer = VLLMMetricsVisualizer(
    style='seaborn-v0_8-whitegrid',
    figsize=(12, 8)
)

# Load and analyze data
df = visualizer.load_metrics("metrics.csv")
available_metrics = visualizer.get_available_metrics("metrics.csv")

print(f"Available metrics: {available_metrics}")

# Plot key performance metrics
visualizer.plot_multiple_metrics(
    file_path="metrics.csv",
    metric_names=[
        "vllm:num_requests_running",
        "vllm:gpu_utilization",
        "vllm:request_latency"
    ],
    title="Key Performance Metrics",
    save_path="performance_metrics.png"
)

# Compare with baseline
visualizer.compare_multiple_metrics(
    file_path1="baseline_metrics.csv",
    file_path2="optimized_metrics.csv",
    metric_names=["vllm:gpu_utilization", "vllm:request_latency"],
    label1="Baseline",
    label2="Optimized",
    title="Performance Comparison",
    save_path="comparison.png"
)

# Generate comprehensive summary
summary = visualizer.generate_summary_report(
    "metrics.csv",
    "detailed_summary.json"
)
```

## Output Formats

### Plots
- High-resolution PNG images (300 DPI)
- Customizable titles and labels
- Support for different matplotlib styles
- Automatic legend generation for labeled metrics

### Summary Reports
JSON format with statistical information:
```json
{
  "file_path": "metrics.json",
  "total_records": 1000,
  "time_range": {
    "start": "2024-01-01T00:00:00",
    "end": "2024-01-01T01:00:00"
  },
  "metrics": {
    "vllm:num_requests_running": {
      "count": 100,
      "mean": 5.2,
      "std": 1.8,
      "min": 0.0,
      "max": 10.0,
      "median": 5.0
    }
  }
}
```

## Integration with Metrics Collector

The visualizer works seamlessly with the vLLM Metrics Collector:

1. **Collect metrics** using the metrics collector:
```bash
python vllm_metrics_collector.py --storage-type json --output metrics
```

2. **Visualize metrics** using the visualizer:
```bash
python vllm_metrics_visualizer.py --file metrics.json --metric "vllm:num_requests_running"
```

3. **Compare runs** by collecting metrics from different configurations:
```bash
# Run 1
python vllm_metrics_collector.py --storage-type csv --output run1_metrics

# Run 2 (different configuration)
python vllm_metrics_collector.py --storage-type csv --output run2_metrics

# Compare
python vllm_metrics_visualizer.py \
    --file run1_metrics.csv \
    --compare-file run2_metrics.csv \
    --metrics "vllm:gpu_utilization" "vllm:request_latency"
```

## Advanced Features

### Custom Plot Styling
```python
# Available matplotlib styles
styles = [
    'default', 'seaborn-v0_8', 'seaborn-v0_8-darkgrid',
    'seaborn-v0_8-whitegrid', 'seaborn-v0_8-dark',
    'seaborn-v0_8-white', 'seaborn-v0_8-ticks'
]

visualizer = VLLMMetricsVisualizer(style='seaborn-v0_8-darkgrid')
```

### Batch Processing
```python
# Process multiple files
files = ["run1_metrics.csv", "run2_metrics.csv", "run3_metrics.csv"]
for file in files:
    visualizer.plot_metric(
        file_path=file,
        metric_name="vllm:gpu_utilization",
        save_path=f"gpu_util_{file.replace('.csv', '.png')}"
    )
```

### Statistical Analysis
```python
# Generate detailed statistics
summary = visualizer.generate_summary_report("metrics.csv", "stats.json")

# Access specific statistics
gpu_stats = summary['metrics']['vllm:gpu_utilization']
print(f"GPU Utilization - Mean: {gpu_stats['mean']:.2f}, Std: {gpu_stats['std']:.2f}")
```

## Error Handling

The visualizer includes robust error handling for:
- Invalid file formats
- Missing metrics
- Data parsing errors
- Plot generation failures
- File I/O errors

## Performance Considerations

- **Large datasets**: The visualizer efficiently handles large metrics files
- **Memory usage**: Optimized for memory efficiency with large datasets
- **Plot generation**: Fast plotting with matplotlib optimization
- **File formats**: Efficient parsing for all supported formats

## Requirements

- Python 3.7+
- pandas>=1.5.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- numpy>=1.21.0

## License

This project is open source and available under the MIT License.
