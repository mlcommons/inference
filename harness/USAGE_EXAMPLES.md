# Comprehensive Usage Examples

This document provides detailed examples for using the MLPerf Inference Harness with various configurations and options.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Model Configuration](#model-configuration)
3. [Dataset Configuration](#dataset-configuration)
4. [Endpoint Configuration](#endpoint-configuration)
5. [Backend Configuration](#backend-configuration)
6. [Scenario Examples](#scenario-examples)
7. [Advanced Examples](#advanced-examples)

### Sample command
```bash
 python harness_main.py --model-category llama3.1-8b --model RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8  --dataset-path cnn_eval.json --dataset-name llama3.1-8b --server-config backendserver/simple.yaml --scenario Server --test-mode performance --batch-size 13368 --num-samples 13368 --output-dir TEST-SERVER --lg-model-name llama3_1-8b --server-target-qps 40
 ```

## Basic Usage

### Example 1: Simplest Case - Using Model Name Auto-Detection

```bash
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json
```

**What happens:**
- Auto-detects dataset name from path (`cnn_eval`)
- Falls back to model name if dataset config not found
- Uses default settings (Offline scenario, performance mode)
- Defaults to completions endpoint

### Example 2: Explicit Model and Dataset

```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --scenario Offline \
    --test-mode performance
```

### Example 3: Using Model-Specific Harness

```bash
python harness/harness_llama3.1_8b.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json
```

**Benefits:**
- Automatically sets `dataset_name="llama3.1-8b"`
- Can add model-specific customizations
- Cleaner command line

## Model Configuration

### Example 4: Different Model Names

```bash
# HuggingFace model name
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json

# Local model path
python harness/harness_main.py \
    --model /path/to/local/model \
    --dataset-path ./cnn_eval.json

# Short model identifier
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json
```

### Example 5: DeepSeek R1 Model

```bash
python language/deepseek-r1/harness_deepseek_r1.py \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --dataset-path ./deepseek_dataset.pkl \
    --dataset-name deepseek-r1
```

### Example 6: Using Model Config File

```bash
# Model config is auto-loaded from configs/models/llama3.1-8b.yaml
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json
```

## Dataset Configuration

### Example 7: Using Dataset Config File

```bash
# Specify exact config file to use
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./my_dataset.pkl \
    --dataset-config-file configs/datasets/my-dataset.yaml
```

### Example 8: Overriding Column Names

```bash
# Override column mappings without creating config file
python harness/harness_main.py \
    --model my-model/MyModel \
    --dataset-path ./dataset.pkl \
    --input-column prompt \
    --input-ids-column token_ids \
    --output-column target
```

### Example 9: Combining Config and Overrides

```bash
# Use config file but override specific column
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-config-file configs/datasets/llama3.1-8b.yaml \
    --input-column custom_input  # Overrides config's input_column
```

### Example 10: Different Dataset Formats

```bash
# JSON dataset
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b

# Pickle dataset
python harness/harness_main.py \
    --model deepseek-r1 \
    --dataset-path ./deepseek_dataset.pkl \
    --dataset-name deepseek-r1

# CSV dataset (needs config with column mappings)
python harness/harness_main.py \
    --model my-model \
    --dataset-path ./my_dataset.csv \
    --dataset-name my-dataset
```

## Endpoint Configuration

### Example 11: Using Completions Endpoint (Default)

```bash
# Explicitly specify completions endpoint
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions
```

### Example 12: Using Chat Completions Endpoint

```bash
# Use chat completions endpoint
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type chat_completions
```

### Example 13: Endpoint Validation Error

If you try to use an endpoint that doesn't exist for the backend:

```bash
# This will fail if backend only supports completions
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type chat_completions \
    --server-config backend=my-backend-only-completions
```

**Error:**
```
ValueError: Endpoint 'chat_completions' is not available for backend 'my-backend-only-completions'. 
Available endpoints: ['completions']
```

## Backend Configuration

### Example 14: Using vLLM Backend

```bash
# vLLM supports both endpoints
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type chat_completions \
    --server-config backend=vllm
```

### Example 15: Using SGLang Backend

```bash
# SGLang also supports both endpoints
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --server-config backend=sglang
```

### Example 16: Backend-Specific Config File

Create `configs/backends/my-backend.yaml`:

```yaml
name: my-backend
description: "My custom backend"

endpoints:
  - completions
  # Only completions available

default_endpoint: completions
```

Then use it:

```bash
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --server-config backend=my-backend
```

## Scenario Examples

### Example 17: Offline Scenario (Throughput)

```bash
# Offline scenario - batch processing
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --test-mode performance \
    --batch-size 13368 \
    --num-samples 13368 \
    --api-server-url http://localhost:8000
```

### Example 18: Server Scenario (Latency)

```bash
# Server scenario - real-time queries
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --server-target-qps 100.0 \
    --server-coalesce-queries true
```

### Example 19: Accuracy Testing

```bash
# Accuracy mode - both scenarios
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --test-mode accuracy \
    --api-server-url http://localhost:8000
```

## Advanced Examples

### Example 20: Full Configuration with All Options

```bash
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --dataset-config-file configs/datasets/llama3.1-8b.yaml \
    --input-column input \
    --input-ids-column tok_input \
    --output-column output \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --server-config backendserver/example_server_config.yaml \
    --batch-size 13368 \
    --num-samples 13368 \
    --enable-metrics \
    --mlflow-experiment-name llama3.1-8b-experiment \
    --mlflow-host localhost \
    --mlflow-port 5000 \
    --output-dir ./harness_output \
    --user-conf user.conf \
    --log-level INFO
```

### Example 21: Python API with All Features

```python
from harness.base_harness import BaseHarness

harness = BaseHarness(
    # Model configuration
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    
    # Dataset configuration
    dataset_path="./cnn_eval.json",
    dataset_name="llama3.1-8b",
    dataset_config_file="configs/datasets/llama3.1-8b.yaml",  # Optional
    input_column="input",  # Optional override
    input_ids_column="tok_input",  # Optional override
    output_column="output",  # Optional override
    
    # Scenario configuration
    scenario="Offline",
    test_mode="performance",
    
    # Server configuration
    api_server_url="http://localhost:8000",
    server_config={
        'backend': 'vllm',
        'endpoint_type': 'completions',  # or 'chat_completions'
    },
    
    # Performance configuration
    batch_size=13368,
    num_samples=13368,
    
    # Metrics and MLflow
    enable_metrics=True,
    metrics_interval=15,
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="my-experiment",
    
    # Output configuration
    output_dir="./harness_output",
)

results = harness.run(
    user_conf="user.conf",
    lg_model_name="llama3_1-8b"
)

print(f"Test status: {results['status']}")
print(f"Duration: {results['duration']} seconds")
```

### Example 22: Server Scenario with Custom Settings

```bash
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --endpoint-type chat_completions \
    --server-target-qps 50.0 \
    --server-coalesce-queries true \
    --enable-metrics \
    --mlflow-experiment-name server-test
```

### Example 23: Using Different Backends

```bash
# vLLM backend
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --server-config backend=vllm

# SGLang backend
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --server-config backend=sglang
```

### Example 24: Local vs Remote Server

```bash
# Let harness start server automatically
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml

# Use existing remote server
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://remote-server:8000
```

### Example 25: Custom Output Directory

```bash
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --output-dir ./my_custom_output \
    --enable-metrics
```

**Output structure:**
```
my_custom_output/
├── harness_output/
├── server/
├── metrics/
├── visualizations/
├── mlperf/
└── environment/
```

## Configuration Priority

The system loads configurations in this order (highest priority first):

1. **Command-line arguments** (e.g., `--input-column`) - Highest priority
2. **Explicit config file** (`--dataset-config-file`)
3. **Model-specific dataset config** (`models/{model}/{dataset}.yaml`)
4. **Model config** (`models/{model}.yaml`)
5. **Dataset config** (`datasets/{dataset}.yaml`)
6. **Defaults** - Lowest priority

### Example 26: Understanding Priority

```bash
# This command:
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --dataset-config-file configs/datasets/llama3.1-8b.yaml \
    --input-column custom_input

# Loads config from llama3.1-8b.yaml BUT overrides input_column with "custom_input"
```

## Troubleshooting Examples

### Example 27: Dataset Config Not Found

```bash
# If config not found, system uses defaults
python harness/harness_main.py \
    --model my-model \
    --dataset-path ./my_dataset.pkl \
    --dataset-name my-dataset
    # Warning: No config found, using defaults
    # Defaults: input_column="input", input_ids_column="tok_input", output_column="output"
```

### Example 28: Invalid Endpoint Type

```bash
# This will fail
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --endpoint-type invalid_endpoint
    # Error: Invalid endpoint_type: invalid_endpoint. Must be 'completions' or 'chat_completions'
```

### Example 29: Endpoint Not Available for Backend

```bash
# If backend only supports completions
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --endpoint-type chat_completions \
    --server-config backend=completions-only-backend
    # Error: Endpoint 'chat_completions' is not available for backend 'completions-only-backend'
```

## Best Practices

### Example 30: Recommended Workflow

```bash
# 1. Create dataset config
# Edit configs/datasets/my-dataset.yaml

# 2. Test with defaults
python harness/harness_main.py \
    --model my-model \
    --dataset-path ./my_dataset.pkl \
    --dataset-name my-dataset

# 3. Add customizations if needed
python harness/harness_main.py \
    --model my-model \
    --dataset-path ./my_dataset.pkl \
    --dataset-name my-dataset \
    --input-column custom_column

# 4. Add metrics for monitoring
python harness/harness_main.py \
    --model my-model \
    --dataset-path ./my_dataset.pkl \
    --dataset-name my-dataset \
    --enable-metrics \
    --mlflow-experiment-name my-experiment
```

### Example 31: Testing Different Configurations

```bash
# Test with completions endpoint
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --endpoint-type completions \
    --output-dir ./results_completions

# Test with chat_completions endpoint
python harness/harness_main.py \
    --model llama3.1-8b \
    --dataset-path ./cnn_eval.json \
    --endpoint-type chat_completions \
    --output-dir ./results_chat_completions

# Compare results
```

## Quick Reference

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name or path | Required |
| `--dataset-path` | Path to dataset file | Required |
| `--dataset-name` | Dataset name for config lookup | Auto-detected |
| `--dataset-config-file` | Path to specific config YAML | Auto-detected |
| `--input-column` | Override input column name | From config |
| `--input-ids-column` | Override input_ids column name | From config |
| `--output-column` | Override output column name | From config |
| `--scenario` | Offline or Server | Offline |
| `--test-mode` | performance or accuracy | performance |
| `--endpoint-type` | completions or chat_completions | completions |
| `--api-server-url` | API server URL | None (start server) |
| `--batch-size` | Batch size | 13368 |
| `--num-samples` | Number of samples | 13368 |
| `--enable-metrics` | Enable metrics collection | False |
| `--mlflow-experiment-name` | MLflow experiment name | None |

