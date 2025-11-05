# MLPerf Inference Harness

A unified harness framework for managing inference servers and clients in MLPerf Inference benchmarks.

## Directory Structure

```
harness/
├── backendserver/          # Backend inference server management
│   ├── inference_server.py  # Base InferenceServer class and implementations
│   ├── README_inference_server.md
│   ├── example_server_config.yaml
│   └── tests/              # Server tests
├── Client/                 # Client implementations
│   ├── base_client.py      # Base client class
│   ├── loadgen_client.py   # LoadGen client (Offline/Server)
│   └── __init__.py
├── data/                   # Dataset processing
│   ├── dataset_processor.py  # Generic dataset processor
│   └── __init__.py
├── metrics/                # Metrics collection and visualization
│   ├── vllm_metrics_collector.py
│   ├── vllm_metrics_visualizer.py
│   └── test/               # Metrics tests
├── harness_llama3.1_8b.py # Example harness for Llama 3.1 8B
└── README.md               # This file
```

## Components

### 1. Backend Server (`backendserver/`)

Manages inference servers (vLLM, SGLang, etc.) with:
- Start/stop functionality
- Heartbeat monitoring
- Process cleanup verification (debug mode)
- Profiling support (nsys, PyTorch, AMD)
- YAML configuration support

See `backendserver/README_inference_server.md` for details.

### 2. Clients (`Client/`)

Client implementations for different benchmarking frameworks:

#### Base Client (`base_client.py`)
- Abstract base class for all clients
- Standard interface: `initialize()`, `run()`, `cleanup()`
- Context manager support

#### LoadGen Client (`loadgen_client.py`)
- MLPerf LoadGen integration
- **Offline Client**: Batch processing scenario
- **Server Client**: Real-time query processing scenario
- Supports API server mode (remote inference)
- Handles dataset loading and query processing

### 3. Dataset Processing (`data/`)

Generic dataset processor that handles:
- **JSON files**: Single objects or arrays
- **Pickle files**: DataFrames or dictionaries
- **Pandas DataFrames**: Direct DataFrame objects
- **CSV files**: Via pandas

Converts to standardized format:
- `input`: List of input text strings
- `input_ids`: List of tokenized input IDs
- `input_lens`: List of input lengths
- `targets`: List of output/targets

## Usage

### Quick Start Examples

#### Example 1: Basic Usage with Model-Specific Harness

```bash
# Using Llama 3.1 8B harness (auto-loads llama3.1-8b.yaml config)
# Harness will start server automatically if --api-server-url is not provided
python harness/harness_llama3.1_8b.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --test-mode performance \
    --batch-size 13368 \
    --num-samples 13368 \
    --output-dir ./harness_output
```

#### Example 1a: Using Existing Backend Server

```bash
# Connect to existing backend server
python harness/harness_llama3.1_8b.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --batch-size 13368
```

#### Example 2: Using Model Category

```bash
# Use model category to select harness, and model name for the actual model
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline
```

#### Example 2a: Using Llama2-70B Category

```bash
# Use llama2-70b category with specific model
python harness/harness_main.py \
    --model-category llama2-70b \
    --model meta-llama/Llama-2-70B-Instruct \
    --dataset-path ./dataset.pkl \
    --scenario Offline
```

#### Example 2b: Auto-Detection (Backward Compatibility)

```bash
# Auto-detects model category from model name (backward compatibility)
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline
```

#### Example 3: Generic Harness with Explicit Configuration

```bash
# Use BaseHarness directly with dataset name
python harness/harness_main.py \
    --use-generic \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --scenario Offline
```

### Advanced Configuration Examples

#### Example 4: Specifying Custom Dataset Config File

```bash
# Use model category with custom dataset config file
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./my_dataset.pkl \
    --dataset-config-file datasets/my-custom-dataset.yaml \
    --scenario Offline
```

#### Example 5: Overriding Column Mappings Programmatically

```bash
# Override column names from command line
python harness/harness_main.py \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --dataset-path ./dataset.pkl \
    --input-column prompt \
    --input-ids-column token_ids \
    --output-column target \
    --scenario Offline
```

#### Example 6: Using Chat Completions Endpoint

```bash
# Use model category with chat completions endpoint
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --api-server-url http://localhost:8000 \
    --endpoint-type chat_completions \
    --scenario Offline
```

#### Example 7: Server Scenario with Metrics

```bash
# Server scenario with metrics collection and MLflow
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --enable-metrics \
    --mlflow-experiment-name llama3.1-8b-server \
    --server-target-qps 100.0
```

#### Example 8: Using Server Config File

```bash
# Start server from YAML config file (includes backend, model, port, etc.)
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --test-mode performance \
    --enable-metrics \
    --output-dir ./harness_output
```

#### Example 8a: Server Config with vLLM Backend

```bash
# Using vLLM backend with custom configuration
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --endpoint-type completions \
    --batch-size 13368 \
    --num-samples 13368 \
    --enable-metrics
```

#### Example 8b: Server Config with SGLang Backend

```bash
# Using SGLang backend (update server config to use sglang backend)
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/sglang_server_config.yaml \
    --scenario Offline \
    --endpoint-type chat_completions \
    --batch-size 1000
```

#### Example 9: DeepSeek R1 with Custom Settings

```bash
# DeepSeek R1 with custom dataset and endpoint
python language/deepseek-r1/harness_deepseek_r1.py \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --dataset-path ./deepseek_dataset.pkl \
    --dataset-name deepseek-r1 \
    --api-server-url http://localhost:8000 \
    --endpoint-type chat_completions \
    --scenario Offline \
    --batch-size 4388 \
    --num-samples 4388
```

#### Example 10: Combining All Options

```bash
# Comprehensive example with all features including backend server configuration
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --dataset-config-file configs/datasets/llama3.1-8b.yaml \
    --input-column input \
    --input-ids-column tok_input \
    --output-column output \
    --scenario Offline \
    --test-mode performance \
    --server-config backendserver/example_server_config.yaml \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --batch-size 13368 \
    --num-samples 13368 \
    --enable-metrics \
    --mlflow-experiment-name my-experiment \
    --mlflow-host localhost \
    --mlflow-port 5000 \
    --mlflow-output-dir ./harness_output \
    --output-dir ./harness_output \
    --user-conf user.conf \
    --log-level INFO
```

#### Example 10a: Backend Server Auto-Start (No API URL)

```bash
# Harness will automatically start server from config file
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --test-mode performance \
    --batch-size 13368 \
    --num-samples 13368 \
    --enable-metrics \
    --output-dir ./harness_output
```

#### Example 10b: Server Scenario with Backend Configuration

```bash
# Server scenario with backend server configuration and QPS settings
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --server-target-qps 100.0 \
    --server-coalesce-queries true \
    --enable-metrics \
    --mlflow-experiment-name server-test \
    --output-dir ./server_output
```

#### Example 10c: Using Llama2-70B with All Options

```bash
# Comprehensive example with llama2-70b category
python harness/harness_main.py \
    --model-category llama2-70b \
    --model meta-llama/Llama-2-70B-Instruct \
    --dataset-path ./dataset.pkl \
    --dataset-name llama2-70b \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --batch-size 1000 \
    --num-samples 1000 \
    --enable-metrics \
    --output-dir ./harness_output
```

### Python API Examples

#### Example 11: Using BaseHarness Directly

```python
from harness.base_harness import BaseHarness

# Basic usage with auto-detected dataset config
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    dataset_name="llama3.1-8b",  # Auto-loads configs/datasets/llama3.1-8b.yaml
    scenario="Offline",
    test_mode="performance",
    batch_size=13368,
    num_samples=13368,
    output_dir="./harness_output"
)

results = harness.run(user_conf="user.conf", lg_model_name="llama3_1-8b")
```

#### Example 11a: BaseHarness with Backend Server Config

```python
from harness.base_harness import BaseHarness

# Using backend server configuration file
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    dataset_name="llama3.1-8b",
    server_config="backendserver/example_server_config.yaml",  # Server config file
    scenario="Offline",
    test_mode="performance",
    batch_size=13368,
    num_samples=13368,
    enable_metrics=True,
    output_dir="./harness_output"
)

results = harness.run(user_conf="user.conf", lg_model_name="llama3_1-8b")
```

#### Example 11b: BaseHarness with Existing Server

```python
from harness.base_harness import BaseHarness

# Connect to existing backend server
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    dataset_name="llama3.1-8b",
    api_server_url="http://localhost:8000",  # Use existing server
    server_config={
        'backend': 'vllm',
        'endpoint_type': 'completions'
    },
    scenario="Offline",
    test_mode="performance",
    batch_size=13368,
    num_samples=13368
)

results = harness.run(user_conf="user.conf", lg_model_name="llama3_1-8b")
```

#### Example 12: Custom Dataset Config File

```python
from harness.base_harness import BaseHarness

# Use specific config file
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./my_dataset.pkl",
    dataset_config_file="configs/datasets/my-dataset.yaml",  # Custom config
    scenario="Offline"
)

results = harness.run()
```

#### Example 13: Programmatic Column Overrides

```python
from harness.base_harness import BaseHarness

# Override columns programmatically
harness = BaseHarness(
    model_name="deepseek-ai/DeepSeek-R1-0528",
    dataset_path="./dataset.pkl",
    input_column="prompt",  # Override input column
    input_ids_column="token_ids",  # Override input_ids column
    output_column="target",  # Override output column
    scenario="Offline"
)

results = harness.run()
```

#### Example 14: Using Chat Completions Endpoint

```python
from harness.base_harness import BaseHarness

# Use chat completions endpoint
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    api_server_url="http://localhost:8000",
    server_config={
        'backend': 'vllm',
        'endpoint_type': 'chat_completions'  # Use chat completions
    },
    scenario="Offline"
)

results = harness.run()
```

#### Example 15: Server Scenario with All Features

```python
from harness.base_harness import BaseHarness

# Server scenario with metrics and MLflow
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    dataset_name="llama3.1-8b",
    scenario="Server",
    test_mode="performance",
    api_server_url="http://localhost:8000",
    server_config={
        'backend': 'vllm',
        'endpoint_type': 'completions',
    },
    server_target_qps=100.0,
    server_coalesce_queries=True,
    enable_metrics=True,
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="llama3.1-8b-server",
    batch_size=100,
    num_samples=1000,
    output_dir="./server_output"
)

results = harness.run()
```

#### Example 15a: Server Scenario with Backend Config File

```python
from harness.base_harness import BaseHarness

# Server scenario using backend server config file
harness = BaseHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    dataset_name="llama3.1-8b",
    server_config="backendserver/example_server_config.yaml",  # Backend config
    scenario="Server",
    test_mode="performance",
    api_server_url="http://localhost:8000",  # Or let harness start server
    server_target_qps=100.0,
    server_coalesce_queries=True,
    enable_metrics=True,
    mlflow_tracking_uri="http://localhost:5000",
    mlflow_experiment_name="llama3.1-8b-server",
    output_dir="./server_output"
)

results = harness.run()
```

#### Example 16: Model-Specific Harness

```python
from harness.harness_llama3.1_8b import Llama31_8BHarness

# Use model-specific harness (extends BaseHarness)
harness = Llama31_8BHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./cnn_eval.json",
    # dataset_name="llama3.1-8b" is auto-set
    scenario="Offline",
    test_mode="performance",
    api_server_url="http://localhost:8000",
    enable_metrics=True
)

results = harness.run(user_conf="user.conf", lg_model_name="llama3_1-8b")
```

### Configuration File Examples

#### Example 17: Creating Custom Dataset Config

Create `configs/datasets/my-dataset.yaml`:

```yaml
name: my-dataset
description: "My custom dataset configuration"

fields:
  input_column: "prompt"
  input_ids_column: "token_ids"
  output_column: "target"
  input_lens_column: null  # Will be calculated

file_format: "pickle"  # or "json", "csv", "auto"

total_sample_count: 10000

model_specific:
  default_model_name: "my-model/MyModel"
```

Then use it:

```bash
python harness/harness_main.py \
    --model my-model/MyModel \
    --dataset-path ./my_dataset.pkl \
    --dataset-name my-dataset
```

#### Example 18: Backend-Specific Endpoint Configuration

Create `configs/backends/my-backend.yaml`:

```yaml
name: my-backend
description: "My custom backend configuration"

endpoints:
  - completions
  # Only completions endpoint available

default_endpoint: completions
```

The system will validate that only `completions` endpoint is used with this backend.

### Common Use Cases

#### Use Case 1: Quick Performance Test

```bash
# Quick test with existing server
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --batch-size 13368 \
    --num-samples 13368 \
    --output-dir ./quick_test
```

#### Use Case 1a: Quick Test with Auto-Start Server

```bash
# Quick test - harness starts server automatically
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --test-mode performance \
    --batch-size 13368 \
    --num-samples 13368
```

#### Use Case 2: Accuracy Testing

```bash
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Offline \
    --test-mode accuracy \
    --api-server-url http://localhost:8000
```

#### Use Case 3: Server Latency Testing

```bash
# Server latency testing with backend server
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --scenario Server \
    --test-mode performance \
    --server-config backendserver/example_server_config.yaml \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --server-target-qps 50.0 \
    --server-coalesce-queries true \
    --enable-metrics \
    --output-dir ./latency_test
```

#### Use Case 4: Testing with Different Endpoints

```bash
# Test with completions endpoint (default)
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --endpoint-type completions \
    --api-server-url http://localhost:8000 \
    --batch-size 13368 \
    --output-dir ./test_completions

# Test with chat_completions endpoint
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --server-config backendserver/example_server_config.yaml \
    --endpoint-type chat_completions \
    --api-server-url http://localhost:8000 \
    --batch-size 13368 \
    --output-dir ./test_chat_completions
```

#### Use Case 5: Using Different Models

```bash
# Llama 3.1 8B
python harness/harness_main.py \
    --model-category llama3.1-8b \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json

# Llama 2 70B
python harness/harness_main.py \
    --model-category llama2-70b \
    --model meta-llama/Llama-2-70B-Instruct \
    --dataset-path ./dataset.pkl

# DeepSeek R1
python language/deepseek-r1/harness_deepseek_r1.py \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --dataset-path ./deepseek_dataset.pkl
```

## Dataset Format

The dataset processor supports multiple formats:

### JSON Format

```json
[
  {
    "input": "Question: What is...",
    "tok_input": [1234, 5678, ...],
    "output": "Answer: ..."
  },
  ...
]
```

Or as a dictionary:
```json
{
  "input": ["Question 1", "Question 2", ...],
  "tok_input": [[1234, ...], [5678, ...], ...],
  "output": ["Answer 1", "Answer 2", ...]
}
```

### Pickle Format

```python
import pandas as pd

df = pd.DataFrame({
    'input': [...],
    'tok_input': [...],
    'output': [...]
})
df.to_pickle('dataset.pkl')
```

## Architecture

### Component Flow

```
Harness
  ├── Backend Server (vLLM/SGLang)
  │   └── Manages inference server lifecycle
  ├── Dataset Processor
  │   └── Loads and processes datasets
  └── LoadGen Client
      ├── Offline Client (batch processing)
      └── Server Client (real-time queries)
```

### Integration with SUT_VLLM_SingleReplica

The harness takes inspiration from `SUT_VLLM_SingleReplica.py`, specifically:
- `VLLMSingleSUTAPI` for API server communication
- Dataset loading from `Dataset` class
- Query processing and response handling
- LoadGen integration patterns

## Extending the Harness

### Adding a New Client

1. Create a new client class in `Client/`:

```python
from Client.base_client import BaseClient

class MyNewClient(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__("myclient", *args, **kwargs)
    
    def initialize(self):
        # Initialize client
        pass
    
    def run(self):
        # Run client logic
        pass
    
    def cleanup(self):
        # Cleanup resources
        pass
```

2. Update `Client/__init__.py` to export the new client.

### Adding a New Backend

1. Create a new server class in `backendserver/`:

```python
from backendserver.inference_server import InferenceServer

class MyBackendServer(InferenceServer):
    def get_backend_name(self):
        return "mybackend"
    
    # Implement required methods...
```

2. Update `backendserver/__init__.py` to export the new server.

## Testing

See individual component directories for testing:
- `backendserver/tests/` - Server tests
- Test harness with sample datasets

## Requirements

- Python 3.7+
- mlperf_loadgen
- requests
- pandas (for dataset processing)
- PyYAML (for server configs)
- transformers (for tokenization)
- psutil (optional, for debug mode)

