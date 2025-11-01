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

### Example: Running Harness for Llama 3.1 8B

```python
from harness.harness_llama3.1_8b import Llama31_8BHarness

harness = Llama31_8BHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./dataset.json",
    scenario="Offline",
    test_mode="performance",
    api_server_url="http://localhost:8000",  # Or let harness start server
    batch_size=13368,
    num_samples=13368,
    enable_metrics=True
)

results = harness.run(user_conf="user.conf", lg_model_name="llama3_1-8b")
```

### Command Line Usage

```bash
python harness/harness_llama3.1_8b.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --enable-metrics \
    --user-conf user.conf \
    --lg-model-name llama3_1-8b
```

### Starting Server from Config

```bash
python harness/harness_llama3.1_8b.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./dataset.json \
    --scenario Offline \
    --server-config backendserver/example_server_config.yaml \
    --enable-metrics
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

