# Harness Architecture

This document describes the architecture of the MLPerf Inference Harness.

## Directory Structure

```
harness/
├── backendserver/          # Backend inference server management
│   ├── inference_server.py  # InferenceServer base class and implementations
│   ├── README_inference_server.md
│   ├── example_server_config.yaml
│   └── tests/              # Server tests
├── Client/                 # Client implementations
│   ├── base_client.py      # BaseClient abstract class
│   ├── loadgen_client.py   # LoadGen client (Offline/Server scenarios)
│   └── __init__.py
├── data/                   # Dataset processing
│   ├── dataset_processor.py  # Generic dataset processor
│   └── __init__.py
├── harness_llama3.1_8b.py # Example harness for Llama 3.1 8B
├── README.md               # Main README
└── ARCHITECTURE.md         # This file
```

## Component Overview

### 1. Backend Server (`backendserver/`)

Manages inference servers (vLLM, SGLang, etc.):

**Key Features:**
- Start/stop server functionality with cleanup
- Heartbeat monitoring (configurable)
- Debug mode for process cleanup verification (especially useful for tensor/data parallel)
- Profiling support (nsys, PyTorch, AMD profiler)
- YAML configuration support
- Environment variable support (CLI/YAML)
- Dictionary and list format for server arguments

**Usage:**
```python
from backendserver import VLLMServer

server = VLLMServer(
    model="meta-llama/Llama-3.1-8B-Instruct",
    port=8000,
    debug_mode=True,
    config={
        'api_server_args': {
            'tensor-parallel-size': '4',
            'trust-remote-code': True,
            'kv-cache-dtype': {'auto': 'fp8'}
        }
    }
)
server.start()
```

### 2. Client (`Client/`)

Client implementations for different benchmarking frameworks:

#### Base Client (`base_client.py`)
- Abstract base class defining the client interface
- Methods: `initialize()`, `run()`, `cleanup()`
- Context manager support

#### LoadGen Client (`loadgen_client.py`)
- MLPerf LoadGen integration
- **Offline Client**: Batch processing scenario
  - Processes queries in batches
  - Suitable for throughput testing
- **Server Client**: Real-time query processing
  - Processes queries individually or in small batches
  - Suitable for latency testing
- Supports API server mode (remote inference)
- Handles dataset loading and query processing

**Usage:**
```python
from Client import create_loadgen_client

client = create_loadgen_client(
    scenario="Offline",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./dataset.json",
    api_server_url="http://localhost:8000",
    batch_size=13368,
    num_samples=13368
)

client.initialize()
# Use with LoadGen...
client.cleanup()
```

### 3. Dataset Processing (`data/`)

Generic dataset processor supporting multiple formats:

**Supported Formats:**
- JSON (single objects, arrays, or dictionaries)
- Pickle (DataFrame or dictionary)
- Pandas DataFrame (direct)
- CSV (via pandas)

**Standardized Output:**
- `input`: List of input text strings
- `input_ids`: List of tokenized input IDs (list of lists)
- `input_lens`: List of input lengths
- `targets`: List of output/targets

**Usage:**
```python
from data.dataset_processor import DatasetProcessor

dataset = DatasetProcessor(
    dataset_path="./dataset.json",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    total_sample_count=13368
)

# Access data
input_ids = dataset.input_ids[0]  # Get first sample's input IDs
sample = dataset.get_sample(0)     # Get full sample dictionary
stats = dataset.get_statistics()   # Get dataset statistics
```

## Integration Flow

### LoadGen Integration Pattern

The harness follows the same pattern as `SUT_VLLM_SingleReplica.py`:

1. **Dataset Loading**:
   ```python
   dataset = DatasetProcessor(dataset_path="...")
   # Dataset exposes input_ids as a list: dataset.input_ids[index]
   ```

2. **LoadGen QSL Construction**:
   ```python
   qsl = lg.ConstructQSL(
       total_samples,      # Total samples in dataset
       num_samples,         # Number of samples for testing
       load_samples_to_ram,    # Callback (no-op, data pre-loaded)
       unload_samples_from_ram # Callback (no-op)
   )
   ```

3. **Query Processing**:
   ```python
   def issue_query(query_samples):
       for q_sample in query_samples:
           # q_sample.index corresponds to dataset index
           input_ids = dataset.input_ids[q_sample.index]
           # Process query...
   ```

4. **Response Handling**:
   ```python
   response = lg.QuerySampleResponse(
       query_id, 
       response_data,  # Token array data pointer
       response_size,   # Size in bytes
       token_count      # Number of tokens
   )
   lg.QuerySamplesComplete([response])
   ```

## Example: Full Harness Usage

### Using the Harness Class

```python
from harness.harness_llama3.1_8b import Llama31_8BHarness

harness = Llama31_8BHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./dataset.json",
    scenario="Offline",
    test_mode="performance",
    api_server_url="http://localhost:8000",  # Or use server_config
    batch_size=13368,
    num_samples=13368,
    enable_metrics=True
)

results = harness.run(user_conf="user.conf", lg_model_name="llama3_1-8b")
```

### Starting Server Automatically

```python
harness = Llama31_8BHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./dataset.json",
    scenario="Offline",
    server_config={
        'backend': 'vllm',
        'port': 8000,
        'config': {
            'api_server_args': {
                'tensor-parallel-size': '4',
                'trust-remote-code': True
            }
        },
        'debug_mode': True
    },
    enable_metrics=True
)

results = harness.run()
```

## Data Flow

```
1. Dataset File (JSON/Pickle/CSV)
   ↓
2. DatasetProcessor
   → Standardized format (input_ids, input, targets)
   ↓
3. LoadGen Client
   → Receives QuerySample objects with index
   → Accesses dataset.input_ids[q_sample.index]
   → Processes queries via API or local model
   → Sends responses back to LoadGen
   ↓
4. Inference Server (vLLM/SGLang)
   → Receives API requests
   → Processes inference
   → Returns responses
   ↓
5. Metrics Collection (optional)
   → Collects server metrics
   → Generates visualizations
```

## Extensibility

### Adding New Backends

1. Create server class in `backendserver/`:
   ```python
   class MyBackendServer(InferenceServer):
       def get_backend_name(self):
           return "mybackend"
       
       def get_launch_command(self):
           return [...]
       
       # Implement other abstract methods...
   ```

2. Register in `backendserver/__init__.py`

3. Update `create_server()` factory function

### Adding New Clients

1. Create client class in `Client/`:
   ```python
   class MyNewClient(BaseClient):
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

2. Register in `Client/__init__.py`

## Key Design Decisions

1. **Separation of Concerns**:
   - Backend servers are separate from clients
   - Dataset processing is independent
   - Metrics collection is optional

2. **Compatibility**:
   - Follows same patterns as `SUT_VLLM_SingleReplica.py`
   - Uses same dataset format (input_ids as list)
   - Compatible with existing LoadGen code

3. **Flexibility**:
   - Supports both API and local modes
   - Multiple dataset formats
   - Configurable via Python or YAML

4. **Debugging**:
   - Debug mode for process cleanup verification
   - Comprehensive logging
   - Metrics collection and visualization

## Migration from SUT_VLLM_SingleReplica

The harness provides a more modular approach:

**Old (SUT_VLLM_SingleReplica.py):**
- Single file with all functionality
- Hardcoded dataset format (JSON -> pandas)
- Server management embedded in SUT classes

**New (Harness):**
- Modular components (server, client, dataset)
- Generic dataset processor (multiple formats)
- Separate server management
- Reusable client infrastructure

Both can coexist - the harness is a refactored, more maintainable version.

