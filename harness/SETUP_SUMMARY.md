# Harness Setup Summary

This document summarizes the refactoring work completed to create the unified MLPerf Inference Harness.

## Completed Tasks

### 1. Directory Structure Created ✓

```
harness/
├── backendserver/          # Backend server management
├── Client/                 # Client implementations  
├── data/                   # Dataset processing
├── tools/                  # Moved from /tools
└── harness_llama3.1_8b.py # Example harness
```

### 2. Files Moved and Created ✓

- **Backend Server**: `inference_server.py` and related files moved to `harness/backendserver/`
- **Tools**: `tools/` directory moved to `harness/tools/`
- **SUT Copy**: `SUT_VLLM_SingleReplica.py` copied to `SUT_llama3.1-8b.py`

### 3. Client Infrastructure Created ✓

#### Base Client (`Client/base_client.py`)
- Abstract base class for all clients
- Standard interface: `initialize()`, `run()`, `cleanup()`
- Context manager support

#### LoadGen Client (`Client/loadgen_client.py`)
- Base `LoadGenClient` class
- `LoadGenOfflineClient`: Batch processing scenario
- `LoadGenServerClient`: Real-time query processing
- Factory function: `create_loadgen_client()`
- API server mode support
- Dataset integration

### 4. Dataset Processor Created ✓

#### Dataset Processor (`data/dataset_processor.py`)
- Supports multiple formats:
  - JSON (single objects, arrays, dictionaries)
  - Pickle (DataFrame, dict)
  - Pandas DataFrame (direct)
  - CSV (via pandas)
- Standardized output:
  - `input_ids`: List[List[int]] - Tokenized input IDs
  - `input`: List[str] - Input text strings
  - `input_lens`: List[int] - Input lengths
  - `targets`: List[Any] - Output/targets
- Compatible with LoadGen pattern:
  - `dataset.input_ids[index]` directly accessible
  - Same pattern as `Dataset` class in `SUT_VLLM_SingleReplica.py`

### 5. Example Harness Created ✓

#### Llama 3.1 8B Harness (`harness_llama3.1_8b.py`)
- Integrates all components:
  - Backend server management
  - LoadGen client (Offline/Server)
  - Dataset processing
  - Metrics collection (optional)
- Command-line interface
- YAML configuration support
- Inspired by `SUT_VLLM_SingleReplica.py` patterns

### 6. Documentation Created ✓

- `README.md`: Main documentation
- `ARCHITECTURE.md`: Architecture overview
- `SETUP_SUMMARY.md`: This file

## Integration Points

### Dataset Usage Pattern

The harness follows the same pattern as `SUT_VLLM_SingleReplica.py`:

**Original Pattern:**
```python
# In SUT_VLLM_SingleReplica.py
self.data_object = Dataset(model_name, dataset_path, total_sample_count=13368)
# Access via: self.data_object.input_ids[q.index]
```

**New Pattern (Harness):**
```python
# In LoadGen Client
self.dataset = DatasetProcessor(dataset_path, model_name, total_sample_count)
# Access via: self.dataset.input_ids[q_sample.index]
```

Both expose `input_ids` as a list accessible by index, maintaining compatibility.

### LoadGen Integration

The harness maintains compatibility with LoadGen:

1. **QSL Construction**:
   ```python
   qsl = lg.ConstructQSL(
       total_samples,      # Total samples in dataset
       num_samples,         # Number of samples for testing
       load_samples_to_ram,    # Callback (data pre-loaded)
       unload_samples_from_ram # Callback (no-op)
   )
   ```

2. **Query Processing**:
   ```python
   def issue_query(query_samples):
       for q_sample in query_samples:
           # q_sample.index corresponds to dataset index
           input_ids = self.dataset.input_ids[q_sample.index]
           # Process query...
   ```

3. **Response Handling**:
   ```python
   token_array = np.array(token_ids, dtype=np.int32)
   response = lg.QuerySampleResponse(
       query_id, 
       token_array.ctypes.data,  # Data pointer
       len(token_array.tobytes()),  # Size
       len(token_ids)              # Token count
   )
   lg.QuerySamplesComplete([response])
   ```

## Usage Examples

### Example 1: Using Harness with External Server

```python
from harness.harness_llama3.1_8b import Llama31_8BHarness

harness = Llama31_8BHarness(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./dataset.json",
    scenario="Offline",
    test_mode="performance",
    api_server_url="http://localhost:8000",  # External server
    batch_size=13368,
    num_samples=13368
)

results = harness.run()
```

### Example 2: Using Harness with Auto-Start Server

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
    }
)

results = harness.run()
```

### Example 3: Direct Client Usage

```python
from Client import create_loadgen_client
from data.dataset_processor import DatasetProcessor

# Initialize client
client = create_loadgen_client(
    scenario="Offline",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    dataset_path="./dataset.json",
    api_server_url="http://localhost:8000"
)

client.initialize()

# Use with LoadGen...
# client.issue_query(query_samples)
# client.flush_queries()

client.cleanup()
```

## Key Features

### Backend Server Features
- ✅ Start/stop with cleanup
- ✅ Heartbeat monitoring
- ✅ Debug mode (process cleanup verification)
- ✅ Profiling support (nsys, PyTorch, AMD)
- ✅ YAML configuration
- ✅ Dictionary/list argument formats
- ✅ Environment variable support

### Client Features
- ✅ Base client interface
- ✅ LoadGen Offline client
- ✅ LoadGen Server client
- ✅ API server mode support
- ✅ Extensible for new clients (GuideLLM, etc.)

### Dataset Features
- ✅ Multiple format support (JSON, Pickle, CSV)
- ✅ Auto-detection of format
- ✅ Standardized output format
- ✅ Statistics and metadata
- ✅ Compatible with LoadGen indexing pattern

## Files Created/Modified

### Created Files:
- `harness/backendserver/inference_server.py` (moved)
- `harness/backendserver/README_inference_server.md` (moved)
- `harness/backendserver/example_server_config.yaml` (moved)
- `harness/backendserver/tests/` (moved)
- `harness/Client/base_client.py` (new)
- `harness/Client/loadgen_client.py` (new)
- `harness/data/dataset_processor.py` (new)
- `harness/harness_llama3.1_8b.py` (new)
- `harness/README.md` (new)
- `harness/ARCHITECTURE.md` (new)
- `harness/SETUP_SUMMARY.md` (new)

### Copied Files:
- `language/llama3.1-8b/SUT_llama3.1-8b.py` (copy of SUT_VLLM_SingleReplica.py)

### Moved Directories:
- `tools/` → `harness/tools/`

## Testing

Tests are available in:
- `harness/backendserver/tests/test_inference_server.py` (unit tests)
- `harness/backendserver/tests/test_server_integration.py` (integration tests)

Run tests:
```bash
# Unit tests
python -m pytest harness/backendserver/tests/test_inference_server.py -v

# Integration tests
python -m pytest harness/backendserver/tests/test_server_integration.py -v
```

## Next Steps

1. **Testing**: Test the harness with actual datasets and servers
2. **GuideLLM Client**: Add GuideLLM client implementation
3. **Local Model Support**: Add local model processing to LoadGen clients
4. **Additional Backends**: Add support for more inference backends
5. **Documentation**: Expand documentation with more examples

## Compatibility Notes

- The harness is compatible with existing `SUT_VLLM_SingleReplica.py` patterns
- Dataset format follows the same structure (input_ids, input, targets)
- LoadGen integration maintains the same interface
- Can be used alongside existing SUT implementations

