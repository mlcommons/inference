# Inference Server Module

This module provides a unified interface for managing inference servers (vLLM, SGLang, etc.) with start/stop functionality, heartbeat monitoring, endpoint discovery, and profiling support.

## Features

- **Multiple Backend Support**: Supports vLLM, SGLang, and custom backends
- **Automatic Server Management**: Start/stop servers with proper cleanup
- **Heartbeat Monitoring**: Configurable health checks
- **Endpoint Discovery**: Automatic endpoint listing for each backend
- **Environment Variable Support**: Set environment variables via CLI or YAML
- **Profiling Integration**: Support for nsys, PyTorch, and AMD profilers
- **YAML Configuration**: Configure servers via YAML files
- **Context Manager Support**: Use servers as context managers
- **Debug Mode**: Verify cleanup of server and child processes (especially useful for tensor/data parallel setups)

## Quick Start

### Using the Module Directly

```python
from inference_server import VLLMServer, SGLangServer, create_server

# Create and start a vLLM server
server = VLLMServer(
    model="meta-llama/Llama-2-7b-hf",
    output_dir="./server_logs",
    port=8000
)

server.start()
# ... use the server ...
server.stop()
```

### Using Context Manager

```python
from inference_server import VLLMServer

with VLLMServer(model="meta-llama/Llama-2-7b-hf", port=8000) as server:
    # Server is automatically started
    print(server.list_endpoints())
    # ... use the server ...
# Server is automatically stopped when exiting context
```

### Using Factory Function

```python
from inference_server import create_server

# Create server based on backend name
server = create_server(
    backend="vllm",  # or "sglang"
    model="meta-llama/Llama-2-7b-hf",
    port=8000
)

server.start()
server.stop()
```

### Using YAML Configuration

```python
from inference_server import start_server_from_config

# Start server from YAML config file
server = start_server_from_config("config.yaml")
# Server is automatically started

# ... use the server ...

server.stop()
```

## YAML Configuration Format

See `example_server_config.yaml` for a complete example.

```yaml
backend: vllm
model: meta-llama/Llama-2-7b-hf
port: 8000
output_dir: ./server_logs

env_vars:
  CUDA_VISIBLE_DEVICES: "0"
  OMP_NUM_THREADS: "16"

config:
  api_server_args:
    - --tensor-parallel-size
    - "1"

profile:
  enabled: true
  tool: nsys
  output_dir: ./profiles
  args:
    - --trace=cuda,nvtx
```

## Command-Line Usage

You can use the module directly from the command line:

```bash
# Start server from YAML config
python -c "from inference_server import start_server_from_config; \
           server = start_server_from_config('config.yaml'); \
           import time; time.sleep(60); server.stop()"
```

## API Reference

### `InferenceServer` (Base Class)

Base class for all inference servers.

**Methods:**
- `start()`: Start the server
- `stop()`: Stop the server
- `check_health()`: Check if server is healthy
- `list_endpoints()`: List available API endpoints
- `get_backend_name()`: Get backend name
- `get_launch_command()`: Get launch command
- `get_health_endpoint()`: Get health check URL

### `VLLMServer`

vLLM inference server implementation.

### `SGLangServer`

SGLang inference server implementation.

### Factory Functions

- `create_server(backend, model, **kwargs)`: Create a server instance
- `load_server_config(config_file)`: Load YAML configuration
- `start_server_from_config(config_file)`: Start server from YAML config

## Configuration Options

### Server Configuration

- `model`: Model name or path (required)
- `output_dir`: Directory for server logs (default: `./server_logs`)
- `port`: Server port (default: `8000`)
- `heartbeat_interval`: Heartbeat check interval in seconds (default: `5`)
- `heartbeat_timeout`: Heartbeat check timeout in seconds (default: `30`)
- `startup_timeout`: Server startup timeout in seconds (default: `600`)
- `env_vars`: Dictionary of environment variables
- `config`: Backend-specific configuration dictionary
- `debug_mode`: Enable debug mode for process cleanup verification (default: `False`)

### Debug Mode

When `debug_mode=True`, the server will track and verify cleanup of all child processes. This is especially useful for tensor parallel and data parallel setups where the server spawns multiple worker processes.

```python
server = VLLMServer(
    model="meta-llama/Llama-2-7b-hf",
    port=8000,
    debug_mode=True,  # Enable cleanup verification
    config={
        'api_server_args': [
            '--tensor-parallel-size', '4'  # 4 GPUs = 4 worker processes
        ]
    }
)
```

When the server stops, debug mode will verify:
- Main server process cleanup
- Process group cleanup
- All tracked child processes cleanup
- System-wide process scanning for orphaned processes

### Profiling Configuration

The module supports three profiling tools:

1. **NSight Systems (nsys)**: NVIDIA profiling tool
   ```yaml
   profile:
     enabled: true
     tool: nsys
     output_dir: ./profiles
     args: [--trace=cuda,nvtx]
   ```

2. **PyTorch Profiler**: Built-in PyTorch profiling
   ```yaml
   profile:
     enabled: true
     tool: pytorch
     output_dir: ./profiles
   ```

3. **AMD Profiler (rocprof)**: AMD profiling tool
   ```yaml
   profile:
     enabled: true
     tool: amd
     output_dir: ./profiles
     args: [--stats]
   ```

## Passing Arguments to Servers

### Via Configuration (Recommended)

When creating a server instance, pass arguments through the `config` parameter:

#### vLLM Server Arguments

The `api_server_args` can be specified in multiple formats:

**Option 1: List format (traditional)**
```python
from inference_server import VLLMServer

server = VLLMServer(
    model="meta-llama/Llama-2-7b-hf",
    port=8000,
    config={
        'api_server_args': [
            '--tensor-parallel-size', '4',      # 4 GPUs for tensor parallelism
            '--pipeline-parallel-size', '1',    # Pipeline parallelism
            '--gpu-memory-utilization', '0.9',  # GPU memory usage
            '--max-model-len', '8192',           # Maximum sequence length
            '--max-num-seqs', '512',            # Maximum concurrent sequences
            '--max-num-batched-tokens', '8192', # Maximum batched tokens
            '--dtype', 'bfloat16',             # Model dtype
            '--trust-remote-code',              # Flag (no value needed)
            '--disable-log-requests',           # Flag (no value needed)
        ]
    }
)
server.start()
```

**Option 2: Dictionary format (recommended)**
```python
from inference_server import VLLMServer

server = VLLMServer(
    model="openai/gpt-oss-120b",
    port=8000,
    env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    config={
        'api_server_args': {
            '--tensor-parallel-size': '4',      # String value
            'tensor-parallel-size': '4',       # Also works without --
            'gpu-memory-utilization': 0.9,     # Numeric value
            'max-model-len': 8192,              # Integer value
            'trust-remote-code': True,         # Flag (True = include flag)
            'disable-log-requests': True,      # Flag (True = include flag)
            'kv-cache-dtype': {                # Dictionary value (will be JSON encoded)
                'auto': 'fp8'
            },
        }
    }
)
server.start()
```

**Option 3: Mixed format**
```python
from inference_server import VLLMServer

server = VLLMServer(
    model="meta-llama/Llama-2-7b-hf",
    port=8000,
    config={
        'api_server_args': {
            '--tensor-parallel-size': '4',
            '--gpu-memory-utilization': '0.9',
            '--trust-remote-code': True,        # Flag
            '--kv-cache-dtype': {               # Dict value
                'auto': 'fp8'
            }
        }
    }
)
server.start()
```

#### SGLang Server Arguments

```python
from inference_server import SGLangServer

server = SGLangServer(
    model="meta-llama/Llama-2-7b-hf",
    port=8000,
    config={
        'server_args': [
            '--tp', '4',                        # Tensor parallelism size
            '--context-length', '8192',          # Context length
            '--mem-fraction-static', '0.9',     # Memory fraction
        ]
    }
)
server.start()
```

### Via YAML Configuration

YAML supports both list and dictionary formats for arguments:

**List format:**
```yaml
backend: vllm
model: meta-llama/Llama-2-7b-hf
port: 8000

config:
  api_server_args:
    - --tensor-parallel-size
    - "4"
    - --gpu-memory-utilization
    - "0.9"
    - --max-model-len
    - "8192"
    - --trust-remote-code
    - --disable-log-requests
```

**Dictionary format (recommended):**
```yaml
backend: vllm
model: openai/gpt-oss-120b
port: 8000

config:
  api_server_args:
    tensor-parallel-size: "4"
    gpu-memory-utilization: 0.9
    max-model-len: 8192
    trust-remote-code: true          # Flag (true = include flag)
    disable-log-requests: true       # Flag (true = include flag)
    kv-cache-dtype:                  # Dictionary value (will be JSON encoded)
      auto: "fp8"

# For SGLang, use server_args:
# config:
#   server_args:
#     - --tp
#     - "4"
#     - --context-length
#     - "8192"
```

### Via API Calls

Once the server is running, you can pass arguments through API calls. However, note that server arguments (like `--tensor-parallel-size`) must be set at server startup. Request-level arguments can be passed in API calls:

#### vLLM API Request Example

```python
import requests

# Server must be started with tensor parallel size already configured
# But you can pass per-request parameters:

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "Hello, world!",
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stop": ["\n"],
        "stream": False
    }
)
```

#### SGLang API Request Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-2-7b-hf",
        "prompt": "Hello, world!",
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False
    }
)
```

### Common Server Arguments

#### vLLM Common Arguments:
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--pipeline-parallel-size`: Number of stages for pipeline parallelism
- `--gpu-memory-utilization`: Fraction of GPU memory to use (0.0-1.0)
- `--max-model-len`: Maximum sequence length
- `--max-num-seqs`: Maximum number of concurrent sequences
- `--max-num-batched-tokens`: Maximum tokens per batch
- `--dtype`: Model data type (float16, bfloat16, float32)
- `--trust-remote-code`: Allow execution of remote code
- `--disable-log-requests`: Disable request logging for performance
- `--enforce-eager`: Use eager mode (slower but more compatible)

#### SGLang Common Arguments:
- `--tp`: Tensor parallelism size
- `--context-length`: Maximum context length
- `--mem-fraction-static`: Fraction of memory for static allocation
- `--mem-fraction-reserved`: Fraction of memory reserved
- `--trust-remote-code`: Trust remote code

### Example: Tensor Parallel Setup with Dictionary Arguments

```python
from inference_server import VLLMServer

# Start server with 4 GPUs using tensor parallelism
server = VLLMServer(
    model="openai/gpt-oss-120b",
    port=8000,
    debug_mode=True,  # Track all 4 worker processes
    env_vars={
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    },
    config={
        'api_server_args': {
            'tensor-parallel-size': '4',         # Use dashes or underscores
            'gpu-memory-utilization': 0.9,      # Numeric value
            'max-model-len': 4096,              # Integer value
            'trust-remote-code': True,           # Flag
            'disable-log-requests': True,       # Flag
            'kv-cache-dtype': {                 # Dictionary argument
                'auto': 'fp8'
            }
        }
    }
)

server.start()
# Server will spawn 4 worker processes (one per GPU)

# ... make API calls ...

server.stop()
# Debug mode will verify all 4 worker processes are cleaned up
```

### Argument Format Guidelines

**Dictionary format supports:**
- String values: `'tensor-parallel-size': '4'`
- Numeric values: `'gpu-memory-utilization': 0.9`
- Integer values: `'max-model-len': 8192`
- Boolean flags: `'trust-remote-code': True` (flag is included)
- Dictionary values: `'kv-cache-dtype': {'auto': 'fp8'}` (JSON encoded)
- Underscores converted to dashes: `'tensor_parallel_size'` → `--tensor-parallel-size`

**List format supports:**
- Traditional format: `['--arg', 'value', '--flag']`
- Flags without values: `['--trust-remote-code']`

## Extending the Module

To add a new backend, create a subclass of `InferenceServer`:

```python
class MyBackendServer(InferenceServer):
    def get_backend_name(self):
        return "mybackend"
    
    def get_binary_path(self):
        return "my-backend-binary"
    
    def get_launch_command(self):
        return [self.binary_path, "--model", self.model, "--port", str(self.port)]
    
    def get_health_endpoint(self):
        return f"{self.server_url}/health"
    
    def list_endpoints(self):
        return {
            "health": f"{self.server_url}/health",
            "api": f"{self.server_url}/api",
        }
```

## Testing

See the `tests/` directory for unit and integration tests.

Run tests:
```bash
# Unit tests
python -m pytest tests/test_inference_server.py -v

# Integration tests (requires server binaries)
python -m pytest tests/test_server_integration.py -v

# Skip integration tests
SKIP_INTEGRATION_TESTS=1 python -m pytest tests/test_server_integration.py -v
```

## Requirements

- Python 3.7+
- requests (for HTTP health checks)
- PyYAML (for YAML configuration support; optional if not using YAML configs)

Install dependencies:
```bash
pip install requests pyyaml
```

For integration tests:
- vLLM or SGLang installed (depending on backend)
- Access to model files

