# Backend Configuration Directory

This directory contains YAML configuration files for different backends (vLLM, SGLang, etc.) that define available API endpoints.

## Purpose

Backend configurations validate that requested endpoints are available for a specific backend, preventing errors when trying to use endpoints that don't exist.

## Configuration Structure

Each backend configuration file follows this structure:

```yaml
name: backend-name
description: "Description of the backend"

endpoints:
  - completions
  - chat_completions

default_endpoint: completions
```

## Example Configurations

### vLLM (`vllm.yaml`)

```yaml
name: vllm
description: "vLLM backend configuration"

endpoints:
  - completions
  - chat_completions

default_endpoint: completions
```

### SGLang (`sglang.yaml`)

```yaml
name: sglang
description: "SGLang backend configuration"

endpoints:
  - completions
  - chat_completions

default_endpoint: completions
```

## Endpoint Validation

When a harness is initialized with an API server URL and endpoint type, the system:

1. Loads the backend configuration based on the backend name (from server_config)
2. Validates that the requested endpoint type exists in the `endpoints` list
3. Raises a `ValueError` if the endpoint is not available

### Example Error

If you try to use `chat_completions` with a backend that only supports `completions`:

```
ValueError: Endpoint 'chat_completions' is not available for backend 'some-backend'. 
Available endpoints: ['completions']
```

## Adding New Backends

To add a new backend:

1. Create `backends/my-backend.yaml`:
   ```yaml
   name: my-backend
   description: "My backend configuration"
   
   endpoints:
     - completions
     # Add other endpoints as needed
   
   default_endpoint: completions
   ```

2. Use in harness:
   ```python
   harness = BaseHarness(
       server_config={'backend': 'my-backend'},
       ...
   )
   ```

## Default Behavior

If no backend config file exists:
- Defaults to allowing both `completions` and `chat_completions` endpoints
- Logs a warning but continues execution
- This ensures backward compatibility

