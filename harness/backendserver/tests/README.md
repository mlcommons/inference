# Inference Server Tests

This directory contains tests for the `inference_server` module.

## Test Files

### `test_inference_server.py`
Unit tests for the inference server module. These tests don't require actual server binaries and can be run quickly.

**Run with:**
```bash
python -m pytest tests/test_inference_server.py -v
```

### `test_server_integration.py`
Integration tests that require actual inference server binaries (vLLM, SGLang) to be installed. These tests actually start and stop servers.

**Skip integration tests:**
```bash
SKIP_INTEGRATION_TESTS=1 python -m pytest tests/test_server_integration.py -v
```

**Run with a specific model:**
```bash
TEST_MODEL=/path/to/model python -m pytest tests/test_server_integration.py -v
```

## Requirements

- Python 3.7+
- pytest (for running tests)
- PyYAML (for YAML configuration tests)

For integration tests:
- vLLM or SGLang installed
- Access to a model (set via `TEST_MODEL` environment variable)

## Running All Tests

```bash
# Unit tests only
python -m pytest tests/test_inference_server.py -v

# Integration tests (requires server binaries)
python -m pytest tests/test_server_integration.py -v

# All tests
python -m pytest tests/ -v
```

## Test Coverage

The tests cover:
- Server initialization and configuration
- Backend name and command generation
- Endpoint listing
- Environment variable handling
- YAML configuration loading
- Profile configuration (nsys, pytorch, amd)
- Context manager functionality
- Error handling
- Server start/stop (integration tests)

