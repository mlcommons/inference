# Harness Refactoring Summary

This document summarizes the refactoring completed to extract common functionality and create reusable components.

## Changes Made

### 1. Base Harness Class (`harness/base_harness.py`)

Created a base harness class with all common functionality:

- **Server Management**: Start/stop inference servers
- **LoadGen Setup**: Common TestSettings configuration
- **Metrics Collection**: Initialize and manage metrics collectors
- **Visualization**: Generate metrics visualizations
- **MLflow Integration**: Optional MLflow tracking
- **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM
- **Stdout Redirection**: Capture all output to files
- **Environment Info**: Collect system environment information
- **Metadata Storage**: Save YAML metadata for later MLflow upload

### 2. Standalone MLflow Upload Script (`upload_to_mlflow.py`)

Created a standalone script that can upload results to MLflow:

- Reads metadata YAML file created during harness run
- Supports uploading results even if MLflow wasn't configured during run
- Can override MLflow server and experiment name via command-line
- Handles all MLflow operations (parameters, metrics, artifacts, descriptions)

**Usage:**
```bash
# Upload using metadata file (uses MLflow config from metadata)
python upload_to_mlflow.py --metadata-file ./harness_output/mlflow_metadata.yaml

# Upload with custom MLflow server
python upload_to_mlflow.py --metadata-file ./harness_output/mlflow_metadata.yaml \
    --mlflow-host localhost --mlflow-port 5000 --mlflow-experiment-name my-experiment
```

### 3. YAML Metadata Storage

When running without MLflow configured, the harness automatically saves metadata to `mlflow_metadata.yaml` containing:

- Test results (status, duration, scenario, etc.)
- Harness configuration (model, dataset, scenario, Server parameters, etc.)
- Output paths (output_dir, mlperf_output_dir, metrics, visualizations)
- MLflow configuration (tracking_uri, experiment_name, output_dir)
- Timestamp

This allows uploading results to MLflow later using the upload script.

### 4. Refactored Llama31_8B Harness (`harness_llama3.1_8b.py`)

The Llama31_8B harness is now minimal and focused on:

- Inheriting from `BaseHarness` for all common functionality
- Implementing only Llama31_8B-specific `run()` method
- Setting up LoadGen QSL and SUT
- Executing LoadGen test
- Handling test results

The file is now ~350 lines (down from ~1000 lines).

## Architecture

```
harness/
├── harness/                    # New base harness module
│   ├── __init__.py
│   └── base_harness.py         # Base harness with common functionality
├── harness_llama3.1_8b.py      # Refactored Llama31_8B harness (minimal)
├── upload_to_mlflow.py         # Standalone MLflow upload script
└── ...                         # Other existing modules
```

## Benefits

1. **Code Reusability**: All common functionality is in the base class
2. **Easy to Extend**: New harnesses can inherit from `BaseHarness` and only implement model-specific code
3. **Flexible MLflow Usage**: Can run without MLflow and upload later
4. **Cleaner Code**: Harness-specific files are much smaller and focused
5. **Maintainability**: Common code changes only need to be made in one place

## Usage Examples

### Running with MLflow
```bash
python harness_llama3.1_8b.py \
    --model <model> \
    --dataset-path <dataset> \
    --scenario Server \
    --mlflow-experiment-name my-experiment \
    --mlflow-host localhost \
    --mlflow-port 5000
```

### Running without MLflow (upload later)
```bash
python harness_llama3.1_8b.py \
    --model <model> \
    --dataset-path <dataset> \
    --scenario Server \
    --output-dir ./my_results
```

After the run completes, upload to MLflow:
```bash
python upload_to_mlflow.py \
    --metadata-file ./my_results/mlflow_metadata.yaml \
    --mlflow-host localhost \
    --mlflow-port 5000 \
    --mlflow-experiment-name my-experiment
```

## Next Steps for Other Harnesses

To create a new harness (e.g., for another model):

1. Inherit from `BaseHarness`
2. Implement only the `run()` method with model-specific LoadGen setup
3. Use `self.setup_loadgen_settings(user_conf, lg_model_name)` for TestSettings
4. Follow the same pattern as `Llama31_8BHarness`

Example:
```python
from harness.base_harness import BaseHarness

class MyModelHarness(BaseHarness):
    def run(self, user_conf: str = "user.conf", lg_model_name: str = "my-model"):
        # Start components (handled by base class)
        # ... setup LoadGen with model-specific settings ...
        # ... run test ...
        # Return test_results
```

## Files Modified

- `harness_llama3.1_8b.py`: Refactored to use `BaseHarness`
- Created `harness/base_harness.py`: New base class
- Created `harness/__init__.py`: Module initialization
- Created `upload_to_mlflow.py`: Standalone upload script

## Testing

The refactored code maintains backward compatibility:
- All existing command-line arguments work the same
- Same output directory structure
- Same MLflow integration behavior
- New metadata file is automatically created when MLflow is not configured

