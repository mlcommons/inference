# Dataset Configuration Design

## Overview

This design provides a generic, configuration-driven approach to handling different datasets and models in MLPerf Inference harnesses. The system uses YAML configuration files to define dataset field mappings and model-specific settings, allowing the same harness code to work with different datasets and scenarios.

## Key Components

### 1. Dataset Configuration System (`harness/data/dataset_config.py`)

- **DatasetConfigLoader**: Loads YAML configuration files for datasets
- **DatasetConfig**: Data class containing dataset field mappings
- **ModelDatasetConfig**: Model-specific dataset configuration

### 2. Enhanced Dataset Processor (`harness/data/dataset_processor.py`)

- Automatically loads dataset configuration from YAML files
- Uses field mappings from config (input_column, input_ids_column, output_column)
- Falls back to defaults if config not available

### 3. Base Harness (`harness/harness/base_harness.py`)

- Works for both Offline and Server scenarios
- Automatically loads dataset configuration
- Provides hooks for model-specific customizations:
  - `_pre_run_setup()`: Pre-run initialization
  - `_post_run_processing()`: Post-run processing
  - `_cleanup_custom()`: Custom cleanup

### 4. Model-Specific Harnesses

- **DeepSeek R1** (`language/deepseek-r1/harness_deepseek_r1.py`): Extends BaseHarness
- **Llama 3.1 8B** (`harness/harness_llama3.1_8b.py`): Extends BaseHarness

## Configuration Files

Configuration files are stored in `harness/data/configs/`:

- `llama3.1-8b.yaml`: Llama 3.1 8B dataset configuration
- `deepseek-r1.yaml`: DeepSeek R1 dataset configuration

### Configuration Structure

```yaml
name: dataset-name
description: "Description"

fields:
  input_column: "text_input"
  input_ids_column: "tok_input"
  output_column: "ref_output"
  input_lens_column: null  # Optional

file_format: "auto"
total_sample_count: 4388

model_specific:
  default_model_name: "model-name"
```

## Usage Examples

### Using BaseHarness Directly

```python
from harness.base_harness import BaseHarness

harness = BaseHarness(
    model_name="deepseek-ai/DeepSeek-R1-0528",
    dataset_path="./dataset.pkl",
    dataset_name="deepseek-r1",
    scenario="Offline",  # or "Server"
    test_mode="performance"
)

results = harness.run()
```

### Creating Model-Specific Harness

```python
from harness.base_harness import BaseHarness

class MyModelHarness(BaseHarness):
    def __init__(self, **kwargs):
        if 'dataset_name' not in kwargs:
            kwargs['dataset_name'] = 'my-dataset'
        super().__init__(**kwargs)
    
    def _pre_run_setup(self):
        # Model-specific setup
        pass
```

## Benefits

1. **Code Reusability**: Same harness code works for different datasets
2. **Easy Configuration**: Add new datasets by creating YAML files
3. **Scenario Agnostic**: Works for both Offline and Server scenarios
4. **Extensible**: Model-specific customizations via subclass hooks
5. **Maintainable**: All dataset info centralized in config files

## Adding New Datasets

1. Create YAML config file in `harness/data/configs/`
2. Define field mappings
3. Use in harness with `dataset_name` parameter

No code changes needed!

## Design Principles

- **Configuration over Code**: Dataset-specific info in YAML, not code
- **Inheritance Hierarchy**: BaseHarness → ModelHarness (if needed)
- **Backward Compatible**: Falls back to defaults if config not available
- **Extensible**: Hooks for model-specific behavior

## File Structure

```
harness/
├── data/
│   ├── dataset_config.py          # Configuration loader
│   ├── dataset_processor.py        # Enhanced processor
│   └── configs/
│       ├── llama3.1-8b.yaml       # Llama config
│       └── deepseek-r1.yaml        # DeepSeek config
├── harness/
│   └── base_harness.py            # Base harness (with dataset config support)
└── harness_llama3.1_8b.py         # Extends BaseHarness

language/
└── deepseek-r1/
    └── harness_deepseek_r1.py     # New DeepSeek harness
```

