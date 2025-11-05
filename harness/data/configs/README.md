# Dataset Configuration System

This directory contains YAML configuration files for different datasets used in MLPerf Inference harnesses.

## Overview

The dataset configuration system allows you to:
- Define field mappings for different datasets (input, input_ids, output columns)
- Specify model-specific settings
- Support both Offline and Server scenarios with the same harness code
- Easily add new datasets without modifying harness code

## Configuration File Structure

Each dataset configuration file follows this structure:

```yaml
name: dataset-name
description: "Description of the dataset"

fields:
  input_column: "text_input"      # Column name for input text
  input_ids_column: "tok_input"    # Column name for tokenized input IDs
  output_column: "ref_output"      # Column name for output/targets
  input_lens_column: null           # Optional: column name for input lengths (if null, calculated from input_ids)

file_format: "auto"                # auto, json, pickle, csv

total_sample_count: 4388           # Optional: default number of samples

model_specific:
  # Model-specific settings can be added here
  default_model_name: "deepseek-ai/DeepSeek-R1-0528"
```

## Example Configurations

### Llama 3.1 8B (`llama3.1-8b.yaml`)

```yaml
name: llama3.1-8b
fields:
  input_column: "input"
  input_ids_column: "tok_input"
  output_column: "output"
```

### DeepSeek R1 (`deepseek-r1.yaml`)

```yaml
name: deepseek-r1
fields:
  input_column: "text_input"
  input_ids_column: "tok_input"
  output_column: "ref_output"
```

## Usage

The harness automatically loads the appropriate configuration based on the dataset name:

```python
from harness.base_harness import BaseHarness

harness = BaseHarness(
    model_name="deepseek-ai/DeepSeek-R1-0528",
    dataset_path="./dataset.pkl",
    dataset_name="deepseek-r1",  # Loads deepseek-r1.yaml automatically
    scenario="Offline"  # or "Server"
)
```

## Adding New Datasets

To add a new dataset:

1. Create a new YAML file in this directory: `my-dataset.yaml`
2. Define the field mappings:
   ```yaml
   name: my-dataset
   fields:
     input_column: "prompt"
     input_ids_column: "token_ids"
     output_column: "target"
   ```
3. Use it in your harness:
   ```python
   harness = BaseHarness(
       dataset_name="my-dataset",  # Loads my-dataset.yaml
       ...
   )
   ```

## Model-Specific Configurations

You can create model-specific configurations by naming files as:
`{model_name}_{dataset_name}.yaml`

For example: `deepseek-ai_DeepSeek-R1-0528_deepseek-r1.yaml`

The system will:
1. First try to load model-specific config
2. Fall back to dataset-specific config
3. Finally use defaults if no config found

## Benefits

- **No code changes needed** when adding new datasets
- **Consistent interface** across different models
- **Easy to maintain** - all dataset info in one place
- **Supports both scenarios** - Offline and Server use same harness

