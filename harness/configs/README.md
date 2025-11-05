# Configuration Directory Structure

This directory contains YAML configuration files organized by type.

## Directory Structure

```
configs/
├── datasets/          # Dataset-specific configurations
│   ├── llama3.1-8b.yaml
│   └── deepseek-r1.yaml
├── models/           # Model-specific configurations
│   ├── llama3.1-8b.yaml
│   └── deepseek-r1.yaml
└── README.md         # This file
```

## Dataset Configurations (`datasets/`)

Dataset configurations define field mappings and dataset-specific settings.

### Example: `datasets/llama3.1-8b.yaml`

```yaml
name: llama3.1-8b
description: "Configuration for Llama 3.1 8B dataset"

fields:
  input_column: "input"
  input_ids_column: "tok_input"
  output_column: "output"
  input_lens_column: null  # Optional

file_format: "auto"
total_sample_count: 13368

model_specific:
  default_model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

## Model Configurations (`models/`)

Model configurations define model-specific settings and default datasets.

### Example: `models/deepseek-r1.yaml`

```yaml
name: deepseek-r1
description: "Model configuration for DeepSeek R1"

model_specific:
  default_model_name: "deepseek-ai/DeepSeek-R1-0528"
  default_dataset: "deepseek-r1"
```

## Configuration Loading Priority

The system loads configurations in this order:

1. **Model-specific dataset config**: `models/{model_name}/{dataset_name}.yaml`
2. **Model general config**: `models/{model_name}.yaml`
3. **Dataset config**: `datasets/{dataset_name}.yaml`
4. **Defaults**: Fallback to default values

## Adding New Configurations

### Adding a New Dataset

1. Create `datasets/my-dataset.yaml`:
   ```yaml
   name: my-dataset
   description: "Configuration for my custom dataset"
   
   fields:
     input_column: "prompt"
     input_ids_column: "token_ids"
     output_column: "target"
     input_lens_column: null  # Optional, calculated if null
   
   file_format: "pickle"  # or "json", "csv", "auto"
   
   total_sample_count: 10000  # Optional
   
   model_specific:
     default_model_name: "my-model/MyModel"
   ```

2. Use in harness via command line:
   ```bash
   python harness/harness_main.py \
       --model my-model/MyModel \
       --dataset-path ./my_dataset.pkl \
       --dataset-name my-dataset
   ```

3. Or use in Python:
   ```python
   from harness.base_harness import BaseHarness
   
   harness = BaseHarness(
       model_name="my-model/MyModel",
       dataset_path="./my_dataset.pkl",
       dataset_name="my-dataset"  # Loads configs/datasets/my-dataset.yaml
   )
   ```

4. Or specify config file directly:
   ```bash
   python harness/harness_main.py \
       --model my-model/MyModel \
       --dataset-path ./my_dataset.pkl \
       --dataset-config-file configs/datasets/my-dataset.yaml
   ```

### Adding a New Model

1. Create `models/my-model.yaml`:
   ```yaml
   name: my-model
   description: "Model configuration for MyModel"
   
   model_specific:
     default_model_name: "my-model/MyModel"
     default_dataset: "my-dataset"
     # Add other model-specific settings here
   ```

2. Use in harness:
   ```bash
   python harness/harness_main.py \
       --model my-model/MyModel \
       --dataset-path ./my_dataset.pkl \
       --dataset-name my-dataset
   ```

   Or in Python:
   ```python
   from harness.base_harness import BaseHarness
   
   harness = BaseHarness(
       model_name="my-model/MyModel",
       dataset_path="./my_dataset.pkl",
       dataset_name="my-dataset"
   )
   ```

## Usage Examples with Configuration

### Example 1: Using Default Dataset Config

```bash
# Auto-loads configs/datasets/llama3.1-8b.yaml
# Includes backend server configuration
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --test-mode performance \
    --batch-size 13368 \
    --num-samples 13368 \
    --output-dir ./harness_output
```

### Example 2: Overriding Config File

```bash
# Use a specific config file instead of auto-detection
# With backend server configuration
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./my_dataset.pkl \
    --dataset-config-file configs/datasets/my-custom-config.yaml \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --batch-size 1000
```

### Example 3: Overriding Column Names

```bash
# Override column mappings from command line
# With backend server and output directory
python harness/harness_main.py \
    --model my-model/MyModel \
    --dataset-path ./dataset.pkl \
    --input-column prompt \
    --input-ids-column tokens \
    --output-column answer \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --test-mode performance \
    --batch-size 1000 \
    --num-samples 1000 \
    --output-dir ./custom_output
```

### Example 4: Combining Config and Overrides

```bash
# Use config file but override specific columns
# With backend server configuration and metrics
python harness/harness_main.py \
    --model my-model/MyModel \
    --dataset-path ./dataset.pkl \
    --dataset-config-file configs/datasets/my-dataset.yaml \
    --input-column custom_prompt \
    --server-config backendserver/example_server_config.yaml \
    --api-server-url http://localhost:8000 \
    --endpoint-type completions \
    --scenario Offline \
    --enable-metrics \
    --output-dir ./results
```

### Example 5: Different Models with Same Dataset

```bash
# Use same dataset config with different models
# Both using same backend server configuration
python harness/harness_main.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --batch-size 13368 \
    --num-samples 13368

python harness/harness_main.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --dataset-path ./cnn_eval.json \
    --dataset-name llama3.1-8b \
    --server-config backendserver/example_server_config.yaml \
    --scenario Offline \
    --batch-size 13368 \
    --num-samples 13368
```

### Example 6: Model-Specific Dataset Config

Create `models/my-model/my-dataset.yaml`:

```yaml
name: my-dataset
description: "My dataset config specific to my-model"

fields:
  input_column: "model_specific_input"
  input_ids_column: "model_specific_tokens"
  output_column: "model_specific_output"
```

This will be loaded when using `my-model` with `my-dataset`:

```bash
python harness/harness_main.py \
    --model my-model/MyModel \
    --dataset-path ./dataset.pkl \
    --dataset-name my-dataset
    # Loads models/my-model/my-dataset.yaml automatically
```

## Benefits

- **Organized**: Clear separation between datasets and models
- **Scalable**: Easy to add new datasets/models
- **Maintainable**: All configuration in one place
- **Flexible**: Supports model-specific overrides

