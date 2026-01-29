# GPT-OSS-120B and Qwen3VL Support Documentation

## Overview

This document describes the additions to the MLPerf Inference harness to support:
- **GPT-OSS-120B**: Large language model with SGLang backend support
- **Qwen3VL**: Multimodal vision-language model with vLLM backend support

Additionally, the harness now supports:
- Different sampling parameters for accuracy vs performance modes
- Offline scenario with back-to-back request sending (instead of batching)
- Load balancing across multiple API servers (see [LOAD_BALANCING.md](LOAD_BALANCING.md))

## Table of Contents

1. [GPT-OSS-120B Support](#gpt-oss-120b-support)
2. [Qwen3VL Support](#qwen3vl-support)
3. [Sampling Parameters](#sampling-parameters)
4. [Offline Back-to-Back Requests](#offline-back-to-back-requests)
5. [Command Line Examples](#command-line-examples)

## Quick Start

### GPT-OSS-120B
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

### Qwen3VL
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output
```

For more examples, see [QUICK_REFERENCE-gpt-oss-qwen3vl.md](QUICK_REFERENCE-gpt-oss-qwen3vl.md)

---

## GPT-OSS-120B Support

### Features

- **SGLang Backend Integration**: Uses SGLang backend with direct `input_ids` support
- **Generation Config**: Supports loading sampling parameters from `generation_config.json`
- **Tokenized Input**: Works with pre-tokenized datasets (no text decoding needed)
- **Mode-Specific Parameters**: Different sampling parameters for accuracy vs performance

### Dataset Requirements

- Dataset should contain pre-tokenized input IDs in a column (typically `tok_input` or `input_tokens`)
- Format: Parquet or Pickle file with tokenized prompts as lists of integers
- Example columns: `tok_input`, `input_tokens`, `ground_truth`, `dataset`

### Generation Config

The harness supports loading generation parameters from a JSON config file. Default location:
```
/mnt/data/nmiriyal/mlperf-inference-6.0-redhat/language/gpt-oss-120b/generation_config.json
```

Example `generation_config.json`:
```json
{
  "max_new_tokens": 32768,
  "temperature": 1.0,
  "top_k": -1,
  "top_p": 1.0
}
```

### Command Line Examples

#### Basic Performance Run
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --batch-size 13368 \
    --num-samples 13368 \
    --api-server-url http://localhost:30000 \
    --server-config configs/backends/sglang.yaml \
    --output-dir ./harness_output
```

#### Accuracy Run with Custom Parameters
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --generation-config /path/to/generation_config.json \
    --max-tokens 32768 \
    --temperature 1.0 \
    --top-k -1 \
    --top-p 1.0 \
    --accuracy-temperature 0.001 \
    --accuracy-top-k 1 \
    --accuracy-top-p 1.0 \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

#### Using Model Category (Auto-detection)
```bash
python harness/harness_main.py \
    --model-category gpt-oss-120b \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

#### Server Scenario
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:30000 \
    --server-target-qps 10.0 \
    --output-dir ./harness_output
```

#### With Back-to-Back Requests (Offline)
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --offline-back-to-back \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

---

## Qwen3VL Support

### Features

- **Multimodal Support**: Handles text + image inputs via messages format
- **Chat Completions API**: Uses OpenAI-compatible chat completions endpoint
- **Guided Decoding**: Optional structured output support via JSON schema
- **Image Processing**: Supports base64-encoded images in messages

### Dataset Requirements

The dataset should contain one of the following:

**Option 1: Pre-formatted Messages Column**
- Column name: `messages`
- Format: List of ChatCompletionMessageParam objects
- Example:
  ```python
  [
      {
          "role": "system",
          "content": "You are a helpful assistant..."
      },
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "Product title: ..."},
              {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
          ]
      }
  ]
  ```

**Option 2: Raw Fields (Auto-conversion)**
- Fields: `product_image` (PIL Image), `product_title`, `product_description`, `system_message`
- The harness will automatically convert these to messages format
- Images are base64-encoded automatically

### Message Format

Messages follow the OpenAI Chat Completions format:
- **System Message**: `{"role": "system", "content": "..."}`
- **User Message**: `{"role": "user", "content": [...]}` where content can be:
  - Text: `{"type": "text", "text": "..."}`
  - Image: `{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`

### Command Line Examples

#### Basic Performance Run
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --batch-size 100 \
    --num-samples 100 \
    --api-server-url http://localhost:8000 \
    --server-config configs/backends/vllm.yaml \
    --output-dir ./harness_output
```

#### Accuracy Run with Guided Decoding
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --use-guided-decoding \
    --accuracy-temperature 0.0 \
    --accuracy-top-p 1.0 \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output
```

#### Using Model Category
```bash
python harness/harness_main.py \
    --model-category qwen3vl \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output
```

#### Server Scenario
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --server-target-qps 5.0 \
    --output-dir ./harness_output
```

---

## Sampling Parameters

### Overview

The harness now supports different sampling parameters for accuracy and performance modes. This allows you to:
- Use deterministic parameters (low temperature, top_k=1) for accuracy evaluation
- Use more creative parameters (higher temperature) for performance testing

### Parameter Configuration

#### GPT-OSS-120B

Parameters can be set via:
1. **Generation Config File**: `--generation-config /path/to/generation_config.json`
2. **Command Line Arguments**:
   - Performance mode: `--temperature`, `--top-k`, `--top-p`
   - Accuracy mode: `--accuracy-temperature`, `--accuracy-top-k`, `--accuracy-top-p`

#### Qwen3VL

Parameters can be set via:
- **Command Line Arguments**:
  - Performance mode: Uses defaults (temperature=0.0, top_p=1.0)
  - Accuracy mode: `--accuracy-temperature`, `--accuracy-top-k`, `--accuracy-top-p`

### Examples

#### GPT-OSS-120B with Different Parameters
```bash
# Performance: creative sampling
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --temperature 1.0 \
    --top-k -1 \
    --top-p 1.0 \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output

# Accuracy: deterministic sampling
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --accuracy-temperature 0.001 \
    --accuracy-top-k 1 \
    --accuracy-top-p 1.0 \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

#### Qwen3VL with Different Parameters
```bash
# Performance mode
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output

# Accuracy mode with custom parameters
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --accuracy-temperature 0.0 \
    --accuracy-top-p 1.0 \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output
```

---

## Offline Back-to-Back Requests

### Overview

By default, the offline scenario sends requests in batches. The `--offline-back-to-back` option sends requests individually, one after another. This is useful for:
- Backends that handle batching internally (like SGLang with continuous batching)
- Testing individual request latencies
- Avoiding client-side batching overhead

### Usage

Add the `--offline-back-to-back` flag to any offline scenario command:

```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --offline-back-to-back \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

### When to Use

- **Use back-to-back**: When the backend handles batching internally (SGLang, vLLM with continuous batching)
- **Use batching (default)**: When you want to test batch processing performance or reduce network overhead

---

## Complete Examples

### GPT-OSS-120B: Full Accuracy Run
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/accuracy_eval_tokenized.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --generation-config /path/to/generation_config.json \
    --max-tokens 32768 \
    --accuracy-temperature 0.001 \
    --accuracy-top-k 1 \
    --accuracy-top-p 1.0 \
    --api-server-url http://localhost:30000 \
    --server-config configs/backends/sglang.yaml \
    --user-conf user.conf \
    --lg-model-name gpt-oss-120b \
    --output-dir ./harness_output_accuracy \
    --enable-metrics \
    --debug-mode
```

### Qwen3VL: Full Performance Run with Metrics
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/product_catalogue.pkl \
    --scenario Offline \
    --test-mode performance \
    --batch-size 100 \
    --num-samples 1000 \
    --api-server-url http://localhost:8000 \
    --server-config configs/backends/vllm.yaml \
    --user-conf user.conf \
    --lg-model-name qwen3-vl-235b-a22b \
    --output-dir ./harness_output_performance \
    --enable-metrics \
    --metrics-interval 15
```

### GPT-OSS-120B: Server Scenario with Custom QPS
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/dataset.pkl \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:30000 \
    --server-target-qps 15.0 \
    --server-coalesce-queries true \
    --output-dir ./harness_output_server
```

### Qwen3VL: Server Scenario with Streaming
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/dataset.pkl \
    --scenario Server \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --server-target-qps 5.0 \
    --output-dir ./harness_output_server
```

---

## Configuration Files

### SGLang Backend Config (for GPT-OSS-120B)

Example: `configs/backends/sglang.yaml`
```yaml
backend: sglang
model: openai/gpt-oss-120b
port: 30000
endpoint_type: completions
config:
  server_args:
    - --tp
    - "8"
    - --context-length
    - "32768"
```

### vLLM Backend Config (for Qwen3VL)

Example: `configs/backends/vllm.yaml`
```yaml
backend: vllm
model: Qwen/Qwen3-VL-235B-A22B-Instruct
port: 8000
endpoint_type: chat_completions
config:
  api_server_args:
    - --tensor-parallel-size
    - "8"
    - --gpu-memory-utilization
    - "0.9"
```

---

## Technical Details

### GPT-OSS-120B Implementation

1. **SGLang Integration**:
   - Uses `/generate` endpoint with `input_ids` directly
   - Format: `{"input_ids": [...], "sampling_params": {...}}`
   - No tokenizer decoding needed (uses pre-tokenized data)

2. **Data Flow**:
   - Dataset → Token IDs → SGLang API → Token IDs → LoadGen

3. **Sampling Parameters**:
   - Stored in client config
   - Selected based on `test_mode` (accuracy vs performance)

### Qwen3VL Implementation

1. **Multimodal Messages**:
   - Messages format with text and base64-encoded images
   - Sent directly to chat completions API
   - Response is text content (not token IDs)

2. **Data Flow**:
   - Dataset → Messages → Chat Completions API → Text → Bytes → LoadGen

3. **Image Processing**:
   - Images are base64-encoded if provided as PIL Images
   - Format: `data:image/{format};base64,{base64_string}`

### Back-to-Back Requests

- **Default (Batching)**: Groups requests into batches of `batch_size`
- **Back-to-Back**: Sends each request individually
- Both modes respect the same sampling parameters

---

## Troubleshooting

### GPT-OSS-120B Issues

**Issue**: "Column 'tok_input' not found"
- **Solution**: Ensure dataset has tokenized input column (`tok_input` or `input_tokens`)

**Issue**: "SGLang server not responding"
- **Solution**: Verify SGLang server is running on the specified port (default: 30000)

**Issue**: "Connection pool is full"
- **Solution**: Increase `max_pool_size` in SGLang backend config or reduce concurrency

### Qwen3VL Issues

**Issue**: "Dataset doesn't have 'messages' column"
- **Solution**: Either add pre-formatted `messages` column or ensure dataset has fields like `product_image`, `product_title`, etc. for auto-conversion

**Issue**: "Image processing failed"
- **Solution**: Ensure PIL/Pillow is installed and images are in supported format (PNG, JPEG, etc.)

**Issue**: "Chat completions endpoint not found"
- **Solution**: Verify backend is vLLM (not SGLang) and endpoint_type is `chat_completions`

---

## Migration Guide

### From Reference Implementation

If migrating from the reference implementations:

#### GPT-OSS-120B
- Replace `run_mlperf.py` with `harness_gpt_oss_120b.py`
- Use `--generation-config` instead of `--generation-config` flag
- Use `--api-server-url` instead of `--server-url`
- Sampling parameters now support mode-specific values

#### Qwen3VL
- Replace `benchmark.py` with `harness_qwen3vl.py`
- Dataset should have `messages` column or raw fields for auto-conversion
- Use standard harness arguments instead of custom CLI

---

## Additional Resources

- GPT-OSS-120B Reference: `/mnt/data/nmiriyal/mlperf-inference-6.0-redhat/language/gpt-oss-120b/`
- Qwen3VL Reference: `/mnt/data/nmiriyal/mlperf-inference-6.0-redhat/multimodal/qwen3-vl/`
- Main Harness README: `README.md`
- Architecture Documentation: `ARCHITECTURE.md`
