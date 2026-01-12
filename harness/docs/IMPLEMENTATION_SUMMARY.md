# Implementation Summary: GPT-OSS-120B and Qwen3VL Support

## Overview

This document summarizes all the changes made to add support for GPT-OSS-120B and Qwen3VL models to the MLPerf Inference harness.

## Files Created

### 1. `harness_gpt_oss_120b.py`
- **Purpose**: Harness implementation for GPT-OSS-120B model
- **Key Features**:
  - SGLang backend integration
  - Generation config loading
  - Mode-specific sampling parameters
  - Direct `input_ids` support (no text decoding)

### 2. `harness_qwen3vl.py`
- **Purpose**: Harness implementation for Qwen3VL multimodal model
- **Key Features**:
  - Multimodal messages format support
  - Chat completions API integration
  - Image processing and base64 encoding
  - Guided decoding support

### 3. Documentation Files
- `docs/gpt-oss-120b-and-qwen3vl-support.md`: Comprehensive guide
- `docs/QUICK_REFERENCE-gpt-oss-qwen3vl.md`: Quick command reference
- `docs/CHANGELOG-gpt-oss-qwen3vl.md`: Detailed changelog
- `docs/IMPLEMENTATION_SUMMARY.md`: This file

## Files Modified

### 1. `Client/loadgen_client.py`
**Changes**:
- Added `use_input_ids` flag for SGLang compatibility
- Added `use_messages` and `multimodal` flags for multimodal support
- Added `offline_back_to_back` option
- Added mode-specific sampling parameter selection (`_get_sampling_params`)
- Added `_process_api_single()` for individual request processing
- Added `_process_multimodal_request()` for multimodal API calls
- Added `_construct_messages_from_dataset()` for message construction
- Added `_process_sglang_response()` for SGLang response handling
- Modified `_process_api_batch()` to support different request formats
- Updated `issue_query()` to support back-to-back mode

**Key Methods Added**:
```python
def _get_sampling_params(self) -> Tuple[float, int, float]
def _process_api_single(self, q_sample, temperature, top_k, top_p)
def _process_multimodal_request(self, query_id, query_index, messages, ...)
def _construct_messages_from_dataset(self, index) -> List[Dict]
def _process_sglang_response(self, query_id, query_index, output_ids, ...)
```

### 2. `data/dataset_processor.py`
**Changes**:
- Added `messages` attribute initialization
- Added extraction of `messages` column from datasets
- Prioritized `messages` column over `input`/`input_ids` for multimodal

**Key Changes**:
- Extracts `messages` column if present
- If `messages` exists, skips `input`/`input_ids` extraction

### 3. `harness/arg_parser.py`
**Changes**:
- Added `--offline-back-to-back` argument
- Updated `parse_common_harness_args()` to pass `offline_back_to_back` to server config

### 4. `harness_main.py`
**Changes**:
- Added model category mappings:
  - `gpt-oss-120b` → `harness_gpt_oss_120b`
  - `qwen3vl` → `harness_qwen3vl`
- Added auto-detection patterns for model names

### 5. `README.md`
**Changes**:
- Added reference to new documentation files
- Added "New Model Support" section

## Architecture Changes

### Request Flow

#### GPT-OSS-120B (SGLang)
```
Dataset (tokenized) → input_ids → SGLang API (/generate) → output_ids → LoadGen
```

#### Qwen3VL (vLLM)
```
Dataset (multimodal) → messages → Chat Completions API → text → bytes → LoadGen
```

### Sampling Parameters

**Before**: Single set of parameters for all modes
**After**: Separate parameters for accuracy and performance modes

```python
# Performance mode
temperature = config.get('temperature', 0.0)
top_k = config.get('top_k', 1)
top_p = config.get('top_p', 1.0)

# Accuracy mode (if specified)
if test_mode == "accuracy":
    temperature = config.get('accuracy_temperature', temperature)
    top_k = config.get('accuracy_top_k', top_k)
    top_p = config.get('accuracy_top_p', top_p)
```

### Offline Request Handling

**Before**: Always batched requests
**After**: Option to send back-to-back

```python
if offline_back_to_back:
    # Send requests individually
    for q_sample in query_samples:
        _process_api_single(q_sample, ...)
else:
    # Batch requests (original behavior)
    _process_api_batch(batch, ...)
```

## Configuration

### GPT-OSS-120B Configuration

**Server Config** (`configs/backends/sglang.yaml`):
```yaml
backend: sglang
model: openai/gpt-oss-120b
port: 30000
endpoint_type: completions
```

**Client Config** (automatically set):
```python
{
    'use_input_ids': True,
    'sglang_endpoint': '/generate',
    'temperature': 1.0,  # performance mode
    'top_k': -1,
    'top_p': 1.0,
    'accuracy_temperature': 0.001,  # accuracy mode
    'accuracy_top_k': 1,
    'accuracy_top_p': 1.0
}
```

### Qwen3VL Configuration

**Server Config** (`configs/backends/vllm.yaml`):
```yaml
backend: vllm
model: Qwen/Qwen3-VL-235B-A22B-Instruct
port: 8000
endpoint_type: chat_completions
```

**Client Config** (automatically set):
```python
{
    'multimodal': True,
    'use_messages': True,
    'use_guided_decoding': False,
    'temperature': 0.0,  # performance mode
    'top_p': 1.0,
    'accuracy_temperature': 0.0,  # accuracy mode
    'accuracy_top_p': 1.0
}
```

## API Formats

### SGLang API Format (GPT-OSS-120B)

**Request**:
```json
{
    "input_ids": [1, 2, 3, ...],
    "sampling_params": {
        "max_new_tokens": 32768,
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0
    }
}
```

**Response**:
```json
{
    "output_ids": [4, 5, 6, ...],
    "text": "generated text...",
    "meta_info": {
        "completion_tokens": 150
    }
}
```

### Chat Completions API Format (Qwen3VL)

**Request**:
```json
{
    "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "messages": [
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
    ],
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0
}
```

**Response**:
```json
{
    "choices": [{
        "message": {
            "content": "generated text..."
        }
    }],
    "usage": {
        "completion_tokens": 150
    }
}
```

## Testing Checklist

### GPT-OSS-120B
- [ ] Performance mode with default parameters
- [ ] Accuracy mode with deterministic parameters
- [ ] Generation config file loading
- [ ] Command-line parameter overrides
- [ ] Back-to-back request mode
- [ ] Server scenario
- [ ] SGLang server connectivity

### Qwen3VL
- [ ] Performance mode with messages
- [ ] Accuracy mode with guided decoding
- [ ] Message construction from dataset fields
- [ ] Image processing (base64 encoding)
- [ ] Server scenario
- [ ] vLLM server connectivity

## Known Issues and Limitations

1. **Qwen3VL Message Construction**: 
   - Simplified version compared to reference implementation
   - Reference uses HuggingFace datasets, this uses local files
   - For production, pre-format messages in dataset

2. **SGLang Batching**:
   - Back-to-back mode recommended (SGLang handles batching)
   - Batch mode may not be optimal for SGLang

3. **Multimodal Dataset**:
   - Currently supports local files only
   - HuggingFace dataset integration not yet implemented

## Future Enhancements

- [ ] Direct HuggingFace dataset support for Qwen3VL
- [ ] Enhanced message construction matching reference exactly
- [ ] Batch processing for multimodal requests (if API supports)
- [ ] Additional multimodal model support
- [ ] Performance optimizations for message construction

## References

- GPT-OSS-120B Reference: `/mnt/data/nmiriyal/mlperf-inference-6.0-redhat/language/gpt-oss-120b/`
- Qwen3VL Reference: `/mnt/data/nmiriyal/mlperf-inference-6.0-redhat/multimodal/qwen3-vl/`
- SGLang Backend: `backends/sglang_backend.py` in reference implementation
- Task Implementation: `src/mlperf_inf_mm_q3vl/task.py` in reference implementation
