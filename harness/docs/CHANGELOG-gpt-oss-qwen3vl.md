# Changelog: GPT-OSS-120B and Qwen3VL Support

## Date: 2024

### Added Features

#### 1. GPT-OSS-120B Model Support
- **New File**: `harness_gpt_oss_120b.py`
- **Features**:
  - SGLang backend integration with direct `input_ids` support
  - Generation config loading from JSON files
  - Mode-specific sampling parameters (accuracy vs performance)
  - Support for pre-tokenized datasets
  - Integration with existing harness infrastructure

#### 2. Qwen3VL Multimodal Model Support
- **New File**: `harness_qwen3vl.py`
- **Features**:
  - Multimodal messages format (text + images)
  - Chat completions API integration
  - Automatic message construction from dataset fields
  - Guided decoding support for structured outputs
  - Base64 image encoding support

#### 3. Enhanced LoadGen Client
- **File**: `Client/loadgen_client.py`
- **New Features**:
  - `use_input_ids` flag for SGLang compatibility
  - `use_messages` flag for multimodal support
  - `offline_back_to_back` option for individual request sending
  - Mode-specific sampling parameter selection
  - Multimodal request processing (`_process_multimodal_request`)
  - Message construction from dataset fields (`_construct_messages_from_dataset`)

#### 4. Enhanced Dataset Processor
- **File**: `data/dataset_processor.py`
- **New Features**:
  - `messages` attribute for multimodal datasets
  - Automatic extraction of `messages` column
  - Support for pre-formatted multimodal samples

#### 5. Argument Parser Updates
- **File**: `harness/arg_parser.py`
- **New Arguments**:
  - `--offline-back-to-back`: Send requests individually in offline scenario
  - Integration with server config for client options

#### 6. Model Category Support
- **File**: `harness_main.py`
- **New Categories**:
  - `gpt-oss-120b` (with variants: `gpt_oss_120b`, `gptoss120b`)
  - `qwen3vl` (with variants: `qwen3-vl`, `qwen3_vl`)
- **Auto-detection**: Model name patterns for automatic category detection

### Technical Changes

#### Client Architecture
- Added support for multiple request formats:
  - Standard: Text prompts → Token IDs
  - SGLang: Input IDs → Direct API
  - Multimodal: Messages → Chat Completions API

#### Sampling Parameters
- Performance mode: Uses configurable temperature, top_k, top_p
- Accuracy mode: Uses separate parameters (typically deterministic)
- Parameters stored in client config and selected at runtime

#### Request Handling
- **Batching Mode (Default)**: Groups requests into batches
- **Back-to-Back Mode**: Sends requests individually
- Both modes respect sampling parameters and test mode

### Breaking Changes

None. All changes are backward compatible with existing harnesses.

### Dependencies

#### New Optional Dependencies
- **PIL/Pillow**: For image processing in Qwen3VL (if using raw image fields)
- **pandas**: Already required, now used for multimodal message construction

#### Existing Dependencies
- All existing dependencies remain the same
- No new required dependencies

### Configuration

#### GPT-OSS-120B
- Requires SGLang backend configuration
- Supports generation config JSON files
- Endpoint type: `completions`

#### Qwen3VL
- Requires vLLM backend configuration
- Endpoint type: `chat_completions`
- Supports guided decoding configuration

### Migration Notes

#### For GPT-OSS-120B Users
1. Update dataset to use tokenized format (if not already)
2. Ensure SGLang server is running on expected port
3. Use `--generation-config` or command-line parameters for sampling

#### For Qwen3VL Users
1. Ensure dataset has `messages` column or fields for auto-conversion
2. Use vLLM backend (not SGLang)
3. Set `endpoint_type` to `chat_completions`

### Known Limitations

1. **Qwen3VL Message Construction**: 
   - Auto-conversion from fields is a simplified version
   - For production use, pre-format messages in dataset
   - Reference implementation uses HuggingFace datasets (not local files)

2. **SGLang Batching**:
   - Back-to-back mode recommended for SGLang (handles batching internally)
   - Batch mode may not provide optimal performance

3. **Multimodal Dataset**:
   - Currently supports local files (Parquet/Pickle)
   - HuggingFace dataset loading not yet integrated

### Future Enhancements

- [ ] Direct HuggingFace dataset support for Qwen3VL
- [ ] Enhanced message construction matching reference implementation exactly
- [ ] Batch processing for multimodal requests (if API supports)
- [ ] Additional multimodal model support
