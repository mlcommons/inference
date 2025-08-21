# MLCommons Inference - General Preprocessing Guide

## Overview

This guide covers common preprocessing patterns across all language models in MLCommons Inference benchmarks. Preprocessing varies by:
1. Model architecture
2. Backend choice (PyTorch, vLLM, SGLang)
3. Task type (summarization, Q&A, etc.)

## Common Tokenizer Setup Pattern

Most models follow this pattern:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"  # Critical for generation
tokenizer.pad_token = tokenizer.eos_token
```

## Backend Dependencies

Different backends have different preprocessing requirements:

| Backend | Input Type | Chat Template Support | Use Case |
|---------|------------|---------------------|----------|
| PyTorch | Tokenized | Varies by model | Distributed inference |
| vLLM | Text | Varies by model | High-throughput serving |
| SGLang | Text | Usually disabled | Optimized serving |

## Dataset Format

All models expect datasets with these common fields:

```python
{
    'text_input': str,      # Raw prompt text (required)
    'tok_input': List[int], # Pre-tokenized input (optional)
    'output': str,          # Expected output for evaluation
}
```

## Model-Specific Preprocessing

### Models Using Chat Templates
- **DeepSeek-R1**: Uses `apply_chat_template` with PyTorch/vLLM
- **Potential others**: Check `uses_chat_template` in backend registry

### Models Using Simple Templates
- **Llama 3.1-8B**: Instruction format for summarization
- **Llama 2-70B**: Custom format with `[INST]` markers
- **Mixtral-8x7B**: Simple instruction format

### Models Using Raw Prompts
- **GPT-J**: Completion-style, no special formatting

## Preprocessing Steps

1. **Load the tokenizer** with appropriate configuration
2. **Apply model-specific formatting** (chat template or instruction format)
3. **Tokenize** with proper truncation and max length
4. **Handle padding** (left-side for generation models)

## Example: Generic Preprocessing Function

```python
def preprocess_for_model(text, model_name, backend="pytorch"):
    """Generic preprocessing based on model and backend"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if chat template should be used
    if should_use_chat_template(model_name, backend):
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            truncation=True,
            max_length=get_max_length(model_name)
        )
    else:
        # Apply model-specific template or use raw text
        formatted_text = apply_model_template(text, model_name)
        tokens = tokenizer.encode(
            formatted_text,
            truncation=True,
            max_length=get_max_length(model_name)
        )
    
    return tokens
```

## Max Context Lengths

| Model | Max Length | Notes |
|-------|------------|-------|
| DeepSeek-R1 | 32,768 | 32K context |
| Llama 3.1-8B | 8,000 | For preprocessing |
| Llama 2-70B | 1,024 | Limited context |
| Mixtral-8x7B | 1,024 | From dataset.py |
| GPT-J | ~2,048 | Standard GPT-J limit |

## Running Inference

```bash
# Set backend
export MLPERF_BACKEND=pytorch  # or vllm, sglang

# PyTorch backend (distributed)
torchrun --nproc_per_node=8 run_eval_mpi.py --input-file data.pkl

# vLLM/SGLang backends
python run_eval.py --input-file data.pkl
```

## Common Issues

1. **Wrong padding side**: Always use `padding_side="left"` for generation
2. **Missing pad token**: Set `pad_token = eos_token`
3. **Backend mismatch**: Ensure preprocessing matches backend requirements
4. **Context overflow**: Respect model's maximum context length

## Validation

To ensure correct preprocessing:

1. Check tokenized length doesn't exceed max
2. Verify special tokens are properly placed
3. Test with a few examples before full dataset
4. Compare against reference outputs

## References

- Model-specific guides in each model's directory
- Backend configuration in `utils/backend_registry.py`
- Tokenization utilities in `utils/tokenization.py`