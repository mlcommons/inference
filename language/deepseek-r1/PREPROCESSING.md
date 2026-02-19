# DeepSeek-R1 Preprocessing

## Model Configuration
- **Model**: `deepseek-ai/DeepSeek-R1`
- **Revision**: `56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad`
- **Max Length**: 32,768 tokens (32K)

## Tokenization
```python
from transformers import AutoTokenizer

# From utils/tokenization.py
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1",
    revision="56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"
)
```

## Preprocessing Method

The preprocessing varies by backend:

### PyTorch/vLLM Backends (Chat Template Enabled)
```python
# From utils/tokenization.py
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    max_length=32768,
    truncation=True
)
```

### SGLang Backend (No Chat Template)
```python
tokens = tokenizer.encode(
    prompt,
    truncation=True,
    max_length=32768
)
```

## Backend Configuration
| Backend | uses_chat_template | input_type |
|---------|-------------------|------------|
| PyTorch | True | tokenized |
| vLLM | True | text |
| SGLang | False | text |

## Dataset Format
Input data should have a `text_input` column containing the prompts.

## Accuracy Target
```
"mean-accuracy": 81.3582
```