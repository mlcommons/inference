# Llama 3.1 8B Preprocessing

## Model Configuration
- **Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Revision**: `be673f326cab4cd22ccfef76109faf68e41aa5f1` (for download)
- **Max Length**: 8,000 tokens (in preprocessing scripts)

## Tokenization
```python
from transformers import AutoTokenizer

# From prepare-calibration.py
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 8000
```

## Prompt Template (CNN/DailyMail Summarization)
```python
# From prepare-calibration.py and download_cnndm.py
instruction_template = "Summarize the following news article in 128 tokens. Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:"

# Tokenize
x["tok_input"] = tokenizer.encode(instruction_template.format_map(x))
```

**Note**: This uses a simple instruction format, NOT the chat template with special tokens.

## Dataset Preparation
```python
# Example from prepare-calibration.py
x = dict()
x["instruction"] = instruction_template
x["input"] = calibration_sample["article"]
x["tok_input"] = tokenizer.encode(instruction_template.format_map(x))
x["output"] = calibration_sample["highlights"]
```

## Accuracy Targets (BF16)
```
Datacenter:
- rouge1: 38.7792
- rouge2: 15.9075
- rougeL: 24.4957
- rougeLsum: 35.793
```