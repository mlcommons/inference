# Dataset Preprocessing Documentation - Llama3.1-8B

## Model: Llama3.1-8B
**Dataset:** CNN/DailyMail 3.0.0  
**Evaluation Task:** Text Summarization

## Data Source
- **Raw Dataset:** Hugging Face `cnn_dailymail` dataset v3.0.0
- **Download Method:** `datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")`
- **License:** Apache 2.0
- **Download Script:** `download_cnndm.py`

## Preprocessing Pipeline

### 1. Tokenization
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 8000
```

### 2. Input Template
```
Summarize the following news article in 128 tokens. Please output the summary only, without any other text.

Article:
{article}

Summary:
```

### 3. Current Implementation
- **Download:** `download_cnndm.py` loads CNN/DailyMail dataset
- **Calibration:** `prepare-calibration.py` creates calibration subset
- **Evaluation:** Uses `evaluation.py` for accuracy assessment

## Missing Documentation (Addresses Issue #2245)

The following preprocessing steps are **not currently documented** but would be needed for full reproducibility:

### 4. Filtering Steps (Recommended)
Based on `llama2-70b/processorca.py` patterns:
- **Language Filter:** English-only content validation
- **Length Filter:** Input/output sequence length limits
- **Quality Filter:** Remove very short summaries
- **Content Filter:** Handle special characters and formatting

### 5. Sampling Strategy (Recommended)
- **Dataset Size:** Specify number of evaluation samples
- **Selection Method:** Random vs stratified sampling
- **Validation:** How to verify preprocessing consistency

## Adaptation Guide

**For Different Tokenizers:**
1. Update `model-id` parameter in scripts
2. Adjust `model_max_length` based on tokenizer capabilities
3. Verify special token handling (pad_token, eos_token)

**For Different Models:**
1. Modify input template format
2. Adjust summary length requirements (currently 128 tokens)
3. Update evaluation criteria as needed

## Files Generated
- **Main Dataset:** Downloaded via `download_cnndm.py`
- **Calibration Set:** Generated via `prepare-calibration.py`
- **Format:** Standard CNN/DailyMail format from Hugging Face

## Next Steps for Full Reproducibility

To fully address issue #2245, consider adding:
1. Complete preprocessing script (similar to `llama2-70b/processorca.py`)
2. Documentation of filtering criteria
3. Sampling methodology
4. Quality validation steps

## See Also
- `llama2-70b/processorca.py` - Reference implementation for comprehensive preprocessing
- `PREPROCESSING-TEMPLATE.md` - Standard template for future models