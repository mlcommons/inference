# Dataset Preprocessing Documentation Template

## Purpose
This template provides a standardized way to document dataset preprocessing steps for MLCommons inference benchmarks, ensuring reproducibility and transparency.

## Template Structure

### Model: [MODEL_NAME]
**Dataset:** [DATASET_NAME]  
**Evaluation Task:** [TASK_DESCRIPTION]

#### Data Source
- **Raw Dataset:** [SOURCE_AND_FORMAT]
- **Download Method:** [HOW_TO_OBTAIN]
- **License:** [LICENSE_INFO]

#### Preprocessing Pipeline

##### 1. Tokenization
```python
# Example based on llama2-70b/processorca.py pattern
from transformers import [TOKENIZER_CLASS]
tokenizer = [TOKENIZER_CLASS].from_pretrained(model_dir)
tokens = tokenizer(text)["input_ids"]
```

##### 2. Filtering Steps
- **Language Filter:** [DESCRIPTION]
- **Length Filter:** [SEQUENCE_LENGTH_LIMITS]
- **Quality Filter:** [QUALITY_CRITERIA]
- **Content Filter:** [CONTENT_RESTRICTIONS]

##### 3. Formatting
- **Input Format:** [INPUT_TEMPLATE]
- **Output Format:** [OUTPUT_TEMPLATE]
- **Special Tokens:** [SPECIAL_TOKEN_HANDLING]

##### 4. Sampling Strategy
- **Total Samples:** [NUMBER]
- **Sampling Method:** [RANDOM/STRATIFIED/OTHER]
- **Validation Split:** [IF_APPLICABLE]

#### Adaptation Guide
**For Different Tokenizers:**
- Modify tokenizer initialization
- Adjust sequence length limits
- Update special token handling

**For Different Models:**
- Update input/output templates
- Adjust filtering criteria
- Modify prompt formatting

#### Files Generated
- **Main Dataset:** [FILENAME_AND_FORMAT]
- **Calibration Set:** [FILENAME_AND_FORMAT]
- **Metadata:** [FILENAME_AND_FORMAT]

#### Verification
- **Expected Sample Count:** [NUMBER]
- **Checksum/Hash:** [IF_AVAILABLE]
- **Quality Metrics:** [ROUGE/BLEU/OTHER]

---

## Example Applications

### Llama3.1-8b (CNN/DailyMail)
**Dataset:** CNN/DailyMail 3.0.0  
**Evaluation Task:** Text Summarization

#### Data Source
- **Raw Dataset:** Hugging Face `cnn_dailymail` dataset v3.0.0
- **Download Method:** `datasets.load_dataset("cnn_dailymail", "3.0.0")`
- **License:** Apache 2.0

#### Preprocessing Pipeline
##### 1. Tokenization
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 8000
```

##### 2. Formatting
- **Input Template:** 
```
Summarize the following news article in 128 tokens. Please output the summary only, without any other text.

Article:
{article}

Summary:
```

##### 3. Current Gaps
- ❌ No documented filtering steps
- ❌ No sampling strategy explanation  
- ❌ No quality control measures
- ❌ No reproducible preprocessing script

### DeepSeek-r1 (Multi-domain Evaluation)
**Dataset:** Ensemble of AIME, MATH500, GPQA, MMLU-Pro, LiveCodeBench  
**Evaluation Task:** Multi-domain Reasoning

#### Data Source
- **Preprocessed Dataset:** Available via Rclone from Cloudflare R2
- **Download Method:** `rclone copy mlc-inference:mlcommons-inference-wg-public/deepseek_r1/`
- **License:** Various (CC0, MIT, CC BY 4.0)

#### Current Gaps
- ❌ No documented preprocessing steps
- ❌ No tokenization details
- ❌ No filtering or sampling explanation
- ❌ No adaptation guide for other models
- ❌ Cannot reproduce from raw sources

---

## Implementation Recommendation

1. **For each model directory**, add `PREPROCESSING.md` following this template
2. **For models with preprocessing scripts**, document the steps in the README
3. **For models using preprocessed data**, provide original preprocessing methodology
4. **Create common utilities** for preprocessing patterns that can be shared across models