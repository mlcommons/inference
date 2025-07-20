# Dataset Preprocessing Documentation - DeepSeek-R1

## Model: DeepSeek-R1
**Dataset:** Multi-domain Evaluation Ensemble  
**Evaluation Task:** Multi-domain Reasoning and Code Generation

## Data Source
- **Preprocessed Dataset:** Available via Rclone from Cloudflare R2 bucket
- **Download Method:** `rclone copy mlc-inference:mlcommons-inference-wg-public/deepseek_r1/`
- **Components:** AIME, MATH500, GPQA, MMLU-Pro, LiveCodeBench (code_generation_lite)
- **Licenses:** 
  - AIME: [CC0](https://creativecommons.org/public-domain/cc0/)
  - MATH500: [MIT](https://opensource.org/license/mit)
  - GPQA: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  - MMLU-Pro: [MIT](https://opensource.org/license/mit)
  - LiveCodeBench: [CC](https://creativecommons.org/share-your-work/cclicenses/)

## Current Implementation

### Files Available
- **Main Dataset:** `mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl`
- **Calibration Set:** `mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl`
- **Format:** Preprocessed pickle files ready for evaluation

### Download Process
```bash
# Install Rclone
sudo -v ; curl https://rclone.org/install.sh | sudo bash

# Configure access
rclone config create mlc-inference s3 provider=Cloudflare \
  access_key_id=f65ba5eef400db161ea49967de89f47b \
  secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b \
  endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

# Download datasets
rclone copy mlc-inference:mlcommons-inference-wg-public/deepseek_r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl ./ -P
rclone copy mlc-inference:mlcommons-inference-wg-public/deepseek_r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl ./ -P
```

## Missing Documentation (Addresses Issue #2245)

The following preprocessing information is **not currently available**, making reproduction and adaptation difficult:

### 1. Original Data Sources
- **Raw Dataset Locations:** Where each component dataset was obtained
- **Version Information:** Specific versions/commits of source datasets
- **Access Methods:** How to obtain raw data independently

### 2. Preprocessing Pipeline
- **Tokenization Method:** Which tokenizer was used and configuration
- **Input Formatting:** How different dataset formats were standardized
- **Quality Filtering:** Criteria for sample inclusion/exclusion
- **Ensemble Strategy:** How multiple datasets were combined

### 3. Dataset Statistics
- **Sample Counts:** Number of samples from each component dataset
- **Distribution:** How samples are balanced across domains
- **Difficulty Levels:** Complexity distribution of included problems

### 4. Validation Process
- **Quality Control:** How preprocessing quality was verified
- **Consistency Checks:** Validation of format standardization
- **Error Handling:** How malformed samples were addressed

## Adaptation Challenges

**For Different Tokenizers:**
- Cannot modify tokenization without access to raw data
- No documentation of original tokenization parameters
- Unable to test preprocessing consistency

**For Different Models:**
- Cannot adapt input formatting without preprocessing scripts
- No guidance on prompt template modifications
- Unable to reproduce dataset with different filtering criteria

## Recommended Improvements

To fully address issue #2245 and improve reproducibility:

### 1. Raw Data Access
- Provide scripts to download original datasets
- Document exact versions and sources used
- Include data licenses and attribution

### 2. Preprocessing Scripts
- Create preprocessing pipeline (similar to `llama2-70b/processorca.py`)
- Document tokenization and formatting steps
- Include quality filtering logic

### 3. Documentation
- Add detailed preprocessing methodology
- Include dataset statistics and composition
- Provide adaptation guidelines

### 4. Validation
- Include preprocessing verification scripts
- Document expected outputs and checksums
- Provide quality metrics

## Temporary Workaround

Until full preprocessing documentation is available:
1. Use provided preprocessed datasets for standard evaluation
2. Contact maintainers for specific adaptation requirements
3. Reference `llama2-70b/processorca.py` for preprocessing patterns
4. Consider contributing preprocessing scripts based on reverse engineering

## See Also
- `llama2-70b/processorca.py` - Reference implementation for comprehensive preprocessing
- `PREPROCESSING-TEMPLATE.md` - Standard template for future models
- Repository issue #2245 - Discussion of preprocessing documentation gaps