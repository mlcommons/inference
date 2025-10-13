# Prefix Efficiency Dashboard

A comprehensive tool for analyzing dataset prefix caching potential with multiple metrics and optional integration with vLLM Prometheus metrics.

## Overview

This tool computes various prefix caching metrics to help optimize inference performance by analyzing how much text can be reused across different inputs. It supports both text data (requiring tokenization) and pre-tokenized data.

### Metrics Computed

- **PRR (Prefix Reuse Ratio)**: Fraction of inputs that share prefixes with other inputs
- **PDI (Prefix Diversity Index)**: Curve showing prefix uniqueness across different lengths
- **PE (Prefix Entropy)**: Information entropy of prefix distribution
- **PUS (Prefix Uniqueness Score)**: Mean number of other inputs sharing the same prefix
- **POL (Prefix Overlap Length)**: Histogram of shared prefix lengths via randomized sampling

## Installation

### Dependencies

```bash
pip install pandas numpy matplotlib tqdm transformers
```

Optional dependencies:
- `datasets` - for Hugging Face dataset support
- `requests` - for vLLM metrics scraping

## Usage

### Basic Command Structure

```bash
python3 prefix_caching_dashboard.py [OPTIONS]
```

### Required Arguments

You must specify either:
- `--dataset` - Hugging Face dataset name
- `--file` - Local file path

### Data Source Options

#### Hugging Face Datasets
```bash
python3 prefix_caching_dashboard.py \
    --dataset tatsu-lab/alpaca \
    --field instruction \
    --tokenizer meta-llama/Meta-Llama-3-8B \
    --prefix-len 128 \
    --sample-size 4000 \
    --plots
```

#### Local Files
```bash
# Text data (requires tokenizer)
python3 prefix_caching_dashboard.py \
    --file data.jsonl \
    --field prompt \
    --tokenizer gpt2 \
    --prefix-len 128 \
    --sample-size 2000 \
    --plots

# Pre-tokenized data (no tokenizer needed)
python3 prefix_caching_dashboard.py \
    --file tokenized_data.pkl \
    --token-column token_ids \
    --prefix-len 128 \
    --sample-size 4000 \
    --plots

# Auto-detect pre-tokenized data
python3 prefix_caching_dashboard.py \
    --file tokenized_data.pkl \
    --inspect
```

## Command Line Options

### Data Source (Required - choose one)
- `--dataset DATASET` - Hugging Face dataset name (e.g., tatsu-lab/alpaca)
- `--file FILE` - Local file path (.txt, .jsonl, .json, .pickle/.pkl)

### Data Configuration
- `--split SPLIT` - Dataset split to use (default: train)
- `--field FIELD` - Field to read from dataset/JSONL/JSON (auto-detected if not specified)
- `--token-column COLUMN` - Column name containing pre-tokenized data (auto-detected if not specified)
- `--sample-size SIZE` - Maximum samples to analyze (default: 2000)

### Tokenization
- `--tokenizer TOKENIZER` - Hugging Face tokenizer name (required for text data, not needed for pre-tokenized data)
- `--prefix-len LENGTH` - Prefix length for PRR/PE/PUS (default: 128)
- `--hash HASH` - Hash function for prefix identity: md5, sha1, sha256, pyhash (default: md5)

### Analysis Configuration
- `--pdi-lens LENGTHS` - Comma-separated prefix lengths for PDI curve (default: 16,32,64,128,256,512)
- `--overlap-pairs PAIRS` - Random pairs to sample for overlap histogram (default: 3000)

### Output Options
- `--plots` - Generate plots (PDI curve + overlap histogram)
- `--outdir DIR` - Directory to write plots and report (default: .)

### File Inspection
- `--inspect` - Inspect file contents and show columns, types, and sample rows
- `--inspect-rows ROWS` - Number of sample rows to show during inspection (default: 5)

### Data Export
- `--export-fields FIELDS` - Comma-separated list of fields to export to text file
- `--export-file FILE` - Output file path for exported fields (default: exported_data.txt)
- `--export-format FORMAT` - Export format: txt, json, csv (default: txt)

### vLLM Integration (Optional)
- `--vllm-metrics-url URL` - Prometheus text endpoint for vLLM (e.g., http://localhost:8000/metrics)
- `--vllm-metrics-file FILE` - Path to a text dump of /metrics

## Usage Examples

### 1. File Inspection

```bash
# Basic inspection
python3 prefix_caching_dashboard.py --file data.pkl --inspect

# Inspection with more sample rows
python3 prefix_caching_dashboard.py --file data.pkl --inspect --inspect-rows 10

# Inspection with specific field
python3 prefix_caching_dashboard.py --file data.json --field prompt --inspect
```

### 2. Data Export

```bash
# Export specific fields to text file
python3 prefix_caching_dashboard.py \
    --file data.pkl \
    --export-fields prompt,response \
    --export-file prompts.txt

# Export to JSON format
python3 prefix_caching_dashboard.py \
    --file data.pkl \
    --export-fields prompt,response \
    --export-format json \
    --export-file data.json

# Export to CSV format
python3 prefix_caching_dashboard.py \
    --file data.pkl \
    --export-fields prompt,response,score \
    --export-format csv \
    --export-file data.csv
```

### 3. Text Data Analysis

```bash
# Full analysis with plots
python3 prefix_caching_dashboard.py \
    --file data.jsonl \
    --field prompt \
    --tokenizer gpt2 \
    --prefix-len 128 \
    --sample-size 2000 \
    --plots

# Analysis with custom PDI lengths
python3 prefix_caching_dashboard.py \
    --file data.jsonl \
    --tokenizer gpt2 \
    --pdi-lens 32,64,128,256 \
    --plots
```

### 4. Pre-tokenized Data Analysis

```bash
# Using specified token column
python3 prefix_caching_dashboard.py \
    --file tokenized_data.pkl \
    --token-column token_ids \
    --prefix-len 128 \
    --sample-size 4000 \
    --plots

# Auto-detect token column
python3 prefix_caching_dashboard.py \
    --file tokenized_data.pkl \
    --prefix-len 128 \
    --plots
```

### 5. Hugging Face Datasets

```bash
# Using HF dataset
python3 prefix_caching_dashboard.py \
    --dataset tatsu-lab/alpaca \
    --field instruction \
    --tokenizer meta-llama/Meta-Llama-3-8B \
    --prefix-len 128 \
    --sample-size 4000 \
    --plots
```

### 6. vLLM Integration

```bash
# With vLLM metrics correlation
python3 prefix_caching_dashboard.py \
    --file data.pkl \
    --token-column token_ids \
    --plots \
    --vllm-metrics-url http://localhost:8000/metrics

# With vLLM metrics file
python3 prefix_caching_dashboard.py \
    --file data.pkl \
    --token-column token_ids \
    --plots \
    --vllm-metrics-file vllm_metrics.txt
```

### 7. Combined Operations

```bash
# Inspect, export, and analyze
python3 prefix_caching_dashboard.py \
    --file data.pkl \
    --inspect \
    --export-fields prompt,response \
    --export-file prompts.txt \
    --token-column token_ids \
    --plots
```

## Output Files

### Generated Files

1. **`metric_summary.txt`** - Human-readable report with all computed metrics
2. **`pdi_curve.png`** - Plot showing Prefix Diversity Index curve
3. **`prefix_overlap_histogram.png`** - Histogram of prefix overlap lengths
4. **Exported data files** - Based on `--export-file` and `--export-format` options

### Sample Output

```
Prefix Efficiency Dashboard — Summary
====================================

Samples analyzed: 2000
Data source: data.pkl
Tokenizer: gpt2
Prefix length (PRR/PE/PUS): 128
Hash: md5
Memory usage: 45.23 MB

PRR  (Prefix Reuse Ratio): 0.3245
PUS  (Prefix Uniqueness Score, mean #others sharing prefix): 2.1567
PE   (Prefix Entropy, bits): 8.2341

POL  (Prefix Overlap Length) stats (random pairs):
  - min: 0
  - p50: 12
  - p90: 45
  - max: 89
  - mean: 15.67

PDI  (Prefix Diversity Index) curve:
  - L=  16: 0.8234
  - L=  32: 0.7456
  - L=  64: 0.6789
  - L= 128: 0.6123
  - L= 256: 0.5456
  - L= 512: 0.4789
```

## File Format Support

### Input Formats

- **`.txt`** - Plain text files (one text per line)
- **`.jsonl`** - JSON Lines format
- **`.json`** - JSON files (arrays or objects)
- **`.pickle/.pkl`** - Pickle files (lists, dicts, or DataFrames)

### Export Formats

- **`.txt`** - Human-readable text format
- **`.json`** - JSON format
- **`.csv`** - CSV format

## Pre-tokenized Data Support

The tool automatically detects pre-tokenized data by looking for:
- Column names containing: `token_ids`, `tokens`, `input_ids`, `encoded_tokens`, `tokenized`, `ids`
- Data types: lists of integers, numpy arrays of integers

When pre-tokenized data is detected:
- No tokenizer is required
- Faster processing (no tokenization step)
- Automatic truncation to specified prefix length

## Tips and Best Practices

1. **For large datasets**: Use `--sample-size` to limit analysis to a manageable subset
2. **For pre-tokenized data**: Use `--token-column` to specify the exact column name
3. **For inspection**: Always run `--inspect` first to understand your data structure
4. **For export**: Use `--export-fields` to extract specific columns for external analysis
5. **For vLLM integration**: Ensure your vLLM server is running and accessible at the metrics URL

## Troubleshooting

### Common Issues

1. **"Tokenizer required" error**: Use `--tokenizer` for text data or `--token-column` for pre-tokenized data
2. **"No data found" error**: Check file path and format
3. **"Field not found" error**: Use `--inspect` to see available fields
4. **Memory issues**: Reduce `--sample-size` for large datasets

### Getting Help

Run with `--help` to see all available options:
```bash
python3 prefix_caching_dashboard.py --help
```

## License

This tool is part of the MLPerf Inference benchmark suite.
