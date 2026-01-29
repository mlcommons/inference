# Quick Reference: GPT-OSS-120B and Qwen3VL Command Examples

## GPT-OSS-120B Quick Examples

### Basic Performance Run
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

### Accuracy Run with Deterministic Parameters
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --accuracy-temperature 0.001 \
    --accuracy-top-k 1 \
    --accuracy-top-p 1.0 \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

### With Generation Config File
```bash
python harness/harness_gpt_oss_120b.py \
    --model openai/gpt-oss-120b \
    --dataset-path /path/to/tokenized_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --generation-config /path/to/generation_config.json \
    --api-server-url http://localhost:30000 \
    --output-dir ./harness_output
```

### Server Scenario
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

### Back-to-Back Requests (Offline)
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

### Using Model Category (Auto-detection)
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

---

## Qwen3VL Quick Examples

### Basic Performance Run
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Offline \
    --test-mode performance \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output
```

### Accuracy Run with Guided Decoding
```bash
python harness/harness_qwen3vl.py \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset-path /path/to/multimodal_dataset.pkl \
    --scenario Offline \
    --test-mode accuracy \
    --use-guided-decoding \
    --accuracy-temperature 0.0 \
    --api-server-url http://localhost:8000 \
    --output-dir ./harness_output
```

### Server Scenario
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

### Using Model Category
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

---

## Common Options

### Sampling Parameters (GPT-OSS-120B)
```bash
# Performance mode
--temperature 1.0 --top-k -1 --top-p 1.0

# Accuracy mode
--accuracy-temperature 0.001 --accuracy-top-k 1 --accuracy-top-p 1.0
```

### Sampling Parameters (Qwen3VL)
```bash
# Accuracy mode
--accuracy-temperature 0.0 --accuracy-top-p 1.0
```

### Server Configuration
```bash
# SGLang (GPT-OSS-120B)
--server-config configs/backends/sglang.yaml

# vLLM (Qwen3VL)
--server-config configs/backends/vllm.yaml
```

### Offline Back-to-Back
```bash
--offline-back-to-back
```

### Metrics Collection
```bash
--enable-metrics --metrics-interval 15
```

### Debug Mode
```bash
--debug-mode
```

---

## Full Command Examples

### GPT-OSS-120B: Complete Accuracy Run
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

### Qwen3VL: Complete Performance Run
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

---

## Notes

- **GPT-OSS-120B**: Requires SGLang server running on port 30000 (default)
- **Qwen3VL**: Requires vLLM server running on port 8000 (default)
- **Dataset Format**: 
  - GPT-OSS-120B: Pre-tokenized (Parquet/Pickle with `tok_input` or `input_tokens` column)
  - Qwen3VL: Multimodal (Parquet/Pickle with `messages` column or raw fields)
- **Back-to-Back**: Recommended for SGLang (handles batching internally)

For detailed documentation, see: `docs/gpt-oss-120b-and-qwen3vl-support.md`
