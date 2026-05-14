# Test 09 - Verify Output Token Length in Performance Mode

This repository provides the config files and scripts to run and verify TEST09 - Verify output token length in performance mode for LLM workloads.

# Table of Contents

1. [Applicable Benchmarks](#applicable-benchmarks)
2. [Introduction](#introduction)
3. [Prerequisites](#prerequisites)
4. [Instructions](#instructions)
5. [Adding New Benchmarks](#adding-new-benchmarks)

## Applicable Benchmarks

| Model        | Min Output Tokens | Max Output Tokens | Dataset Size | Notes                          |
| ------------ | ----------------- | ----------------- | ------------ | ------------------------------ |
| gpt-oss-120b | 1150.38           | 1406.02           | 6396         | ±10% of 1278.20 reference mean |

## Introduction

The purpose of this test is to ensure that models are generating outputs of expected length during performance runs. This prevents cheating by truncating outputs to artificially improve throughput metrics.

**Key Verification:**

| Metric             | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| Mean output tokens | Average number of output tokens across all samples                       |
| Min threshold      | Benchmark-specific minimum - ensures outputs are not truncated           |
| Max threshold      | Benchmark-specific maximum - ensures outputs are not artificially padded |

The compliance thresholds are defined in the benchmark's `audit.config` file via the `test09_min_output_tokens` and `test09_max_output_tokens` fields. Each benchmark defines its own bounds based on the reference implementation.

## Prerequisites

1. Python 3.8 or later
2. The MLPerf accuracy log (`mlperf_log_accuracy.json`) from a compliance run

## Instructions

### Part I: Setup

Copy the provided `audit.config` from the benchmark subdirectory to your benchmark's working directory:

```bash
# For gpt-oss-120b
cp compliance/TEST09/gpt-oss-120b/audit.config /path/to/benchmark/working/dir/
```

The `audit.config` contains both LoadGen settings and the compliance thresholds:

```
# LoadGen settings
*.*.mode = 2
*.*.accuracy_log_sampling_target = 10000
*.*.min_query_count = 6396
...

# TEST09 Compliance Thresholds (read by run_verification.py)
*.*.test09_min_output_tokens = 1150.38
*.*.test09_max_output_tokens = 1406.02
```

### Part II: Run the benchmark

Run the benchmark as you normally would. LoadGen will read `audit.config` and log all inference results.

```bash
# Example for gpt-oss-120b
python3 run_mlperf.py --scenario offline --input-file /path/to/dataset.parquet ...
```

Verify that `audit.config` was properly read by checking `mlperf_log_detail.txt` for the detection message.

**Important:** Remove `audit.config` after the test to prevent accidentally running in compliance mode.

### Part III: Run verification

```bash
python3 run_verification.py \
    -c COMPLIANCE_DIR \
    -o OUTPUT_DIR \
    --audit-config /path/to/audit.config
```

**Arguments:**

| Argument                 | Required | Description                                                        |
| ------------------------ | -------- | ------------------------------------------------------------------ |
| `-c`, `--compliance_dir` | Yes      | Path to compliance test logs (contains `mlperf_log_accuracy.json`) |
| `-o`, `--output_dir`     | Yes      | Output directory for submission artifacts                          |
| `--audit-config`         | No\*     | Path to audit.config containing thresholds                         |
| `--min-output-tokens`    | No\*     | Override minimum threshold (CLI takes precedence)                  |
| `--max-output-tokens`    | No\*     | Override maximum threshold (CLI takes precedence)                  |

\*At least one of `--audit-config` or both `--min-output-tokens` and `--max-output-tokens` must be provided.

### Example: gpt-oss-120b

**Dataset:** Use `perf/perf_eval_ref.parquet` (the performance dataset) from the gpt-oss dataset download for TEST09 compliance runs.

**Generation Config:** Use performance generation config as-is (`max_output_len=10240`, `reasoning_effort=low`). This test verifies mean output sequence length is within ±10% of reference (1278.20 tokens).

```bash
python3 compliance/TEST09/run_verification.py \
    -c /path/to/compliance/run/logs/ \
    -o /path/to/submission/compliance/gpt-oss-120b/Offline \
    --audit-config compliance/TEST09/gpt-oss-120b/audit.config
```

**Expected output:**

```
================================================================================
TEST09: Verify Output Token Length in Performance Mode
================================================================================
Reading audit.config from: compliance/TEST09/gpt-oss-120b/audit.config
Found min_output_tokens in audit.config: 1150.38
Found max_output_tokens in audit.config: 1406.02

Using thresholds:
  Min output tokens: 1150.38
  Max output tokens: 1406.02
================================================================================

Parsing MLPerf accuracy log...
Loaded 6396 entries as JSON array

Computing output token lengths for 6396 samples...

================================================================================
Output Token Length Statistics
================================================================================
Total samples: 6396
Mean output tokens: 1278.20
Min output tokens: 518
Max output tokens: 3154
Std deviation: 247.0

================================================================================
Verification Results
================================================================================
Mean output tokens: 1278.20
Min threshold: 1150.38 -> PASS
Max threshold: 1406.02 -> PASS

Overall: TEST PASS
```

### Part IV: Submit

The verification script copies the following files to the output directory:

```
TEST09/
├── verify_output_len.txt
├── accuracy/
│   └── mlperf_log_accuracy.json
└── performance/
    └── run_1/
        ├── mlperf_log_summary.txt
        └── mlperf_log_detail.txt
```

These files must be submitted as part of the compliance audit trail.

## Adding New Benchmarks

To add TEST09 support for a new benchmark:

### 1. Create benchmark-specific audit.config

Create `compliance/TEST09/<benchmark>/audit.config`:

```conf
# LoadGen settings
*.*.mode = 2
*.*.accuracy_log_sampling_target = <dataset_size_or_larger>
*.*.min_query_count = <dataset_size>
*.*.min_duration = 0
*.*.sample_concatenate_permutation = 0

# TEST09 Compliance Thresholds
# Define bounds based on reference implementation output lengths
*.*.test09_min_output_tokens = <min_threshold>
*.*.test09_max_output_tokens = <max_threshold>
```

### 2. Update this README

Add the benchmark to the "Applicable Benchmarks" table with:

- Model name
- Min output tokens threshold
- Max output tokens threshold
- Dataset size

### 3. Update submission checker

Add the model to `models_TEST09` list in `tools/submission/submission_checker/constants.py`.

## Troubleshooting

### Mean output tokens below minimum

1. Check that the model is not truncating outputs prematurely
2. Verify the stop token / EOS handling is correct
3. Ensure max_new_tokens or similar settings match the reference implementation

### Mean output tokens above maximum

1. Check for excessive padding or repetition in outputs
2. Verify the model is correctly detecting end-of-sequence
3. Review generation parameters (temperature, top_p, etc.)

### No samples found

1. Verify `accuracy_log_sampling_target` is set high enough to capture all samples
2. Check that the compliance run completed successfully
3. Ensure `mlperf_log_accuracy.json` is in the expected location
