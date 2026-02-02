# Test 07 - Verify accuracy in performance mode (full dataset)

This repository provides the config files and scripts to run and verify TEST07 - Verify accuracy in performance mode for workloads that require logging all samples and checking against an accuracy threshold.

# Table of Contents

1. [Applicable Benchmarks](#applicable-benchmarks)
2. [Introduction](#introduction)
3. [Prerequisites](#prerequisites)
4. [Instructions](#instructions)
5. [Adding New Benchmarks](#adding-new-benchmarks)

## Applicable Benchmarks

| Model        | Accuracy Threshold | Score Pattern            | Dataset Size |
| ------------ | ------------------ | ------------------------ | ------------ |
| gpt-oss-120b | 60.698             | `'exact_match': <score>` | 990          |

## Introduction

The purpose of this test is to ensure that valid inferences are being performed in performance mode for workloads where accuracy must be verified by logging **all** samples and computing the accuracy score against a compliance threshold.
The compliance threshold is defined in the benchmark's `audit.config` file via the `test07_accuracy_threshold` field.

## Prerequisites

1. Python 3.8 or later
2. The benchmark's accuracy evaluation script
3. Reference data file containing ground truth annotations (if required by accuracy script)

## Instructions

### Part I: Setup

Copy the provided `audit.config` from the benchmark subdirectory to your benchmark's working directory:

```bash
# For gpt-oss-120b
cp compliance/TEST07/gpt-oss-120b/audit.config /path/to/benchmark/working/dir/
```

The `audit.config` contains both LoadGen settings and the compliance threshold:

```
# LoadGen settings example for gpt-oss-120b
*.*.mode = 2
*.*.accuracy_log_sampling_target = 10000
*.*.min_query_count = 990
...

# TEST07 Compliance Threshold (read by run_verification.py)
*.*.test07_accuracy_threshold = 60.698
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
    --accuracy-script "ACCURACY_COMMAND" \
    --audit-config /path/to/audit.config
```

**Arguments:**

| Argument                 | Required | Description                                                             |
| ------------------------ | -------- | ----------------------------------------------------------------------- |
| `-c`, `--compliance_dir` | Yes      | Path to compliance test logs (contains `mlperf_log_accuracy.json`)      |
| `-o`, `--output_dir`     | Yes      | Output directory for submission artifacts                               |
| `--accuracy-script`      | Yes      | Command to run accuracy evaluation. Use `{accuracy_log}` as placeholder |
| `--audit-config`         | No\*     | Path to audit.config containing `test07_accuracy_threshold`             |
| `--accuracy-threshold`   | No\*     | Override threshold (CLI takes precedence over audit.config)             |
| `--score-pattern`        | No       | Regex to extract score (default: `'exact_match':\s*([\d.]+)`)           |

\*At least one of `--audit-config` or `--accuracy-threshold` must be provided.

### Example: gpt-oss-120b

**Dataset:** Use `acc/acc_eval_compliance_gpqa.parquet` from the gpt-oss dataset download for TEST07 compliance runs.

**Generation Config:** Use perf generation config (`max_output_len=10240`, `reasoning_effort=low`).

```bash
python3 compliance/TEST07/run_verification.py \
    -c /path/to/compliance/run/logs/ \
    -o /path/to/submission/compliance/gpt-oss-120b/Offline \
    --audit-config compliance/TEST07/gpt-oss-120b/audit.config \
    --accuracy-script "python3 language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log {accuracy_log} \
        --reference-data /path/to/acc/acc_eval_compliance_gpqa.parquet \
        --tokenizer openai/gpt-oss-120b"
```

**Expected output:**

```
Reading audit.config from: compliance/TEST07/gpt-oss-120b/audit.config
Found threshold in audit.config: 60.698
Using accuracy threshold: 60.698

Running accuracy evaluation script...
================================================================================
...
'exact_match': 62.15

================================================================================
Accuracy score: 62.15
Accuracy threshold: 60.698

Accuracy check pass: True
TEST07 verification complete
```

### Part IV: Submit

The verification script copies the following files to the output directory:

```
TEST07/
├── verify_accuracy.txt
├── accuracy/
│   └── mlperf_log_accuracy.json
└── performance/
    └── run_1/
        ├── mlperf_log_summary.txt
        └── mlperf_log_detail.txt
```

These files must be submitted as part of the compliance audit trail.

## Adding New Benchmarks

To add TEST07 support for a new benchmark:

### 1. Create benchmark-specific audit.config

Create `compliance/TEST07/<benchmark>/audit.config`:

```conf
# LoadGen settings
*.*.mode = 2
*.*.accuracy_log_sampling_target = <dataset_size_or_larger>
*.*.min_query_count = <dataset_size>
*.*.min_duration = 0
*.*.sample_concatenate_permutation = 0

# TEST07 Compliance Threshold
*.*.test07_accuracy_threshold = <threshold_value>
```

### 2. Update this README

Add the benchmark to the "Applicable Benchmarks" table with:

- Model name
- Accuracy threshold
- Score pattern (regex to extract accuracy from script output)
- Dataset size

### 3. Ensure accuracy script compatibility

The benchmark's accuracy evaluation script must:

- Accept `mlperf_log_accuracy.json` as input
- Output the accuracy score in a parseable format (e.g., `'exact_match': 62.15`)

## Troubleshooting

### Accuracy below threshold

1. Verify the reference data matches the dataset used during the compliance run
2. Check that all samples were logged (compare entry count in `mlperf_log_accuracy.json` with expected dataset size)
3. Review `verify_accuracy.txt` for detailed breakdown

### Score not parsed

1. Check that your accuracy script outputs the score in the expected format
2. Use `--score-pattern` to provide a custom regex matching your output
3. The pattern should have one capture group for the numeric score

### Threshold not found

1. Ensure `--audit-config` points to the correct file
2. Verify the file contains `*.*.test07_accuracy_threshold = <value>`
3. Alternatively, provide `--accuracy-threshold` directly
