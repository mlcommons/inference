# Test 08 - Verify accuracy in performance mode for DLRMv3

This repository provides the config files and scripts to run and verify TEST08 - Verify accuracy in performance mode for the DLRMv3 benchmark.

# Table of Contents
1. [Introduction](#introduction)
2. [Why TEST08 Instead of TEST01](#why-test08-instead-of-test01)
3. [Prerequisites](#prerequisites)
4. [Instructions](#instructions)
5. [Pass Criteria](#pass-criteria)

## Introduction

The purpose of this test is to ensure that valid inferences are being performed in performance mode for the DLRMv3 benchmark. By default, the inference result that is returned from SUT to LoadGen is not written to the accuracy JSON file and thus not checked for accuracy. In this test, the inference results of a subset of the total samples issued by LoadGen are written to the accuracy JSON.

## Why TEST08 Instead of TEST01

TEST01 uses `qsl_idx` (Query Sample Library index) as the key to match samples between the accuracy run and the performance run. However, **DLRMv3 cannot use `qsl_idx` for sample matching** due to its streaming dataset constraints.

### The Streaming Constraint

DLRMv3 uses a streaming dataset where queries must be executed in a specific temporal order to satisfy causality constraints:
- The dataset is organized by timestamps (`ts_idx`), where each timestamp contains multiple user queries
- Queries at timestamp `t` can only use historical data from timestamps `< t`
- The execution order is determined by `StreamingQuerySampler.run_order`, not by the `qsl_idx` generated from LoadGen

### Why `qsl_idx` Cannot Be Used

In the DLRMv3 implementation (see `inference/recommendation/dlrm_v3/main.py`):
1. LoadGen generates `qsl_idx` values that do not respect the streaming/timestamp constraints
2. The `StreamingQuerySampler` maintains its own execution order (`run_order`) based on timestamp boundaries
3. Samples are selected using `(ts_idx, query_idx)` pairs that correspond to the actual data being processed, not the LoadGen-provided `qsl_idx`

### Solution: Match by `(ts_idx, query_idx)`

TEST08 addresses this by:
1. Including `ts_idx` and `query_idx` in each sample's output data
2. Matching samples between accuracy and performance runs using the `(ts_idx, query_idx)` pair
3. Comparing the Normalized Entropy (NE) values for matched samples within a tolerance threshold

## Verification Method

For each sample, the data contains:
- `ts_idx`: Timestamp index
- `query_idx`: Query index
- `predictions`: Model prediction scores
- `labels`: Ground truth labels
- `weights`: Sample weights
- `candidate_size`: Number of candidates

Samples from the performance run are matched with samples from the accuracy run using the `(ts_idx, query_idx)` pair. For matched samples, the Normalized Entropy (NE) is computed and compared. The test passes if the relative difference in NE values is within the specified tolerance (default: 0.1%).

## Prerequisites

1. Python 3.3 or later
2. NumPy installed (`pip install numpy`)
3. Submission runs have already been completed with results in the standard submission directory structure

## Instructions

### Part I: Run the benchmark with audit.config

1. Copy the provided `audit.config` from the `dlrm-v3` subdirectory to the directory where the benchmark is being run from:

```bash
cp compliance/TEST08/dlrm-v3/audit.config /path/to/benchmark/working/dir/
```

2. Run the benchmark in performance mode. LoadGen will read `audit.config` and log a subset of inference results.

3. Verify that `audit.config` was properly read by checking that LoadGen has found `audit.config` in `mlperf_log_detail.txt`.

4. **Important:** Remove `audit.config` after the test to prevent accidentally running in compliance mode.

### Part II: Run the verification script

```bash
python3 run_verification.py \
    -r <path to mlperf_log_accuracy.json from accuracy run> \
    -t <path to mlperf_log_accuracy.json from compliance test run> \
    [--tolerance TOLERANCE]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `-r`, `--reference_accuracy` | Yes | Path to `mlperf_log_accuracy.json` from the accuracy run |
| `-t`, `--test_accuracy` | Yes | Path to `mlperf_log_accuracy.json` from the compliance test run |
| `--tolerance` | No | Relative tolerance for NE comparison (default: 0.001 = 0.1%) |

**Example:**

```bash
python3 compliance/TEST08/run_verification.py \
    -r /path/to/results/dlrm-v3/Offline/accuracy/mlperf_log_accuracy.json \
    -t /path/to/compliance/TEST08/mlperf_log_accuracy.json
```

**Expected output:**

```
Verifying accuracy. This might take a while...
Reading accuracy mode results...
Reading performance mode results...

num_acc_log_entries = 34996
num_perf_log_entries = 238
num_matched = 238
num_unmatched = 0
num_ne_mismatch = 0
tolerance = 0.10%

TEST PASS
TEST08 verification complete
```

## Pass Criteria

To pass this test, the following criteria must be satisfied:

1. **All samples matched:** Every sample in the performance accuracy log must have a corresponding sample in the reference accuracy log, matched by `(ts_idx, query_idx)`.

2. **NE values within tolerance:** For all matched samples, the Normalized Entropy (NE) computed from the performance run must be within the specified relative tolerance of the NE from the accuracy run.

   The relative difference is calculated as:
   ```
   relative_diff = |perf_ne - acc_ne| / |acc_ne|
   ```

   By default, this must be ≤ 0.1% (tolerance = 0.001).


If any of these criteria are not met, the test will report `TEST FAIL`.
