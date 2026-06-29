# TEST09 Compliance for E2E-RAG Workload

## Overview

TEST09 verifies that the output token length during performance runs matches expected values to prevent output truncation cheating.

## Statistics from Reference Implementation

Based on 5 production runs (4021 total answer_generator invocations):

| Run | Samples | Avg OSL |
|-----|---------|---------|
| 1   | 794     | 221     |
| 2   | 805     | 258     |
| 3   | 810     | 273     |
| 4   | 813     | 211     |
| 5   | 799     | 214     |

**Weighted Mean OSL:** 235.47 tokens

**TEST09 Thresholds (±10%):**
- Min output tokens: 211.92
- Max output tokens: 259.02

## Usage

### Automated Workflow (Recommended)

Use the automated compliance test script that handles setup, run, verification, and cleanup:

```bash
cd inference/e2e

# Run TEST09 only
bash run_compliance_test09.sh

# Or run all compliance tests
bash run_all_compliance_tests.sh
```

The script will:
1. ✓ Copy audit.config to working directory
2. ✓ Run performance test with LoadGen compliance logging
3. ✓ Verify output token length thresholds
4. ✓ Copy results to submission directory
5. ✓ Clean up audit.config automatically

**Environment Variables (optional):**

```bash
# Override defaults
export DATABASE=vector_html_hnsw_len768_ov32_word.db
export MAX_ASYNC_QUERIES=10
export MAX_WORKERS=10
export PERF_CACHE_FILE=logs_result.json  # Use cached LLM responses

bash run_compliance_test09.sh
```

### Manual Workflow

### Part I: Setup

Copy the audit.config to your working directory:

```bash
cd inference/e2e-rag
cp ../compliance/TEST09/e2e-rag/audit.config ./
```

### Part II: Run Performance Test

Run the benchmark as normal. LoadGen will automatically detect `audit.config`:

```bash
bash reference_mlperf_perf.sh
```

Or directly:

```bash
python3 reference_mlperf.py \
    --dataset_path data/frames_dataset.tsv \
    --database vector_html_hnsw_len768_ov32_word.db \
    --scenario Offline \
    --log_dir run_output_test09 \
    --perf_count 824
```

**Important:** Remove `audit.config` after the test to avoid running in compliance mode unintentionally.

### Part III: Run Verification

```bash
python3 inference/e2e/third_party/mlperf-inference/compliance/TEST09/run_verification.py \
    -c run_output_test09 \
    -o submission/compliance/e2e-rag/Offline \
    --audit-config ../compliance/TEST09/e2e-rag/audit.config
```

### Expected Output

```
================================================================================
TEST09: Verify Output Token Length in Performance Mode
================================================================================
Output Token Length Statistics
================================================================================
Total samples: 824
Mean output tokens: 235.47
Min output tokens: 1
Max output tokens: 2829
Std deviation: ~150

================================================================================
Verification Results
================================================================================
Mean output tokens: 235.47
Min threshold: 211.92 -> PASS
Max threshold: 259.02 -> PASS

Overall: TEST PASS
```

## Notes

- **Component Measured:** `answer_generator` - this is the only component that LoadGen logs (the final answer generation step)
- **Dataset:** Full 824 queries from frames_dataset.tsv
- **Workload:** Multi-hop RAG with iterative retrieval (max 5 iterations)
- The thresholds are based on the answer_generator component only, not the intermediate multi-shot retrieval steps
- Other components (check_sufficiency, generate_search_queries, evaluate_document_relevance) are internal to the SUT and not measured by TEST09
