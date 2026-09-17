# Accuracy Tests vs Performance Tests - Critical Differences

## The Critical Distinction

### Accuracy Tests (Normal Mode)
**Purpose**: Measure how well the model performs on the task
- ✅ Use REAL LLM responses (no cache)
- ✅ Measure retrieval precision/recall/F1
- ✅ Measure LLM judge accuracy
- ✅ Test with different models/prompts/parameters
- ❌ **NEVER use --perf-test-mode flag**
- ❌ **NEVER set INFERENCE_PERF_TEST_MODE**

### Performance Tests (Perf Test Mode)
**Purpose**: Measure system performance (latency, throughput)
- ✅ Use CACHED responses for deterministic workload
- ✅ Measure real LLM latency
- ✅ Measure retrieval/reranking speed
- ✅ Enable run-to-run comparisons
- ✅ **MUST use --perf-test-mode flag**
- ✅ **MUST set INFERENCE_PERF_TEST_MODE**

## The Problem You Found

### Your Accuracy Test Log Showed:
```
================================================================================
PERFORMANCE TEST MODE ENABLED
================================================================================
Loading performance test cache from output_multi_shot_full_w20_20260520_153456/llm_logs_multi_shot_20260520_153501.json
  Cache loaded successfully:
    - Total cached responses: 7522
```

**This is WRONG for an accuracy test!** ❌

The test was using cached LLM responses from a previous run, which means:
- ❌ Not testing real LLM behavior
- ❌ Not measuring current model accuracy
- ❌ Just replaying old responses
- ❌ Accuracy metrics are meaningless (from old model/prompts)

## Root Cause

Your `run_accuracy_test.sh` had:
```bash
# WRONG - This enables perf test mode!
INFERENCE_PERF_TEST_MODE="output_multi_shot_full_w20_20260520_153456/llm_logs_multi_shot_20260520_153501.json" \
OPENROUTER_API_KEY=sk-or-v1-... \
bash scripts/run_multi_shot.sh 1 1
```

## The Fix

### Correct `run_accuracy_test.sh` (Fixed):
```bash
#!/bin/bash
# Accuracy test - DO NOT use perf test mode

echo "Running ACCURACY test (NOT performance test mode)"
echo "This will use real LLM calls to measure actual model accuracy"

# DO NOT SET INFERENCE_PERF_TEST_MODE for accuracy tests!
OPENROUTER_API_KEY=sk-or-v1-... \
bash scripts/run_multi_shot.sh 1 1
```

**Key change**: Removed `INFERENCE_PERF_TEST_MODE` environment variable

### Correct `run_perf_test.sh` (Performance Test):
```bash
#!/bin/bash
# Performance test - MUST use perf test mode

echo "Running PERFORMANCE test (with cached responses)"
echo "This will measure system performance with deterministic workload"

# MUST SET INFERENCE_PERF_TEST_MODE for performance tests!
INFERENCE_PERF_TEST_MODE="output_multi_shot_full_w20_20260520_153456/llm_logs_multi_shot_20260520_153501.json" \
OPENROUTER_API_KEY=sk-or-v1-... \
bash scripts/run_multi_shot.sh 1 1
```

## Expected Behavior

### Accuracy Test (Normal Mode)

**Log should show:**
```
=== Multi-shot retrieval ===
  Model (grader):    openai/gpt-4o-mini
  Model (query gen): openai/gpt-4o-mini
  ...

# NO "PERFORMANCE TEST MODE ENABLED" message!

[Query 1/50]

ITERATION 1/5
────────────────────────────────────────────────────────────────────────────────

  Evaluating documents and generating queries...
  [ITERATION 1] Decomposing original query via generate_search_queries...
    [DEBUG] Query gen LLM raw output: {
  "queries": [
    "Harriet Lane mother",
    "James A. Garfield mother maiden name"
  ],
  ...

✓ Real LLM call made
✓ Real response used in pipeline
✓ Accuracy measured against current model behavior
```

**No messages about**:
- ❌ "PERFORMANCE TEST MODE ENABLED"
- ❌ "Loading performance test cache"
- ❌ "Returning simulated response"
- ❌ "Using cached response"

### Performance Test (Perf Test Mode)

**Log should show:**
```
=== Multi-shot retrieval ===
  ...

================================================================================
PERFORMANCE TEST MODE ENABLED
================================================================================
Loading performance test cache from ...
  Cache loaded successfully:
    - Total cached responses: 7522
================================================================================

[Query 1/50]

ITERATION 1/5
────────────────────────────────────────────────────────────────────────────────

  Evaluating documents and generating queries...
  [ITERATION 1] Decomposing original query via generate_search_queries...
    [PERF TEST MODE] Will attempt real LLM call for performance measurement...
    [PERF TEST MODE] Returning simulated response (LLM generated: 245 chars, Simulated: 247 chars)
    [DEBUG] Query gen LLM raw output: {
  "queries": [
    "query X",
    "query Y"
  ],
  ...

✓ Real LLM call made (performance measured)
✓ Cached response returned to pipeline (deterministic)
✓ Performance benchmarking with identical workload
```

## Why This Distinction Matters

### Accuracy Testing Use Cases

1. **Testing a new model**
   ```bash
   # Test gpt-4o vs gpt-4o-mini
   INFERENCE_MODEL="openai/gpt-4o" bash scripts/run_multi_shot.sh 50 1
   # Compare accuracy scores
   ```

2. **Testing prompt changes**
   ```bash
   # Modified prompt in multi_shot_retrieval.py
   bash scripts/run_multi_shot.sh 50 1
   # See if accuracy improved
   ```

3. **Testing different temperatures**
   ```bash
   INFERENCE_TEMPERATURE=0.0 bash scripts/run_multi_shot.sh 50 1
   INFERENCE_TEMPERATURE=1.0 bash scripts/run_multi_shot.sh 50 1
   # Compare deterministic vs creative responses
   ```

4. **Testing retrieval parameters**
   ```bash
   INFERENCE_TOP_K_RETRIEVER=10 bash scripts/run_multi_shot.sh 50 1
   INFERENCE_TOP_K_RETRIEVER=20 bash scripts/run_multi_shot.sh 50 1
   # See if more docs help accuracy
   ```

**All of these MUST use real LLM responses (no perf test mode)**

### Performance Testing Use Cases

1. **Comparing system configurations**
   ```bash
   # Run 1: 1 worker
   INFERENCE_PERF_TEST_MODE="cache.json" bash scripts/run_multi_shot.sh 50 1
   
   # Run 2: 4 workers
   INFERENCE_PERF_TEST_MODE="cache.json" bash scripts/run_multi_shot.sh 50 4
   
   # Compare throughput (identical workload)
   ```

2. **Testing hardware changes**
   ```bash
   # Run 1: CPU
   INFERENCE_PERF_TEST_MODE="cache.json" INFERENCE_DEVICE=cpu bash scripts/run_multi_shot.sh 50 1
   
   # Run 2: GPU
   INFERENCE_PERF_TEST_MODE="cache.json" INFERENCE_DEVICE=cuda bash scripts/run_multi_shot.sh 50 1
   
   # Compare speed
   ```

3. **Testing optimization changes**
   ```bash
   # Run 1: Before optimization
   INFERENCE_PERF_TEST_MODE="cache.json" bash scripts/run_multi_shot.sh 50 1
   
   # (Make code changes)
   
   # Run 2: After optimization
   INFERENCE_PERF_TEST_MODE="cache.json" bash scripts/run_multi_shot.sh 50 1
   
   # Compare latency (same LLM overhead)
   ```

**All of these MUST use perf test mode (cached responses)**

## Comparison Table

| Aspect | Accuracy Test | Performance Test |
|--------|---------------|------------------|
| **LLM Responses** | Real, live calls | Real calls + Cached returns |
| **Purpose** | Measure quality | Measure speed |
| **Workload** | Variable (LLM dependent) | Fixed (deterministic) |
| **Flag** | No --perf-test-mode | --perf-test-mode REQUIRED |
| **Run-to-run** | Different responses | Identical responses |
| **Measures** | Precision/Recall/F1/Accuracy | Latency/Throughput |
| **When to use** | Testing models/prompts | Testing system/hardware |
| **Cache needed** | ❌ No | ✅ Yes |
| **Network** | ✅ Required (API calls) | ✅ Required (API calls) |
| **Cost** | ~$0.003/query | ~$0.003/query (same) |

## Commands Summary

### Generate Cache (One-Time)

First, run a normal accuracy test to generate the cache:

```bash
# This creates the cache file for future perf tests
OPENROUTER_API_KEY="sk-or-v1-..." bash scripts/run_multi_shot.sh 50 1

# Output: output_multi_shot_n50_w1_20260528_HHMMSS/llm_logs_multi_shot_20260528_HHMMSS.json
# Save this file path for perf tests
```

### Accuracy Test (Testing Model Quality)

```bash
# Method 1: Using run_accuracy_test.sh
./run_accuracy_test.sh

# Method 2: Direct call
OPENROUTER_API_KEY="sk-or-v1-..." bash scripts/run_multi_shot.sh 50 1

# Method 3: With specific parameters
OPENROUTER_API_KEY="sk-or-v1-..." \
INFERENCE_TEMPERATURE=0.0 \
INFERENCE_MODEL="openai/gpt-4o" \
bash scripts/run_multi_shot.sh 50 1
```

**Key: NO INFERENCE_PERF_TEST_MODE variable!**

### Performance Test (Testing System Speed)

```bash
# Method 1: Using run_perf_test.sh
./run_perf_test.sh

# Method 2: Direct call
INFERENCE_PERF_TEST_MODE="output_dir/llm_logs_multi_shot_*.json" \
OPENROUTER_API_KEY="sk-or-v1-..." \
bash scripts/run_multi_shot.sh 50 1

# Method 3: With numactl
numactl -N 1 -m 1 bash -c "
INFERENCE_PERF_TEST_MODE='output_dir/llm_logs_multi_shot_*.json' \
OPENROUTER_API_KEY='sk-or-v1-...' \
bash scripts/run_multi_shot.sh 50 1
"
```

**Key: MUST have INFERENCE_PERF_TEST_MODE variable!**

## Verification

### How to Verify Accuracy Test (No Cache)

Check the log file:
```bash
# Should NOT find these messages
grep "PERFORMANCE TEST MODE ENABLED" accuracy_test.log
# Result: (no output) ✓

grep "Loading performance test cache" accuracy_test.log
# Result: (no output) ✓

grep "Returning simulated response" accuracy_test.log
# Result: (no output) ✓
```

### How to Verify Performance Test (With Cache)

Check the log file:
```bash
# Should find these messages
grep "PERFORMANCE TEST MODE ENABLED" perf_test.log
# Result: PERFORMANCE TEST MODE ENABLED ✓

grep "Loading performance test cache" perf_test.log
# Result: Loading performance test cache from ... ✓

grep "Returning simulated response" perf_test.log
# Result: [PERF TEST MODE] Returning simulated response ... ✓
```

## Common Mistakes

### Mistake 1: Using Cache for Accuracy Test ❌
```bash
# WRONG - Don't do this!
INFERENCE_PERF_TEST_MODE="cache.json" bash scripts/run_multi_shot.sh 50 1
# Result: Measuring old model accuracy, not current!
```

### Mistake 2: Not Using Cache for Performance Test ❌
```bash
# WRONG - Don't do this!
bash scripts/run_multi_shot.sh 50 1
# Result: Different workloads, can't compare performance!
```

### Mistake 3: Mixing Modes ❌
```bash
# WRONG - Don't compare these!
Run 1: INFERENCE_PERF_TEST_MODE="cache.json" ...  # Perf mode
Run 2: (no perf test mode)                        # Normal mode
# Result: Different workloads, comparison invalid!
```

## Summary

✅ **Accuracy Test Fixed**: Removed `INFERENCE_PERF_TEST_MODE` from `run_accuracy_test.sh`

✅ **Clear Separation**: 
- Accuracy tests: `run_accuracy_test.sh` (no cache)
- Performance tests: `run_perf_test.sh` (with cache)

✅ **Rule of Thumb**:
- Testing **what** the model does → Accuracy test (no cache)
- Testing **how fast** the system does it → Performance test (with cache)

✅ **Key Indicator**:
- See "PERFORMANCE TEST MODE ENABLED" → Using cache
- Don't see it → Real LLM responses

**Always check your logs to ensure you're running the right type of test!**
