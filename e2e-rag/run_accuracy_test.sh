#!/bin/bash
# Accuracy test - DO NOT use perf test mode (need real LLM responses)
# Performance test mode should ONLY be used for performance benchmarking, not accuracy testing

echo "Running ACCURACY test (NOT performance test mode)"
echo "This will use real LLM calls to measure actual model accuracy"
echo ""

# DO NOT SET INFERENCE_PERF_TEST_MODE for accuracy tests!
# Only set API key
OPENROUTER_API_KEY=sk-or-v1-******** bash scripts/run_multi_shot.sh 1 1

echo ""
echo "=== Accuracy Test Complete ==="
echo "This was a real accuracy test with live LLM calls"
