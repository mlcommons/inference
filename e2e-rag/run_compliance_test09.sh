#!/bin/bash
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# TEST09 Compliance Test Runner for E2E DocGrader Workload
# Automates: setup -> run -> verify -> cleanup workflow

set -e  # Exit on error

echo "=============================================================================="
echo "TEST09 Compliance Test for E2E-RAG Workload"
echo "=============================================================================="
echo "Start time: $(date)"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLIANCE_DIR="${SCRIPT_DIR}/../compliance/TEST09/e2e-rag"
AUDIT_CONFIG="${COMPLIANCE_DIR}/audit.config"
WORKING_AUDIT_CONFIG="${SCRIPT_DIR}/audit.config"
TEST09_VERIFICATION="${SCRIPT_DIR}/third_party/mlperf-inference/compliance/TEST09/run_verification.py"

# Directories
export WORKSPACE_DIR=${WORKSPACE_DIR:-"${SCRIPT_DIR}"}
export DATA_DIR=${DATA_DIR:-"frames-benchmark-dataset"}
export DATASET_PATH="${DATA_DIR}/frames_dataset.tsv"
export DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word.db}"
export RUN_LOGS=${WORKSPACE_DIR}/run_output_test09
export OUTPUT_DIR=${WORKSPACE_DIR}/output_test09
export SUBMISSION_DIR=${WORKSPACE_DIR}/submission/compliance/e2e-rag/Offline
export SCENARIO="${SCENARIO:-Offline}"

# Performance testing - full dataset for compliance
export PERF_COUNT=824

# Threading configuration
export MAX_ASYNC_QUERIES=${MAX_ASYNC_QUERIES:-10}
export MAX_WORKERS=${MAX_WORKERS:-10}

# Multi-shot retrieval parameters
export MAX_ITERATIONS=${MAX_ITERATIONS:-5}
export MAX_SUB_QUERIES=${MAX_SUB_QUERIES:-3}
export TOP_K_RETRIEVER=${TOP_K_RETRIEVER:-10}

# Model paths
export RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat_e5-base-v2/e5-base-v2}
export RERANKER_MODEL=${RERANKER_MODEL:-colbert-ir_colbertv2.0/colbertv2.0}

# LLM service configuration
export LLM_SERVICE_URL=${LLM_SERVICE_URL:-http://127.0.0.1:8192/v1/chat/completions}
export LLM_MODEL=${LLM_MODEL:-gpt-oss-20b-mxfp4}
export QUERY_SERVICE_URL=${QUERY_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}
export QUERY_MODEL=${QUERY_MODEL:-gpt-oss-120b-mxfp4}
export SUFFICIENCY_SERVICE_URL=${SUFFICIENCY_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}
export SUFFICIENCY_MODEL=${SUFFICIENCY_MODEL:-gpt-oss-120b-mxfp4}
export JUDGE_SERVICE_URL=${JUDGE_SERVICE_URL:-http://127.0.0.1:8192/v1/chat/completions}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-oss-20b-mxfp4}

# Performance cache file (optional - for faster testing)
export PERF_CACHE_FILE=${PERF_CACHE_FILE:-""}

echo "Configuration:"
echo "  DATASET_PATH: ${DATASET_PATH}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  PERF_COUNT: ${PERF_COUNT}"
echo "  RUN_LOGS: ${RUN_LOGS}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  SUBMISSION_DIR: ${SUBMISSION_DIR}"
echo "  MAX_ASYNC_QUERIES: ${MAX_ASYNC_QUERIES}"
echo "  MAX_WORKERS: ${MAX_WORKERS}"
echo ""

# ============================================================================
# Part I: Setup
# ============================================================================
echo "=============================================================================="
echo "PART I: Setup"
echo "=============================================================================="

# Verify audit.config exists
if [ ! -f "${AUDIT_CONFIG}" ]; then
    echo "ERROR: audit.config not found at ${AUDIT_CONFIG}"
    echo "Please ensure compliance configuration is set up."
    exit 1
fi

# Verify verification script exists
if [ ! -f "${TEST09_VERIFICATION}" ]; then
    echo "ERROR: run_verification.py not found at ${TEST09_VERIFICATION}"
    exit 1
fi

# Create directories
mkdir -p "${RUN_LOGS}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SUBMISSION_DIR}"

# Copy audit.config to working directory
echo "Copying audit.config to working directory..."
cp "${AUDIT_CONFIG}" "${WORKING_AUDIT_CONFIG}"
echo "✓ audit.config copied to ${WORKING_AUDIT_CONFIG}"
echo ""

# ============================================================================
# Part II: Run Performance Test with Compliance Logging
# ============================================================================
echo "=============================================================================="
echo "PART II: Run Performance Test"
echo "=============================================================================="
echo "Running MLPerf LoadGen with TEST09 compliance logging..."
echo ""

# Build perf cache argument if file exists
PERF_CACHE_ARG=""
if [ -n "${PERF_CACHE_FILE}" ] && [ -f "${PERF_CACHE_FILE}" ]; then
    PERF_CACHE_ARG="--perf-test-mode ${PERF_CACHE_FILE}"
    echo "Using cached LLM responses from: ${PERF_CACHE_FILE}"
fi

# Run loadgen performance test
# Note: LoadGen automatically detects audit.config in the current directory
python3 reference_mlperf.py \
    --dataset_path ${DATASET_PATH} \
    --database ${DATABASE} \
    --scenario ${SCENARIO} \
    --log_dir ${RUN_LOGS} \
    --output_dir ${OUTPUT_DIR} \
    --perf_count ${PERF_COUNT} \
    --max-iterations ${MAX_ITERATIONS} \
    --max-sub-queries ${MAX_SUB_QUERIES} \
    --top_k_retriever ${TOP_K_RETRIEVER} \
    --max_workers ${MAX_WORKERS} \
    --retriever_model ${RETRIEVER_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --llm_service_url ${LLM_SERVICE_URL} \
    --llm_model ${LLM_MODEL} \
    --query_service_url ${QUERY_SERVICE_URL} \
    --query_model ${QUERY_MODEL} \
    --sufficiency-service-url ${SUFFICIENCY_SERVICE_URL} \
    --sufficiency-model ${SUFFICIENCY_MODEL} \
    --judge_service_url ${JUDGE_SERVICE_URL} \
    --judge_model ${JUDGE_MODEL} \
    ${PERF_CACHE_ARG}

TEST_EXIT_CODE=$?

if [ ${TEST_EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Performance test failed with exit code ${TEST_EXIT_CODE}"
    echo "Cleaning up audit.config..."
    rm -f "${WORKING_AUDIT_CONFIG}"
    exit ${TEST_EXIT_CODE}
fi

echo ""
echo "✓ Performance test completed successfully"
echo ""

# ============================================================================
# Part III: Verify Compliance
# ============================================================================
echo "=============================================================================="
echo "PART III: Verify Compliance"
echo "=============================================================================="

# Check if accuracy log exists
if [ ! -f "${RUN_LOGS}/mlperf_log_accuracy.json" ]; then
    echo "ERROR: mlperf_log_accuracy.json not found in ${RUN_LOGS}"
    echo "Compliance verification requires accuracy log."
    rm -f "${WORKING_AUDIT_CONFIG}"
    exit 1
fi

echo "Running TEST09 verification..."
echo ""

python3 "${TEST09_VERIFICATION}" \
    -c "${RUN_LOGS}" \
    -o "${SUBMISSION_DIR}/.." \
    --audit-config "${AUDIT_CONFIG}"

VERIFY_EXIT_CODE=$?

echo ""
if [ ${VERIFY_EXIT_CODE} -eq 0 ]; then
    echo "✓ TEST09 verification PASSED"
else
    echo "✗ TEST09 verification FAILED"
fi
echo ""

# ============================================================================
# Part IV: Cleanup
# ============================================================================
echo "=============================================================================="
echo "PART IV: Cleanup"
echo "=============================================================================="

echo "Removing audit.config from working directory..."
rm -f "${WORKING_AUDIT_CONFIG}"
echo "✓ audit.config removed"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=============================================================================="
echo "TEST09 Compliance Test Summary"
echo "=============================================================================="
echo "End time: $(date)"
echo ""
echo "Logs saved to:"
echo "  Run logs:        ${RUN_LOGS}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Submission:      ${SUBMISSION_DIR}"
echo ""
echo "Submission artifacts (to be uploaded):"
echo "  ${SUBMISSION_DIR}/verify_output_len.txt"
echo "  ${SUBMISSION_DIR}/accuracy/mlperf_log_accuracy.json"
echo "  ${SUBMISSION_DIR}/performance/run_1/mlperf_log_summary.txt"
echo "  ${SUBMISSION_DIR}/performance/run_1/mlperf_log_detail.txt"
echo ""

if [ ${VERIFY_EXIT_CODE} -eq 0 ]; then
    echo "Status: ✓ COMPLIANCE TEST PASSED"
    echo "=============================================================================="
    exit 0
else
    echo "Status: ✗ COMPLIANCE TEST FAILED"
    echo "=============================================================================="
    exit 1
fi
