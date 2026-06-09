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

# Performance test script for E2E DocGrader workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Configuration
export WORKSPACE_DIR=${WORKSPACE_DIR:-"/workspace"}
export DATA_DIR=${DATA_DIR:-"data"}
export DATASET_PATH="${DATA_DIR}/frames_dataset.tsv"
export DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word.db}"
export RUN_LOGS=${WORKSPACE_DIR}/run_output
export OUTPUT_DIR=${WORKSPACE_DIR}/output
export SCENARIO="${SCENARIO:-Offline}"

# Performance testing - limit queries
export PERF_COUNT=${PERF_COUNT:-824}

# Multi-shot retrieval parameters
export MAX_ITERATIONS=${MAX_ITERATIONS:-5}
export MAX_SUB_QUERIES=${MAX_SUB_QUERIES:-3}
export TOP_K_RETRIEVER=${TOP_K_RETRIEVER:-10}

# Model paths (use local cached models)
export RETRIEVER_MODEL=${RETRIEVER_MODEL:-/data/model/e5-base-v2}
export RERANKER_MODEL=${RERANKER_MODEL:-/data/model/colbertv2.0}

# LLM service configuration
# Default to local vLLM server (set OPENROUTER_API_KEY to use OpenRouter instead)
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-sk-or-v1-****}

# Separate service endpoints for each LLM component
export LLM_SERVICE_URL=${LLM_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}
export LLM_MODEL=${LLM_MODEL:-gpt-oss-20b}

export QUERY_SERVICE_URL=${QUERY_SERVICE_URL:-http://127.0.0.1:8124/v1/chat/completions}
export QUERY_MODEL=${QUERY_MODEL:-gpt-oss-120b}

export JUDGE_SERVICE_URL=${JUDGE_SERVICE_URL:-http://127.0.0.1:8125/v1/chat/completions}
export JUDGE_MODEL=${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}

echo "Configuration:"
echo "  DATASET_PATH: ${DATASET_PATH}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  PERF_COUNT: ${PERF_COUNT}"
echo "  MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "  MAX_SUB_QUERIES: ${MAX_SUB_QUERIES}"
echo "  RETRIEVER_MODEL: ${RETRIEVER_MODEL}"
echo "  RERANKER_MODEL: ${RERANKER_MODEL}"
echo "  LLM_SERVICE_URL: ${LLM_SERVICE_URL}"
echo "  LLM_MODEL: ${LLM_MODEL}"
echo "  QUERY_SERVICE_URL: ${QUERY_SERVICE_URL}"
echo "  QUERY_MODEL: ${QUERY_MODEL}"
echo "  JUDGE_SERVICE_URL: ${JUDGE_SERVICE_URL}"
echo "  JUDGE_MODEL: ${JUDGE_MODEL}"

# Run loadgen performance test
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
    --retriever_model ${RETRIEVER_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --llm_service_url ${LLM_SERVICE_URL} \
    --llm_model ${LLM_MODEL} \
    --query_service_url ${QUERY_SERVICE_URL} \
    --query_model ${QUERY_MODEL} \
    --judge_service_url ${JUDGE_SERVICE_URL} \
    --judge_model ${JUDGE_MODEL}

echo "Time Stop: $(date +%s)"
