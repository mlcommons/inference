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

# Accuracy test script for E2E DocGrader workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Configuration
export WORKSPACE_DIR=${WORKSPACE_DIR:-"/workspace"}
export DATA_DIR=${DATA_DIR:-"data"}
export DATASET_PATH="${DATA_DIR}/frames_dataset.tsv"
export DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word.db}"
export RUN_LOGS=${WORKSPACE_DIR}/run_output
export OUTPUT_DIR=${WORKSPACE_DIR}/output
export SCENARIO="${SCENARIO:-Offline}"

# Threading configuration
export MAX_ASYNC_QUERIES=${MAX_ASYNC_QUERIES:-10}
export MAX_WORKERS=${MAX_WORKERS:-10}

# Accuracy testing - use all queries (or specify subset)
export PERF_COUNT=${PERF_COUNT:-824}  # Empty = all queries

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

# Default to local vLLM server
export LLM_SERVICE_URL=${LLM_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}
export LLM_MODEL=${LLM_MODEL:-gpt-oss-20b}
export QUERY_MODEL=${QUERY_MODEL:-gpt-oss-120b}
echo "  LLM Service URL: ${LLM_SERVICE_URL}"

echo "Configuration:"
echo "  DATASET_PATH: ${DATASET_PATH}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  PERF_COUNT: ${PERF_COUNT:-all queries}"
echo "  MAX_ASYNC_QUERIES: ${MAX_ASYNC_QUERIES} (loadgen query dispatch)"
echo "  MAX_WORKERS: ${MAX_WORKERS} (SUT thread pool)"
echo "  MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "  MAX_SUB_QUERIES: ${MAX_SUB_QUERIES}"
echo "  RETRIEVER_MODEL: ${RETRIEVER_MODEL}"
echo "  RERANKER_MODEL: ${RERANKER_MODEL}"

# Build perf_count argument
PERF_COUNT_ARG=""
if [ -n "${PERF_COUNT}" ]; then
    PERF_COUNT_ARG="--perf_count ${PERF_COUNT}"
fi

# Update user.conf with threading configuration
sed -i "s/^e2e.Offline.max_async_queries = .*/e2e.Offline.max_async_queries = ${MAX_ASYNC_QUERIES}/" user.conf

# Run loadgen accuracy test
python3 reference_mlperf.py \
    --dataset_path ${DATASET_PATH} \
    --database ${DATABASE} \
    --scenario ${SCENARIO} \
    --log_dir ${RUN_LOGS} \
    --output_dir ${OUTPUT_DIR} \
    ${PERF_COUNT_ARG} \
    --max-iterations ${MAX_ITERATIONS} \
    --max-sub-queries ${MAX_SUB_QUERIES} \
    --top_k_retriever ${TOP_K_RETRIEVER} \
    --max_workers ${MAX_WORKERS} \
    --retriever_model ${RETRIEVER_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --llm_service_url ${LLM_SERVICE_URL} \
    --llm_model ${LLM_MODEL} \
    --query_model ${QUERY_MODEL} \
    --accuracy

echo "Time Stop: $(date +%s)"
