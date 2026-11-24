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

# Accuracy test script for E2E-RAG-QnA workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Configuration
export WORKSPACE_DIR=${WORKSPACE_DIR:-"/workspace"}
export DATA_DIR=${DATA_DIR:-"frames-benchmark-dataset"}
export DATASET_PATH="${DATA_DIR}/frames_dataset.tsv"
export DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word.db}"
export RUN_LOGS=${WORKSPACE_DIR}/run_output_e2e-rag-qna/accuracy
export OUTPUT_DIR=${WORKSPACE_DIR}/output_e2e-rag-qna/accuracy
export SCENARIO="${SCENARIO:-Offline}"

# Threading configuration
export QNA_MAX_WORKERS=${QNA_MAX_WORKERS:-10}

# Accuracy testing - use all queries (or specify subset)

# Use the first N questions, overriding user.conf (0 = user.conf)
export TOTAL_SAMPLE_COUNT=${TOTAL_SAMPLE_COUNT:-0}

# Multi-shot retrieval parameters
export MAX_ITERATIONS=${MAX_ITERATIONS:-5}
export MAX_SUB_QUERIES=${MAX_SUB_QUERIES:-3}
export TOP_K_RETRIEVER=${TOP_K_RETRIEVER:-10}

# Model paths (use local downloaded models)
export EMBEDDING_MODEL=${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}
export RERANKER_MODEL=${RERANKER_MODEL:-colbert-ir_colbertv2.0/colbertv2.0}

# LLM service configuration
# Default to local vLLM server (set OPENROUTER_API_KEY to use OpenRouter instead)
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-sk-or-v1-****}

# Default to local vLLM server
export GRADER_SERVICE_URL=${GRADER_SERVICE_URL:-http://127.0.0.1:8192/v1/chat/completions}
export GRADER_MODEL=${GRADER_MODEL:-gpt-oss-20B/gpt-oss-20b}
export QUERY_MODEL=${QUERY_MODEL:-gpt-oss_model/gpt-oss-120b}

# Query and sufficiency use 120B model on port 8123
export QUERY_SERVICE_URL=${QUERY_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}
export SUFFICIENCY_SERVICE_URL=${SUFFICIENCY_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}
export SUFFICIENCY_MODEL=${SUFFICIENCY_MODEL:-${QUERY_MODEL}}

# Judge LLM configuration (for accuracy evaluation)
export JUDGE_SERVICE_URL=${JUDGE_SERVICE_URL:-http://127.0.0.1:8125/v1/chat/completions}
export JUDGE_MODEL=${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}

echo "  Grader Service URL: ${GRADER_SERVICE_URL}"
echo "  Judge Service URL: ${JUDGE_SERVICE_URL}"
echo "  Judge Model: ${JUDGE_MODEL}"

echo "Configuration:"
echo "  DATASET_PATH: ${DATASET_PATH}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
echo "  QNA_MAX_WORKERS: ${QNA_MAX_WORKERS} (SUT thread pool)"
echo "  MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "  MAX_SUB_QUERIES: ${MAX_SUB_QUERIES}"
echo "  EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
echo "  RERANKER_MODEL: ${RERANKER_MODEL}"

# Run loadgen accuracy test
python3 reference_mlperf.py \
    --dataset_path ${DATASET_PATH} \
    --database ${DATABASE} \
    --scenario ${SCENARIO} \
    --total_sample_count ${TOTAL_SAMPLE_COUNT} \
    --log_dir ${RUN_LOGS} \
    --output_dir ${OUTPUT_DIR} \
    --max-iterations ${MAX_ITERATIONS} \
    --max-sub-queries ${MAX_SUB_QUERIES} \
    --top_k_retriever ${TOP_K_RETRIEVER} \
    --max_workers ${QNA_MAX_WORKERS} \
    --embedding_model ${EMBEDDING_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --grader_service_url ${GRADER_SERVICE_URL} \
    --grader_model ${GRADER_MODEL} \
    --query_model ${QUERY_MODEL} \
    --query_service_url ${QUERY_SERVICE_URL} \
    --sufficiency_service_url ${SUFFICIENCY_SERVICE_URL} \
    --sufficiency_model ${SUFFICIENCY_MODEL} \
    --judge_service_url ${JUDGE_SERVICE_URL} \
    --judge_model ${JUDGE_MODEL} \
    --accuracy

EXIT_CODE=$?

echo "Time Stop: $(date +%s)"
exit ${EXIT_CODE}
