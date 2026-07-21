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

# Accuracy test script for E2E-RAG-Datasetup workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Configuration
export WORKSPACE_DIR=${WORKSPACE_DIR:-"/workspace"}
export DOCUMENTS_DIR=${DOCUMENTS_DIR:-"doc_html"}
export DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word}"
export RUN_LOGS=${WORKSPACE_DIR}/run_output_e2e-rag-db/accuracy
export OUTPUT_DIR=${WORKSPACE_DIR}/output_e2e-rag-db/accuracy
export SCENARIO="${SCENARIO:-Offline}"

# Chunking configuration
export CHUNK_SIZE=${CHUNK_SIZE:-768}
export CHUNK_OVERLAP=${CHUNK_OVERLAP:-32}
export TEXT_BOUNDARY=${TEXT_BOUNDARY:-"word"}

# Model paths (use local downloaded models)
export RETRIEVER_MODEL=${RETRIEVER_MODEL:-intfloat_e5-base-v2/e5-base-v2}
export RERANKER_MODEL=${RERANKER_MODEL:-colbert-ir_colbertv2.0/colbertv2.0}

# Device configuration
export DEVICE=${DEVICE:-"auto"}
export NUM_EMBEDDING_DEVICES=${NUM_EMBEDDING_DEVICES:-1}

# Vector database configuration
export VECTOR_INDEX_METHOD=${VECTOR_INDEX_METHOD:-"hnsw"}

# Reference DB manifest for cross-system verification (corpus fingerprint,
# sample-embedding cosine, probe-query top-K ranks). Set to "" to skip.
export MANIFEST=${MANIFEST:-scripts/db_manifest_intel_xpu.json.gz}
export COSINE_THRESHOLD=${COSINE_THRESHOLD:-0.9999}
export TOP_K_DEPTH=${TOP_K_DEPTH:-3}

# Performance options
export BENCHMARK=${BENCHMARK:-false}
export MAX_WORKERS=${MAX_WORKERS:-4}

echo "Configuration:"
echo "  DOCUMENTS_DIR: ${DOCUMENTS_DIR}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  CHUNK_SIZE: ${CHUNK_SIZE}"
echo "  CHUNK_OVERLAP: ${CHUNK_OVERLAP}"
echo "  TEXT_BOUNDARY: ${TEXT_BOUNDARY}"
echo "  RETRIEVER_MODEL: ${RETRIEVER_MODEL}"
echo "  RERANKER_MODEL: ${RERANKER_MODEL}"
echo "  DEVICE: ${DEVICE}"
echo "  NUM_EMBEDDING_DEVICES: ${NUM_EMBEDDING_DEVICES}"
echo "  VECTOR_INDEX_METHOD: ${VECTOR_INDEX_METHOD}"
echo "  MAX_WORKERS: ${MAX_WORKERS}"
echo "  BENCHMARK: ${BENCHMARK}"

# Validate documents directory exists
if [ ! -d "${DOCUMENTS_DIR}" ]; then
    echo "ERROR: Documents directory not found: ${DOCUMENTS_DIR}"
    echo "Please set DOCUMENTS_DIR to point to your HTML documents directory"
    exit 1
fi

# Count HTML files
HTML_COUNT=$(find "${DOCUMENTS_DIR}" -maxdepth 1 -name "*.html" | wc -l)
echo "  HTML files found: ${HTML_COUNT}"

if [ ${HTML_COUNT} -eq 0 ]; then
    echo "ERROR: No HTML files found in ${DOCUMENTS_DIR}"
    exit 1
fi

# Update user.conf to send all HTML files at once
if [ -f "user.conf" ]; then
    # Update max_async_queries to match HTML count (send all at once)
    # Update min_query_count to match HTML count
    if grep -q "e2e-rag-db.Offline.max_async_queries" user.conf; then
        sed -i "s/^e2e-rag-db.Offline.max_async_queries = .*/e2e-rag-db.Offline.max_async_queries = ${HTML_COUNT}/" user.conf
    else
        echo "e2e-rag-db.Offline.max_async_queries = ${HTML_COUNT}" >> user.conf
    fi

    if grep -q "e2e-rag-db.Offline.min_query_count" user.conf; then
        sed -i "s/^e2e-rag-db.Offline.min_query_count = .*/e2e-rag-db.Offline.min_query_count = ${HTML_COUNT}/" user.conf
    else
        echo "e2e-rag-db.Offline.min_query_count = ${HTML_COUNT}" >> user.conf
    fi

    echo "  Loadgen configured to dispatch all ${HTML_COUNT} files at once (accuracy mode)"
    echo "  SUT will process with ${MAX_WORKERS} parallel workers"
fi

# Build benchmark argument
BENCHMARK_ARG=""
if [ "${BENCHMARK}" = "true" ]; then
    BENCHMARK_ARG="--benchmark"
fi

# Run loadgen accuracy test
python3 reference_mlperf_datasetup.py \
    --documents_dir ${DOCUMENTS_DIR} \
    --database ${DATABASE} \
    --scenario ${SCENARIO} \
    --log_dir ${RUN_LOGS} \
    --output_dir ${OUTPUT_DIR} \
    --chunk_size ${CHUNK_SIZE} \
    --chunk_overlap ${CHUNK_OVERLAP} \
    --text_boundary ${TEXT_BOUNDARY} \
    --retriever_model ${RETRIEVER_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --device ${DEVICE} \
    --num_embedding_devices ${NUM_EMBEDDING_DEVICES} \
    --vector_index_method ${VECTOR_INDEX_METHOD} \
    --max_workers ${MAX_WORKERS} \
    --accuracy \
    ${BENCHMARK_ARG}

EXIT_CODE=$?

echo "Time Stop: $(date +%s)"

# If successful, run accuracy evaluation
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Running Accuracy Evaluation"
    echo "============================================================"

    # Build optional manifest argument (skip check if MANIFEST is empty)
    MANIFEST_ARG=""
    if [ -n "${MANIFEST}" ]; then
        MANIFEST_ARG="--manifest ${MANIFEST} --cosine_threshold ${COSINE_THRESHOLD} --top_k_depth ${TOP_K_DEPTH}"
    fi

    python3 datasetup_accuracy_eval.py \
        --log_dir ${RUN_LOGS} \
        --output_dir ${OUTPUT_DIR} \
        --database ${DATABASE}.db \
        --retriever_model ${RETRIEVER_MODEL} \
        ${MANIFEST_ARG}

    EVAL_EXIT_CODE=$?

    if [ ${EVAL_EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Accuracy evaluation failed"
        exit ${EVAL_EXIT_CODE}
    fi
fi

exit ${EXIT_CODE}
