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

# Datasetup performance test script for E2E-RAG-Datasetup workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Run from e2e-rag/ (paths below are relative to it).
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

# Configuration
export RUN_LOGS=${WORKSPACE_DIR}/run_output_e2e-rag-db/performance
export OUTPUT_DIR=${WORKSPACE_DIR}/output_e2e-rag-db/performance

echo "Configuration:"
echo "  INGESTION_DOC_DIR: ${INGESTION_DOC_DIR}"
echo "  INGESTION_DB: ${INGESTION_DB}"
echo "  SCENARIO: ${SCENARIO}"
echo "  INGESTION_CHUNK_SIZE: ${INGESTION_CHUNK_SIZE}"
echo "  INGESTION_CHUNK_OVERLAP: ${INGESTION_CHUNK_OVERLAP}"
echo "  INGESTION_TEXT_BOUNDARY: ${INGESTION_TEXT_BOUNDARY}"
echo "  EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
echo "  RERANKER_MODEL: ${RERANKER_MODEL}"
echo "  INGESTION_DEVICE: ${INGESTION_DEVICE}"
echo "  INGESTION_NUM_EMBEDDING_DEVICES: ${INGESTION_NUM_EMBEDDING_DEVICES}"
echo "  INGESTION_VECTOR_INDEX_METHOD: ${INGESTION_VECTOR_INDEX_METHOD}"
echo "  INGESTION_MAX_WORKERS: ${INGESTION_MAX_WORKERS}"
echo "  INGESTION_BENCHMARK: ${INGESTION_BENCHMARK}"

# Validate documents directory exists
if [ ! -d "${INGESTION_DOC_DIR}" ]; then
    echo "ERROR: Documents directory not found: ${INGESTION_DOC_DIR}"
    echo "Please set INGESTION_DOC_DIR to point to your HTML documents directory"
    exit 1
fi

# Count HTML files
HTML_COUNT=$(find "${INGESTION_DOC_DIR}" -maxdepth 1 -name "*.html" | wc -l)
echo "  HTML files found: ${HTML_COUNT}"

if [ ${HTML_COUNT} -eq 0 ]; then
    echo "ERROR: No HTML files found in ${INGESTION_DOC_DIR}"
    exit 1
fi

echo "  LoadGen dispatches all ${HTML_COUNT} files at once; SUT uses ${INGESTION_MAX_WORKERS} workers"

# Build benchmark argument
BENCHMARK_ARG=""
if [ "${INGESTION_BENCHMARK}" = "true" ]; then
    BENCHMARK_ARG="--benchmark"
fi

# Run loadgen ingestion test
python3 main_ingestion.py \
    --documents_dir ${INGESTION_DOC_DIR} \
    --database ${INGESTION_DB} \
    --scenario ${SCENARIO} \
    --log_dir ${RUN_LOGS} \
    --output_dir ${OUTPUT_DIR} \
    --chunk_size ${INGESTION_CHUNK_SIZE} \
    --chunk_overlap ${INGESTION_CHUNK_OVERLAP} \
    --text_boundary ${INGESTION_TEXT_BOUNDARY} \
    --embedding_model ${EMBEDDING_MODEL} \
    --reranker_model ${RERANKER_MODEL} \
    --device ${INGESTION_DEVICE} \
    --num_embedding_devices ${INGESTION_NUM_EMBEDDING_DEVICES} \
    --vector_index_method ${INGESTION_VECTOR_INDEX_METHOD} \
    --max_workers ${INGESTION_MAX_WORKERS} \
    ${BENCHMARK_ARG}

EXIT_CODE=$?

echo "Time Stop: $(date +%s)"
exit ${EXIT_CODE}
