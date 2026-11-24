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

# Performance test script for E2E-RAG-QnA workload with MLPerf Loadgen

echo "Time Start: $(date +%s)"

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

# Configuration
export RUN_LOGS=${WORKSPACE_DIR}/run_output_e2e-rag-qna/performance
export OUTPUT_DIR=${WORKSPACE_DIR}/output_e2e-rag-qna/performance

echo "Configuration:"
echo "  DATASET_PATH: ${DATASET_PATH}"
echo "  DATABASE: ${DATABASE}"
echo "  SCENARIO: ${SCENARIO}"
echo "  TOTAL_SAMPLE_COUNT: ${TOTAL_SAMPLE_COUNT}"
echo "  PERF_CACHE_FILE: ${PERF_CACHE_FILE}"
echo "  QNA_MAX_WORKERS: ${QNA_MAX_WORKERS} (SUT thread pool)"
echo "  MAX_ITERATIONS: ${MAX_ITERATIONS}"
echo "  MAX_SUB_QUERIES: ${MAX_SUB_QUERIES}"
echo "  EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
echo "  RERANKER_MODEL: ${RERANKER_MODEL}"
echo "  GRADER_SERVICE_URL: ${GRADER_SERVICE_URL}"
echo "  GRADER_MODEL: ${GRADER_MODEL}"
echo "  QUERY_SERVICE_URL: ${QUERY_SERVICE_URL}"
echo "  QUERY_MODEL: ${QUERY_MODEL}"
echo "  JUDGE_SERVICE_URL: ${JUDGE_SERVICE_URL}"
echo "  JUDGE_MODEL: ${JUDGE_MODEL}"

# A performance run replays the recorded LLM responses; without them the
# trajectory is not the reference one and the run is not valid.
if [ ! -f "${PERF_CACHE_FILE}" ]; then
    echo "ERROR: PERF_CACHE_FILE not found: ${PERF_CACHE_FILE}" >&2
    exit 1
fi

# Run loadgen performance test
python3 reference_mlperf.py \
    --dataset_path ${DATASET_PATH} \
    --database ${DATABASE} \
    --scenario ${SCENARIO} \
    --device ${QNA_DEVICE} \
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
    --query_service_url ${QUERY_SERVICE_URL} \
    --query_model ${QUERY_MODEL} \
    --sufficiency_service_url ${SUFFICIENCY_SERVICE_URL} \
    --sufficiency_model ${SUFFICIENCY_MODEL} \
    --judge_service_url ${JUDGE_SERVICE_URL} \
    --judge_model ${JUDGE_MODEL} \
    --perf-test-mode ${PERF_CACHE_FILE}

EXIT_CODE=$?

echo "Time Stop: $(date +%s)"
exit ${EXIT_CODE}
