#!/bin/bash
# =============================================================================
# Multi-shot retrieval experiment
#
# Usage (from repo root):
#   bash scripts/run_multi_shot.sh [N_QUERIES] [NUM_WORKERS]
#   MODE=perf bash scripts/run_multi_shot.sh ...   # replay PERF_CACHE_FILE
#
#   N_QUERIES, NUM_WORKERS: optional positional overrides for N_QUERIES
#                           and QNA_MAX_WORKERS in config.sh.
#                           Use 'all' for the full dataset (824 queries).
#
# Configuration:
#   config.default.sh; override with config.sh, or one-off via env var,
#   e.g.:  QNA_DEVICE=cpu bash scripts/run_multi_shot.sh 50
#
# Prerequisites:
#   - Local vLLM servers running: 20B on port 8192, 120B on port 8123 (default)
#   - OR set OPENROUTER_API_KEY environment variable to use OpenRouter
#   - scripts/run_ingestion.sh has been run (vector DB exists)
#
# =============================================================================

set -e

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

# Architecture:
# - Document grader: GRADER_MODEL via GRADER_SERVICE_URL
# - Query generator: QUERY_MODEL via QUERY_SERVICE_URL
# - Sufficiency checker + final answer: SUFFICIENCY_MODEL via SUFFICIENCY_SERVICE_URL
# - Embeddings / reranking: device controlled by QNA_EMBEDDING_DEVICE / RERANKER_DEVICE

TEMPERATURE="${TEMPERATURE:-1.0}"
REASONING="${REASONING:-medium}"
MAX_RETRIES="${MAX_RETRIES:-5}"
N_QUERIES="${N_QUERIES:-5}"

# Empty -> not passed; the Python defaults apply.

# accuracy: live LLM responses. perf: replay PERF_CACHE_FILE (must exist).
MODE="${MODE:-accuracy}"
case "${MODE}" in
    accuracy) REPLAY_ARGS=() ;;
    perf)
        [[ -f "${PERF_CACHE_FILE}" ]] || { echo "ERROR: MODE=perf but PERF_CACHE_FILE not found: ${PERF_CACHE_FILE}" >&2; exit 1; }
        REPLAY_ARGS=(--perf-test-mode "${PERF_CACHE_FILE}") ;;
    *) echo "ERROR: MODE must be accuracy or perf (got '${MODE}')" >&2; exit 1 ;;
esac

# Positional args override config.
N_QUERIES="${1:-${N_QUERIES}}"
NUM_WORKERS="${2:-${QNA_MAX_WORKERS}}"

if [[ "${N_QUERIES}" == "all" ]]; then
    EVAL_FLAG="--eval"
    TAG="full"
else
    EVAL_FLAG="--eval ${N_QUERIES}"
    TAG="n${N_QUERIES}"
fi

OUTPUT_DIR="output_multi_shot_${TAG}_w${NUM_WORKERS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

RESULT_JSON="${OUTPUT_DIR}/result_multi_shot_${TAG}.json"
LOG_FILE="${OUTPUT_DIR}/run.log"
SCORE_FILE="${OUTPUT_DIR}/score_multi_shot_${TAG}.txt"

echo "=== Multi-shot retrieval ==="
echo "  Model (grader):    ${GRADER_MODEL}"
echo "  Model (query gen): ${QUERY_MODEL}"
echo "  DB:          ${DATABASE}"
echo "  Workers:     ${NUM_WORKERS}"
echo "  Queries:     ${N_QUERIES}"
echo "  Device:      ${QNA_DEVICE} (embedding=${QNA_EMBEDDING_DEVICE}, reranker=${RERANKER_DEVICE})"
echo "  Temperature: ${TEMPERATURE}  Reasoning: ${REASONING}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

python3 -u multi_shot_retrieval.py \
    --db "${DATABASE}" \
    ${EVAL_FLAG} \
    --max-iterations "${MAX_ITERATIONS}" \
    --max-sub-queries "${MAX_SUB_QUERIES}" \
    --device "${QNA_DEVICE}" \
    --embedding-device "${QNA_EMBEDDING_DEVICE}" \
    --reranker-device "${RERANKER_DEVICE}" \
    --retrieval_strategy fixed_k \
    --embedding_model "${EMBEDDING_MODEL}" \
    --top_k_retriever "${TOP_K_RETRIEVER}" \
    --generate-answer \
    --num-workers "${NUM_WORKERS}" \
    --temperature "${TEMPERATURE}" \
    --reasoning "${REASONING}" \
    --max-retries "${MAX_RETRIES}" \
    --output-dir "${OUTPUT_DIR}" \
    --grader_model "${GRADER_MODEL}" \
    --query_model "${QUERY_MODEL}" \
    --grader_service_url "${GRADER_SERVICE_URL}" \
    --query_service_url "${QUERY_SERVICE_URL}" \
    --sufficiency_service_url "${SUFFICIENCY_SERVICE_URL}" \
    ${SUFFICIENCY_MODEL:+--sufficiency_model "${SUFFICIENCY_MODEL}"} \
    "${REPLAY_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

if [[ -f "${OUTPUT_DIR}/result_multi_shot.json" ]]; then
    mv "${OUTPUT_DIR}/result_multi_shot.json" "${RESULT_JSON}"
    echo "Saved results to ${RESULT_JSON}"
fi

echo ""
echo "=== Scoring with LLM judge ==="
python3 -u evaluate.py "${RESULT_JSON}" \
    --dataset "${DATASET_PATH}" \
    --judge_service_url "${JUDGE_SERVICE_URL}" \
    --judge_model "${JUDGE_MODEL}" \
    --batch-size 4

echo ""
echo "=== Done ==="
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Results:     ${RESULT_JSON}"
echo "  Score:       ${SCORE_FILE}"
echo "  Run log:     ${LOG_FILE}"
