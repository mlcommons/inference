#!/bin/bash
# =============================================================================
# Multi-shot retrieval experiment
#
# Usage (from repo root):
#   bash scripts/run_multi_shot.sh [N_QUERIES] [NUM_WORKERS]
#
#   N_QUERIES, NUM_WORKERS: optional positional overrides for N_QUERIES
#                           and QNA_MAX_WORKERS in config.sh.
#                           Use 'all' for the full dataset (824 queries).
#
# Configuration:
#   See config.template.sh. Override with config.sh, or one-off via env var,
#   e.g.:  QNA_DEVICE=cpu bash scripts/run_multi_shot.sh 50
#
# Prerequisites:
#   - Local vLLM servers running: 20B on port 8192, 120B on port 8123 (default)
#   - OR set OPENROUTER_API_KEY environment variable to use OpenRouter
#   - scripts/run_ingestion.sh has been run (vector DB exists)
#
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

# Architecture:
# - Document grader: GRADER_MODEL via GRADER_SERVICE_URL
# - Query generator: QUERY_MODEL via QUERY_SERVICE_URL
# - Sufficiency checker + final answer: SUFFICIENCY_MODEL via SUFFICIENCY_SERVICE_URL
# - Embeddings / reranking: device controlled by QNA_EMBEDDING_DEVICE / RERANKER_DEVICE

QNA_DEVICE="${QNA_DEVICE:-cpu}"
QNA_EMBEDDING_DEVICE="${QNA_EMBEDDING_DEVICE:-${QNA_DEVICE}}"
RERANKER_DEVICE="${RERANKER_DEVICE:-${QNA_DEVICE}}"
DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
TOP_K_RETRIEVER="${TOP_K_RETRIEVER:-15}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
MAX_SUB_QUERIES="${MAX_SUB_QUERIES:-3}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REASONING="${REASONING:-medium}"
MAX_RETRIES="${MAX_RETRIES:-5}"
N_QUERIES="${N_QUERIES:-5}"
QNA_MAX_WORKERS="${QNA_MAX_WORKERS:-1}"
GRADER_SERVICE_URL="${GRADER_SERVICE_URL:-http://127.0.0.1:8192/v1/chat/completions}"
GRADER_MODEL="${GRADER_MODEL:-gpt-oss-20b-mxfp4}"
QUERY_MODEL="${QUERY_MODEL:-gpt-oss-120b-mxfp4}"

# Empty -> not passed; the Python defaults apply.
QUERY_SERVICE_URL="${QUERY_SERVICE_URL:-}"
SUFFICIENCY_SERVICE_URL="${SUFFICIENCY_SERVICE_URL:-}"
SUFFICIENCY_MODEL="${SUFFICIENCY_MODEL:-gpt-oss-120b-mxfp4}"
JUDGE_SERVICE_URL="${JUDGE_SERVICE_URL:-http://127.0.0.1:8125/v1/chat/completions}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PERF_CACHE_FILE="${PERF_CACHE_FILE:-}"

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
    ${QUERY_SERVICE_URL:+--query_service_url "${QUERY_SERVICE_URL}"} \
    ${SUFFICIENCY_SERVICE_URL:+--sufficiency_service_url "${SUFFICIENCY_SERVICE_URL}"} \
    ${SUFFICIENCY_MODEL:+--sufficiency_model "${SUFFICIENCY_MODEL}"} \
    ${PERF_CACHE_FILE:+--perf-test-mode "${PERF_CACHE_FILE}"} \
    2>&1 | tee "${LOG_FILE}"

if [[ -f "${OUTPUT_DIR}/result_multi_shot.json" ]]; then
    mv "${OUTPUT_DIR}/result_multi_shot.json" "${RESULT_JSON}"
    echo "Saved results to ${RESULT_JSON}"
fi

echo ""
echo "=== Scoring with LLM judge ==="
python3 -u evaluate.py "${RESULT_JSON}" \
    --dataset "${DATASET:-data/frames_dataset.tsv}" \
    --judge_service_url "${JUDGE_SERVICE_URL}" \
    --judge_model "${JUDGE_MODEL}" \
    --batch-size 4

echo ""
echo "=== Done ==="
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Results:     ${RESULT_JSON}"
echo "  Score:       ${SCORE_FILE}"
echo "  Run log:     ${LOG_FILE}"
