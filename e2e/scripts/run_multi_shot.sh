#!/bin/bash
# =============================================================================
# Multi-shot retrieval experiment
#
# Usage (from repo root):
#   bash scripts/run_multi_shot.sh [N_QUERIES] [NUM_WORKERS]
#
#   N_QUERIES, NUM_WORKERS: optional positional overrides for INFERENCE_N_QUERIES
#                           and INFERENCE_NUM_WORKERS in config.sh.
#                           Use 'all' for the full dataset (824 queries).
#
# Configuration:
#   See config.template.sh. Override with config.sh, or one-off via env var,
#   e.g.:  INFERENCE_DEVICE=cpu bash scripts/run_multi_shot.sh 50
#
# Prerequisites:
#   - OPENROUTER_API_KEY environment variable set
#   - scripts/run_ingestion.sh has been run (vector DB exists)
#
# For strict memory binding, invoke under numactl:
#   numactl --membind=0 bash scripts/run_multi_shot.sh ...
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable not set"
    echo "Usage: OPENROUTER_API_KEY=\"sk-or-v1-YOUR_KEY_HERE\" bash $0"
    exit 1
fi

# Architecture:
# - Document grader: INFERENCE_MODEL via INFERENCE_LLM_URL
# - Sufficiency checker / query generator / answer generator: INFERENCE_QUERY_MODEL via INFERENCE_LLM_URL
# - Embeddings / reranking: device controlled by INFERENCE_EMBEDDING_DEVICE / INFERENCE_RERANKER_DEVICE

INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}"
INFERENCE_EMBEDDING_DEVICE="${INFERENCE_EMBEDDING_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_RERANKER_DEVICE="${INFERENCE_RERANKER_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_DB="${INFERENCE_DB:-vector_html_hnsw_len768_ov32_word}"
INFERENCE_RETRIEVER_MODEL="${INFERENCE_RETRIEVER_MODEL:-/data/model/e5-base-v2}"
INFERENCE_TOP_K_RETRIEVER="${INFERENCE_TOP_K_RETRIEVER:-15}"
INFERENCE_MAX_ITERATIONS="${INFERENCE_MAX_ITERATIONS:-5}"
INFERENCE_MAX_SUB_QUERIES="${INFERENCE_MAX_SUB_QUERIES:-3}"
INFERENCE_TEMPERATURE="${INFERENCE_TEMPERATURE:-1.0}"
INFERENCE_MAX_RETRIES="${INFERENCE_MAX_RETRIES:-5}"
INFERENCE_N_QUERIES="${INFERENCE_N_QUERIES:-5}"
INFERENCE_NUM_WORKERS="${INFERENCE_NUM_WORKERS:-1}"
INFERENCE_LLM_URL="${INFERENCE_LLM_URL:-http://127.0.0.1:8123/v1/chat/completions}"
INFERENCE_MODEL="${INFERENCE_MODEL:-/model/gpt-oss-20b-mxfp4}"
INFERENCE_QUERY_MODEL="${INFERENCE_QUERY_MODEL:-/model/gpt-oss-120b-mxfp4}"

# Per-component endpoint splits. Empty -> inherit INFERENCE_LLM_URL / INFERENCE_MODEL.
INFERENCE_GRADER_URL="${INFERENCE_GRADER_URL:-}"
INFERENCE_GRADER_MODEL="${INFERENCE_GRADER_MODEL:-}"
INFERENCE_QUERY_URL="${INFERENCE_QUERY_URL:-}"
INFERENCE_SUFFICIENCY_URL="${INFERENCE_SUFFICIENCY_URL:-}"
INFERENCE_SUFFICIENCY_MODEL="${INFERENCE_SUFFICIENCY_MODEL:-}"
INFERENCE_JUDGE_URL="${INFERENCE_JUDGE_URL:-https://openrouter.ai/api/v1/chat/completions}"
INFERENCE_JUDGE_MODEL="${INFERENCE_JUDGE_MODEL:-openai/gpt-oss-20b}"

# Positional args override config.
N_QUERIES="${1:-${INFERENCE_N_QUERIES}}"
NUM_WORKERS="${2:-${INFERENCE_NUM_WORKERS}}"

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
echo "  Model (grader):    ${INFERENCE_MODEL}"
echo "  Model (query gen): ${INFERENCE_QUERY_MODEL}"
echo "  DB:          ${INFERENCE_DB}"
echo "  Workers:     ${NUM_WORKERS}"
echo "  Queries:     ${N_QUERIES}"
echo "  Device:      ${INFERENCE_DEVICE} (embedding=${INFERENCE_EMBEDDING_DEVICE}, reranker=${INFERENCE_RERANKER_DEVICE})"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

python3 -u multi_shot_retrieval.py \
    --retrieval_method vector \
    --db "${INFERENCE_DB}" \
    ${EVAL_FLAG} \
    --max-iterations "${INFERENCE_MAX_ITERATIONS}" \
    --max-sub-queries "${INFERENCE_MAX_SUB_QUERIES}" \
    --device "${INFERENCE_DEVICE}" \
    --embedding-device "${INFERENCE_EMBEDDING_DEVICE}" \
    --reranker-device "${INFERENCE_RERANKER_DEVICE}" \
    --retrieval_strategy fixed_k \
    --retriever_model "${INFERENCE_RETRIEVER_MODEL}" \
    --top_k_retriever "${INFERENCE_TOP_K_RETRIEVER}" \
    --generate-answer \
    --num-workers "${NUM_WORKERS}" \
    --temperature "${INFERENCE_TEMPERATURE}" \
    --max-retries "${INFERENCE_MAX_RETRIES}" \
    --output-dir "${OUTPUT_DIR}" \
    --llm_model "${INFERENCE_MODEL}" \
    --query_model "${INFERENCE_QUERY_MODEL}" \
    --llm_service_url "${INFERENCE_LLM_URL}" \
    ${INFERENCE_GRADER_URL:+--grader-service-url "${INFERENCE_GRADER_URL}"} \
    ${INFERENCE_GRADER_MODEL:+--grader-model "${INFERENCE_GRADER_MODEL}"} \
    ${INFERENCE_QUERY_URL:+--query-service-url "${INFERENCE_QUERY_URL}"} \
    ${INFERENCE_SUFFICIENCY_URL:+--sufficiency-service-url "${INFERENCE_SUFFICIENCY_URL}"} \
    ${INFERENCE_SUFFICIENCY_MODEL:+--sufficiency-model "${INFERENCE_SUFFICIENCY_MODEL}"} \
    2>&1 | tee "${LOG_FILE}"

if [[ -f "${OUTPUT_DIR}/result_multi_shot.json" ]]; then
    mv "${OUTPUT_DIR}/result_multi_shot.json" "${RESULT_JSON}"
    echo "Saved results to ${RESULT_JSON}"
fi

echo ""
echo "=== Scoring with LLM judge ==="
python3 -u evaluate.py "${RESULT_JSON}" \
    --dataset "${DATASET:-data/frames_dataset.tsv}" \
    --judge-url "${INFERENCE_JUDGE_URL}" \
    --judge-model "${INFERENCE_JUDGE_MODEL}" \
    --batch-size 4

echo ""
echo "=== Done ==="
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Results:     ${RESULT_JSON}"
echo "  Score:       ${SCORE_FILE}"
echo "  Run log:     ${LOG_FILE}"
