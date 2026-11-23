#!/bin/bash
# =============================================================================
# Single-shot retrieval experiment
#
# Usage (from repo root):
#   bash scripts/run_single_shot.sh [N_QUERIES]
#
#   N_QUERIES: optional positional override for N_QUERIES.
#              Use 'all' for the full dataset.
#
# Configuration: see config.template.sh.
#
# Prerequisites:
#   - LLM server running (or OPENROUTER_API_KEY for remote judge)
#   - scripts/run_ingestion.sh has been run (vector DB exists)
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

QNA_DEVICE="${QNA_DEVICE:-cpu}"
QNA_EMBEDDING_DEVICE="${QNA_EMBEDDING_DEVICE:-${QNA_DEVICE}}"
RERANKER_DEVICE="${RERANKER_DEVICE:-${QNA_DEVICE}}"
DATABASE="${DATABASE:-vector_html_hnsw_len768_ov32_word}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
TOP_K_RETRIEVER="${TOP_K_RETRIEVER:-10}"
N_QUERIES="${N_QUERIES:-5}"
GRADER_SERVICE_URL="${GRADER_SERVICE_URL:-http://127.0.0.1:8192/v1/chat/completions}"
GRADER_MODEL="${GRADER_MODEL:-gpt-oss-20b-mxfp4}"
JUDGE_SERVICE_URL="${JUDGE_SERVICE_URL:-http://127.0.0.1:8125/v1/chat/completions}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

N_QUERIES="${1:-${N_QUERIES}}"

if [[ "${N_QUERIES}" == "all" ]]; then
    EVAL_FLAG="--eval"
    TAG="full"
else
    EVAL_FLAG="--eval ${N_QUERIES}"
    TAG="n${N_QUERIES}"
fi

OUTPUT_DIR="output_single_shot_${TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"
RESULT_JSON="${OUTPUT_DIR}/result_single_shot_${TAG}.json"
LOG_FILE="${OUTPUT_DIR}/run.log"

echo "=== Single-shot retrieval ==="
echo "  Model:       ${GRADER_MODEL}"
echo "  DB:          ${DATABASE}"
echo "  Queries:     ${N_QUERIES}"
echo "  Device:      ${QNA_DEVICE} (embedding=${QNA_EMBEDDING_DEVICE}, reranker=${RERANKER_DEVICE})"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

python3 -u single_shot_retrieval.py \
    --db "${DATABASE}" \
    ${EVAL_FLAG} \
    --device "${QNA_DEVICE}" \
    --embedding-device "${QNA_EMBEDDING_DEVICE}" \
    --reranker-device "${RERANKER_DEVICE}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --top_k_retriever "${TOP_K_RETRIEVER}" \
    --generate-answer \
    --grader_model "${GRADER_MODEL}" \
    --grader_service_url "${GRADER_SERVICE_URL}" \
    2>&1 | tee "${LOG_FILE}"

if [[ -f result_single_shot.json ]]; then
    mv result_single_shot.json "${RESULT_JSON}"
fi

if [[ -f "${RESULT_JSON}" ]]; then
    echo ""
    echo "=== Scoring with LLM judge ==="
    python3 -u evaluate.py "${RESULT_JSON}" \
        --dataset "${DATASET:-data/frames_dataset.tsv}" \
        --judge_service_url "${JUDGE_SERVICE_URL}" \
        --judge_model "${JUDGE_MODEL}" \
        --batch-size 4
fi

echo ""
echo "=== Done ==="
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Results:     ${RESULT_JSON}"
echo "  Run log:     ${LOG_FILE}"
