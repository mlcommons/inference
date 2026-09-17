#!/bin/bash
# =============================================================================
# Single-shot retrieval experiment
#
# Usage (from repo root):
#   bash scripts/run_single_shot.sh [N_QUERIES]
#
#   N_QUERIES: optional positional override for INFERENCE_N_QUERIES.
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

INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}"
INFERENCE_EMBEDDING_DEVICE="${INFERENCE_EMBEDDING_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_RERANKER_DEVICE="${INFERENCE_RERANKER_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_DB="${INFERENCE_DB:-vector_html_hnsw_len768_ov32_word}"
INFERENCE_RETRIEVER_MODEL="${INFERENCE_RETRIEVER_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
INFERENCE_TOP_K_RETRIEVER="${INFERENCE_TOP_K_RETRIEVER:-15}"
INFERENCE_N_QUERIES="${INFERENCE_N_QUERIES:-5}"
INFERENCE_LLM_URL="${INFERENCE_LLM_URL:-http://127.0.0.1:8192/v1/chat/completions}"
INFERENCE_MODEL="${INFERENCE_MODEL:-gpt-oss-20b-mxfp4}"
INFERENCE_JUDGE_URL="${INFERENCE_JUDGE_URL:-https://openrouter.ai/api/v1/chat/completions}"
INFERENCE_JUDGE_MODEL="${INFERENCE_JUDGE_MODEL:-openai/gpt-oss-20b}"

N_QUERIES="${1:-${INFERENCE_N_QUERIES}}"

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
echo "  Model:       ${INFERENCE_MODEL}"
echo "  DB:          ${INFERENCE_DB}"
echo "  Queries:     ${N_QUERIES}"
echo "  Device:      ${INFERENCE_DEVICE} (embedding=${INFERENCE_EMBEDDING_DEVICE}, reranker=${INFERENCE_RERANKER_DEVICE})"
echo "  Output dir:  ${OUTPUT_DIR}"
echo ""

python3 -u single_shot_retrieval.py \
    --db "${INFERENCE_DB}" \
    ${EVAL_FLAG} \
    --device "${INFERENCE_DEVICE}" \
    --embedding-device "${INFERENCE_EMBEDDING_DEVICE}" \
    --reranker-device "${INFERENCE_RERANKER_DEVICE}" \
    --retriever_model "${INFERENCE_RETRIEVER_MODEL}" \
    --top_k_retriever "${INFERENCE_TOP_K_RETRIEVER}" \
    --generate-answer \
    --llm_model "${INFERENCE_MODEL}" \
    --llm_service_url "${INFERENCE_LLM_URL}" \
    2>&1 | tee "${LOG_FILE}"

if [[ -f result_single_shot.json ]]; then
    mv result_single_shot.json "${RESULT_JSON}"
fi

if [[ -f "${RESULT_JSON}" ]]; then
    echo ""
    echo "=== Scoring with LLM judge ==="
    python3 -u evaluate.py "${RESULT_JSON}" \
        --dataset "${DATASET:-data/frames_dataset.tsv}" \
        --judge-url "${INFERENCE_JUDGE_URL}" \
        --judge-model "${INFERENCE_JUDGE_MODEL}" \
        --batch-size 4
fi

echo ""
echo "=== Done ==="
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Results:     ${RESULT_JSON}"
echo "  Run log:     ${LOG_FILE}"
