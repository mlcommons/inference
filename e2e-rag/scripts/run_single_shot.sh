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

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

N_QUERIES="${N_QUERIES:-5}"

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
        --dataset "${DATASET_PATH}" \
        --judge_service_url "${JUDGE_SERVICE_URL}" \
        --judge_model "${JUDGE_MODEL}" \
        --batch-size 4
fi

echo ""
echo "=== Done ==="
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Results:     ${RESULT_JSON}"
echo "  Run log:     ${LOG_FILE}"
