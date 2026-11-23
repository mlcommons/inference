#!/bin/bash
# =============================================================================
# Oracle E2E evaluation — feeds ground-truth Wikipedia articles to the LLM.
# Bypasses retrieval entirely; gives an upper bound on answer accuracy.
#
# Usage (from repo root):
#   bash scripts/run_oracle.sh [N_QUERIES]
#
#   N_QUERIES: optional positional override for N_QUERIES.
#              Use 'all' for the full dataset.
#
# Configuration: see config.template.sh (ORACLE_* knobs).
#
# Prerequisites:
#   - LLM server running on the configured ORACLE_SERVICE_URL
#   - wiki_articles/ directory exists (run download_docs.py first)
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

# The oracle answers from the gold articles, so it uses the answer model (120B).
ORACLE_SERVICE_URL="${ORACLE_SERVICE_URL:-${SUFFICIENCY_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}}"
ORACLE_MODEL="${ORACLE_MODEL:-${SUFFICIENCY_MODEL:-gpt-oss-120b-mxfp4}}"
N_QUERIES="${N_QUERIES:-5}"
ORACLE_BATCH_SIZE="${ORACLE_BATCH_SIZE:-4}"
ORACLE_TIMEOUT="${ORACLE_TIMEOUT:-2400}"
ORACLE_DATASET="${ORACLE_DATASET:-frames-benchmark-dataset/frames_dataset.tsv}"
ORACLE_WIKI_DIR="${ORACLE_WIKI_DIR:-wiki_articles}"
JUDGE_SERVICE_URL="${JUDGE_SERVICE_URL:-${ORACLE_SERVICE_URL}}"
JUDGE_MODEL="${JUDGE_MODEL:-${ORACLE_MODEL}}"

THINKING_FLAG=""
if [[ "${ORACLE_ENABLE_THINKING}" == "1" ]]; then
    THINKING_FLAG="--enable-thinking"
fi

N_QUERIES="${1:-${N_QUERIES}}"
if [[ "${N_QUERIES}" == "all" ]]; then
    TAG="full"
    MAX_QUERIES_FLAG=""
else
    TAG="n${N_QUERIES}"
    MAX_QUERIES_FLAG="--max-queries ${N_QUERIES}"
fi

CHECKPOINT="oracle_checkpoint_${TAG}.pkl"
SCORE_FILE="score_oracle_${TAG}.txt"
LOG_FILE="oracle_${TAG}.log"

echo "=== Oracle E2E evaluation ==="
echo "  Model:       ${ORACLE_MODEL}"
echo "  Dataset:     ${ORACLE_DATASET}"
echo "  Wiki dir:    ${ORACLE_WIKI_DIR}"
echo "  Queries:     ${N_QUERIES}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo ""

echo "=== Step 1: Generating answers with oracle documents ==="
python3 -u oracle_single_shot.py \
    --dataset "${ORACLE_DATASET}" \
    --wiki-articles-dir "${ORACLE_WIKI_DIR}" \
    --checkpoint-file "${CHECKPOINT}" \
    --service-url "${ORACLE_SERVICE_URL}" \
    --model-name "${ORACLE_MODEL}" \
    --batch-size "${ORACLE_BATCH_SIZE}" \
    --timeout "${ORACLE_TIMEOUT}" \
    ${THINKING_FLAG} \
    ${MAX_QUERIES_FLAG} \
    |& tee "${LOG_FILE}"

echo ""
echo "=== Step 2: Scoring with LLM judge ==="
python3 -u evaluate.py "${CHECKPOINT}" \
    --dataset "${ORACLE_DATASET}" \
    --judge_service_url "${JUDGE_SERVICE_URL}" \
    --judge_model "${JUDGE_MODEL}" \
    --batch-size 4 \
    |& tee "${SCORE_FILE}"

echo ""
echo "=== Done ==="
echo "  Oracle log:  ${LOG_FILE}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Score:       ${SCORE_FILE}"
