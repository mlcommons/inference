#!/bin/bash
# =============================================================================
# Oracle E2E evaluation — feeds ground-truth Wikipedia articles to the LLM.
# Bypasses retrieval entirely; gives an upper bound on answer accuracy.
#
# Usage (from repo root):
#   bash scripts/run_oracle.sh [N_QUERIES]
#
#   N_QUERIES: optional positional override for INFERENCE_N_QUERIES.
#              Use 'all' for the full dataset.
#
# Configuration: see config.template.sh (INFERENCE_ORACLE_* knobs).
#
# Prerequisites:
#   - LLM server running on the configured INFERENCE_LLM_URL
#   - wiki_articles/ directory exists (run download_docs.py first)
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

INFERENCE_LLM_URL="${INFERENCE_LLM_URL:-http://127.0.0.1:8123/v1/chat/completions}"
INFERENCE_MODEL="${INFERENCE_MODEL:-/model/gpt-oss-20b-mxfp4}"
INFERENCE_N_QUERIES="${INFERENCE_N_QUERIES:-5}"
INFERENCE_ORACLE_BATCH_SIZE="${INFERENCE_ORACLE_BATCH_SIZE:-4}"
INFERENCE_ORACLE_TIMEOUT="${INFERENCE_ORACLE_TIMEOUT:-2400}"
INFERENCE_ORACLE_DATASET="${INFERENCE_ORACLE_DATASET:-data/frames_dataset.tsv}"
INFERENCE_ORACLE_WIKI_DIR="${INFERENCE_ORACLE_WIKI_DIR:-wiki_articles}"
INFERENCE_JUDGE_URL="${INFERENCE_JUDGE_URL:-${INFERENCE_LLM_URL}}"
INFERENCE_JUDGE_MODEL="${INFERENCE_JUDGE_MODEL:-${INFERENCE_MODEL}}"

THINKING_FLAG=""
if [[ "${INFERENCE_ORACLE_ENABLE_THINKING}" == "1" ]]; then
    THINKING_FLAG="--enable-thinking"
fi

N_QUERIES="${1:-${INFERENCE_N_QUERIES}}"
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
echo "  Model:       ${INFERENCE_MODEL}"
echo "  Dataset:     ${INFERENCE_ORACLE_DATASET}"
echo "  Wiki dir:    ${INFERENCE_ORACLE_WIKI_DIR}"
echo "  Queries:     ${N_QUERIES}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo ""

echo "=== Step 1: Generating answers with oracle documents ==="
python3 -u oracle_single_shot.py \
    --dataset "${INFERENCE_ORACLE_DATASET}" \
    --wiki-articles-dir "${INFERENCE_ORACLE_WIKI_DIR}" \
    --checkpoint-file "${CHECKPOINT}" \
    --service-url "${INFERENCE_LLM_URL}" \
    --model-name "${INFERENCE_MODEL}" \
    --batch-size "${INFERENCE_ORACLE_BATCH_SIZE}" \
    --timeout "${INFERENCE_ORACLE_TIMEOUT}" \
    ${THINKING_FLAG} \
    ${MAX_QUERIES_FLAG} \
    |& tee "${LOG_FILE}"

echo ""
echo "=== Step 2: Scoring with LLM judge ==="
python3 -u evaluate.py "${CHECKPOINT}" \
    --dataset "${INFERENCE_ORACLE_DATASET}" \
    --judge-url "${INFERENCE_JUDGE_URL}" \
    --judge-model "${INFERENCE_JUDGE_MODEL}" \
    --batch-size 4 \
    |& tee "${SCORE_FILE}"

echo ""
echo "=== Done ==="
echo "  Oracle log:  ${LOG_FILE}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Score:       ${SCORE_FILE}"
