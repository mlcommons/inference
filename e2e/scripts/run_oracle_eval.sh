#!/bin/bash
# =============================================================================
# Oracle E2E Evaluation — GPT-OSS 20B
#
# Bypasses retrieval entirely. Feeds ground-truth Wikipedia articles directly
# to the LLM and then scores the answers with an LLM judge.
# This gives an upper-bound on end-to-end answer accuracy.
#
# Usage:
#   bash scripts/run_oracle_eval.sh [N_QUERIES]
#
#   N_QUERIES: number of queries to evaluate (default: 5)
#              Use 'all' or omit for the full dataset
#
# Examples:
#   bash scripts/run_oracle_eval.sh 5     # quick sanity check
#   bash scripts/run_oracle_eval.sh 50    # pilot run
#   bash scripts/run_oracle_eval.sh all   # full evaluation
#
# Prerequisites:
#   - LLM server running on port 8123 with gpt-oss-20b-mxfp4 model
#   - wiki_articles/ directory exists (run download_docs.py first)
# =============================================================================

set -e

MODEL="/model/gpt-oss-20b-mxfp4"
LLM_URL="http://127.0.0.1:8123/v1/chat/completions"
DATASET="data/frames_dataset.tsv"
WIKI_DIR="wiki_articles"

N_QUERIES="${1:-5}"
if [[ "${N_QUERIES}" == "all" ]]; then
    TAG="full"
    MAX_QUERIES_FLAG=""
else
    TAG="n${N_QUERIES}"
    MAX_QUERIES_FLAG="--max-queries ${N_QUERIES}"
fi

CHECKPOINT="oracle_checkpoint_${TAG}.pkl"
SCORE_FILE="score_oracle_${TAG}.txt"
LOG_FILE="logs_2048/oracle_${TAG}.log"

mkdir -p logs_2048

echo "=== Oracle E2E Evaluation: GPT-OSS 20B ==="
echo "  Model:       ${MODEL}"
echo "  Dataset:     ${DATASET}"
echo "  Wiki dir:    ${WIKI_DIR}"
echo "  Queries:     ${N_QUERIES}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo ""

# ── Step 1: Generate LLM answers using oracle (ground-truth) documents ────────
echo "=== Step 1: Generating answers with oracle documents ==="
python3 -u oracle_single_shot.py \
    --dataset "${DATASET}" \
    --wiki-articles-dir "${WIKI_DIR}" \
    --checkpoint-file "${CHECKPOINT}" \
    --service-url "${LLM_URL}" \
    --model-name "${MODEL}" \
    --batch-size 4 \
    ${MAX_QUERIES_FLAG} \
    |& tee "${LOG_FILE}"

echo ""

# ── Step 2: Score with LLM judge ──────────────────────────────────────────────
echo "=== Step 2: Scoring with LLM judge ==="
python3 -u evaluate.py "${CHECKPOINT}" \
    --dataset "${DATASET}" \
    --judge-url "${LLM_URL}" \
    --judge-model "${MODEL}" \
    --batch-size 4 \
    |& tee "${SCORE_FILE}"

echo ""
echo "=== Done ==="
echo "  Oracle log:  ${LOG_FILE}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "  Score:       ${SCORE_FILE}"
