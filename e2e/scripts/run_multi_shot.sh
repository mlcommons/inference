#!/bin/bash
# =============================================================================
# Multi-shot baseline experiment — GPT-OSS 20B
#
# Usage:
#   bash scripts/run_multi_shot.sh [N_QUERIES]
#
#   N_QUERIES: number of queries to evaluate (default: 50 for pilot run)
#              Use 'all' or omit for the full dataset (824 queries)
#
# Examples:
#   bash scripts/run_multi_shot.sh 50    # fast pilot run
#   bash scripts/run_multi_shot.sh all   # full evaluation
#
# Prerequisites:
#   - LLM server running on port 8123 with gpt-oss-20b-mxfp4 model
#   - scripts/setup_db.sh has been run (vector DB exists)
#   - evaluate.py judge server running on port 8124 (for scoring)
# =============================================================================

set -e

MODEL="/model/gpt-oss-20b-mxfp4"
DB="vector_html_hnsw_len768_ov32_word"
LLM_URL="http://127.0.0.1:8123/v1/chat/completions"

# Eval size: positional arg or default 5
N_QUERIES="${1:-5}"
if [[ "${N_QUERIES}" == "all" ]]; then
    EVAL_FLAG="--eval"
    TAG="full"
else
    EVAL_FLAG="--eval ${N_QUERIES}"
    TAG="n${N_QUERIES}"
fi

RESULT_JSON="result_multi_shot_len768_${TAG}.json"
LOG_FILE="logs_768/test_${TAG}.log"
SCORE_FILE="score_multi_shot_len768_${TAG}.txt"

echo "=== Multi-shot baseline: GPT-OSS 20B ==="
echo "  Model:       ${MODEL}"
echo "  DB:          ${DB}"
echo "  Queries:     ${N_QUERIES}"
echo "  Result file: ${RESULT_JSON}"
echo ""

# ── Run multi-shot retrieval ──────────────────────────────────────────────────
python3 -u multi_shot_retrieval.py \
    --retrieval_method vector \
    --db "${DB}" \
    ${EVAL_FLAG} \
    --no-rerank \
    --max-iterations 5 \
    --max-sub-queries 3 \
    --device xpu \
    --retrieval_strategy fixed_k \
    --top_k_retriever 15 \
    --top_k_reranking 15 \
    --generate-answer \
    --llm_model "${MODEL}" \
    --llm_service_url "${LLM_URL}"

# Rename output to avoid overwrite on next run
if [[ -f "result_multi_shot.json" ]]; then
    mv result_multi_shot.json "${RESULT_JSON}"
    echo "Saved results to ${RESULT_JSON}"
fi

# ── Score with LLM judge ──────────────────────────────────────────────────────
echo ""
echo "=== Scoring with LLM judge ==="
python3 -u evaluate.py "${RESULT_JSON}" \
    --dataset "${DATASET:-data/frames_dataset.tsv}" \
    --judge-url "${LLM_URL}" \
    --judge-model "${MODEL}" \
    --batch-size 4


echo ""
echo "=== Done ==="
echo "  Retrieval log: ${LOG_FILE}"
echo "  Results:       ${RESULT_JSON}"
echo "  Score:         ${SCORE_FILE}"
