#!/bin/bash
# =============================================================================
# Multi-shot baseline experiment — GPT-OSS 20B (grader) + GPT-OSS 120B (query gen + sufficiency)
#
# Usage:
#   OPENROUTER_API_KEY="sk-or-v1-YOUR_KEY_HERE" bash scripts/run_multi_shot.sh [N_QUERIES]
#
#   N_QUERIES: number of queries to evaluate (default: 50 for pilot run)
#              Use 'all' or omit for the full dataset (824 queries)
#
# Examples:
#   OPENROUTER_API_KEY="sk-or-v1-..." bash scripts/run_multi_shot.sh 50    # fast pilot run
#   OPENROUTER_API_KEY="sk-or-v1-..." bash scripts/run_multi_shot.sh all   # full evaluation
#
# Prerequisites:
#   - OPENROUTER_API_KEY environment variable set
#   - scripts/setup_db.sh has been run (vector DB exists)
#   - Running on Intel GNR server (for CPU NUMA configuration)
# =============================================================================

set -e

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable not set"
    echo "Usage: OPENROUTER_API_KEY=\"sk-or-v1-YOUR_KEY_HERE\" bash $0"
    exit 1
fi

MODEL="/model/gpt-oss-20b-mxfp4"
QUERY_MODEL="/model/gpt-oss-120b-mxfp4"
DB="vector_html_hnsw_len768_ov32_word"
LLM_URL="http://127.0.0.1:8123/v1/chat/completions"

# Architecture:
# - Document grader: gpt-oss-20b via OpenRouter (openai/gpt-oss-20b)
# - Sufficiency checker: gpt-oss-120b via OpenRouter (openai/gpt-oss-120b)
# - Query generator: gpt-oss-120b via OpenRouter (openai/gpt-oss-120b)
# - Answer generator: gpt-oss-120b via OpenRouter (openai/gpt-oss-120b)
# - Embeddings: CPU with numactl (node 1, cores 43-85, memory 1)
# - Reranking: CPU with numactl (node 1, cores 43-85, memory 1)

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

echo "=== Multi-shot baseline: GPT-OSS 20B (grader) + GPT-OSS 120B (query gen) ==="
echo "  Model (grader):    ${MODEL}"
echo "  Model (query gen): ${QUERY_MODEL}"
echo "  DB:          ${DB}"
echo "  Queries:     ${N_QUERIES}"
echo "  Result file: ${RESULT_JSON}"
echo ""

# ── Run multi-shot retrieval ──────────────────────────────────────────────────
# Note: Embeddings and reranking now run on CPU with NUMA configuration
# NUMA node 1, cores 43-85, memory from node 1
echo "Running with CPU-based embeddings and reranking (NUMA node 1, cores 43-85)"

numactl --cpunodebind=1 --membind=1 --physcpubind=43-85 \
python3 -u multi_shot_retrieval.py \
    --retrieval_method vector \
    --db "${DB}" \
    ${EVAL_FLAG} \
    --max-iterations 5 \
    --max-sub-queries 3 \
    --device cpu \
    --retrieval_strategy fixed_k \
    --retriever_model /data/model/e5-base-v2 \
    --top_k_retriever 15 \
    --generate-answer \
    --llm_model "${MODEL}" \
    --query_model "${QUERY_MODEL}" \
    --llm_service_url "${LLM_URL}"

# Rename output to avoid overwrite on next run
if [[ -f "result_multi_shot.json" ]]; then
    mv result_multi_shot.json "${RESULT_JSON}"
    echo "Saved results to ${RESULT_JSON}"
fi

# ── Score with LLM judge ──────────────────────────────────────────────────────
echo ""
echo "=== Scoring with LLM judge (OpenRouter gpt-oss-20b) ==="
python3 -u evaluate.py "${RESULT_JSON}" \
    --dataset "${DATASET:-data/frames_dataset.tsv}" \
    --judge-url "https://openrouter.ai/api/v1/chat/completions" \
    --judge-model "openai/gpt-oss-20b" \
    --batch-size 4


echo ""
echo "=== Done ==="
echo "  Retrieval log: ${LOG_FILE}"
echo "  Results:       ${RESULT_JSON}"
echo "  Score:         ${SCORE_FILE}"
