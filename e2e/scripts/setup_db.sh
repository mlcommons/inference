#!/bin/bash
# =============================================================================
# Setup: Build passages JSON and vector DB from downloaded HTML documents
#
# Run this once before any retrieval experiments.
# Assumes: doc_html/ is populated and data/frames_dataset.tsv exists.
#
# Usage (from workspace root):
#   bash scripts/setup_db.sh
#
# Output:
#   passages/doc_html_len768_ov32_word.json    — passage chunks
#   vector_html_hnsw_len768_ov32_word.db       — FAISS HNSW vector index
# =============================================================================

set -e

NUM_EMBEDDING_DEVICES="${NUM_EMBEDDING_DEVICES:-4}"

echo "=== Step 1: Extract passages from HTML documents ==="
mkdir -p passages

python3 -u read_docs.py doc_html doc_html_text \
    --fixed-length 768 \
    --fixed-overlap 32 \
    --text-boundary word \
    --json passages/doc_html_len768_ov32_word.json \
    |& tee setup_read_docs.log

echo ""
echo "=== Step 2: Build FAISS HNSW vector index ==="

python3 -u single_shot_retrieval.py \
    --ingest passages/doc_html_len768_ov32_word.json \
    --db vector_html_hnsw_len768_ov32_word \
    --retrieval_method vector \
    --vector_index_method hnsw \
    --device xpu \
    --num_embedding_devices "${NUM_EMBEDDING_DEVICES}" \
    |& tee setup_build_db.log

echo ""
echo "=== DB setup complete ==="
echo "  Passages: passages/doc_html_len768_ov32_word.json"
echo "  Vector DB: vector_html_hnsw_len768_ov32_word.db"
