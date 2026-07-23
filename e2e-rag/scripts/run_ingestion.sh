#!/bin/bash
# =============================================================================
# Setup: Build passages JSON and vector DB from downloaded HTML documents
#
# Run this once before any retrieval experiments.
# Assumes: doc_html/ is populated and data/frames_dataset.tsv exists.
#
# Usage (from repo root):
#   bash scripts/run_ingestion.sh
#
# Output (paths configurable via INGESTION_PASSAGES_JSON / INGESTION_DB):
#   ${INGESTION_PASSAGES_JSON}    — passage chunks
#   ${INGESTION_DB}.db            — FAISS HNSW vector index
# =============================================================================

set -e

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

INGESTION_DEVICE="${INGESTION_DEVICE:-cpu}"
INGESTION_EMBEDDING_DEVICE="${INGESTION_EMBEDDING_DEVICE:-${INGESTION_DEVICE}}"
INGESTION_NUM_EMBEDDING_DEVICES="${INGESTION_NUM_EMBEDDING_DEVICES:-4}"
INGESTION_CHUNK_LEN="${INGESTION_CHUNK_LEN:-768}"
INGESTION_CHUNK_OVERLAP="${INGESTION_CHUNK_OVERLAP:-32}"
INGESTION_RETRIEVER_MODEL="${INGESTION_RETRIEVER_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
INGESTION_DOC_DIR="${INGESTION_DOC_DIR:-doc_html}"
INGESTION_PASSAGES_JSON="${INGESTION_PASSAGES_JSON:-passages/doc_html_len${INGESTION_CHUNK_LEN}_ov${INGESTION_CHUNK_OVERLAP}_word.json}"
INGESTION_DB="${INGESTION_DB:-vector_html_hnsw_len${INGESTION_CHUNK_LEN}_ov${INGESTION_CHUNK_OVERLAP}_word}"

echo "=== Step 1: Extract passages from HTML documents ==="
mkdir -p "$(dirname "${INGESTION_PASSAGES_JSON}")"

python3 -u read_docs.py "${INGESTION_DOC_DIR}" "${INGESTION_DOC_DIR}_text" \
    --fixed-length "${INGESTION_CHUNK_LEN}" \
    --fixed-overlap "${INGESTION_CHUNK_OVERLAP}" \
    --text-boundary word \
    --json "${INGESTION_PASSAGES_JSON}" \
    |& tee setup_read_docs.log

echo ""
echo "=== Step 2: Build FAISS HNSW vector index ==="

python3 -u single_shot_retrieval.py \
    --ingest "${INGESTION_PASSAGES_JSON}" \
    --db "${INGESTION_DB}" \
    --device "${INGESTION_DEVICE}" \
    --embedding-device "${INGESTION_EMBEDDING_DEVICE}" \
    --retriever_model "${INGESTION_RETRIEVER_MODEL}" \
    --num_embedding_devices "${INGESTION_NUM_EMBEDDING_DEVICES}" \
    |& tee setup_build_db.log

echo ""
echo "=== DB setup complete ==="
echo "  Passages: ${INGESTION_PASSAGES_JSON}"
echo "  Vector DB: ${INGESTION_DB}.db"
