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

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

echo "=== Step 1: Extract passages from HTML documents ==="
mkdir -p "$(dirname "${INGESTION_PASSAGES_JSON}")"

python3 -u read_docs.py "${INGESTION_DOC_DIR}" "${INGESTION_DOC_DIR}_text" \
    --fixed-length "${INGESTION_CHUNK_SIZE}" \
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
    --embedding_model "${EMBEDDING_MODEL}" \
    --num_embedding_devices "${INGESTION_NUM_EMBEDDING_DEVICES}" \
    |& tee setup_build_db.log

echo ""
echo "=== DB setup complete ==="
echo "  Passages: ${INGESTION_PASSAGES_JSON}"
echo "  Vector DB: ${INGESTION_DB}.db"
