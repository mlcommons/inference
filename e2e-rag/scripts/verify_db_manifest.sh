#!/bin/bash
# =============================================================================
# Verify this system's vector DB against a reference manifest.
#
# Usage (from repo root):
#   bash scripts/verify_db_manifest.sh MANIFEST [COSINE_THRESHOLD] [TOP_K_DEPTH]
#
#   MANIFEST:         path to the reference manifest JSON (required).
#   COSINE_THRESHOLD: minimum sample-embedding cosine similarity (default: 0.9999).
#   TOP_K_DEPTH:      probe-query top-K rank match depth (default: 3).
#
# Configuration: see config.template.sh (uses INFERENCE_DB and
# INFERENCE_RETRIEVER_MODEL).
# =============================================================================

set -e

if [[ -z "$1" ]]; then
    echo "ERROR: manifest path required" >&2
    echo "Usage: $0 MANIFEST [COSINE_THRESHOLD] [TOP_K_DEPTH]" >&2
    exit 1
fi

CONFIG="${CONFIG:-config.sh}"
if [[ -f "${CONFIG}" ]]; then
    source "${CONFIG}"
else
    echo "WARNING: ${CONFIG} not found; using built-in defaults" >&2
fi

INFERENCE_DB="${INFERENCE_DB:-vector_html_hnsw_len768_ov32_word}"
INFERENCE_RETRIEVER_MODEL="${INFERENCE_RETRIEVER_MODEL:-intfloat_e5-base-v2/e5-base-v2}"

MANIFEST="$1"
COSINE_THRESHOLD="${2:-0.9999}"
TOP_K_DEPTH="${3:-3}"

echo "=== Verifying DB against manifest ==="
echo "  DB:                ${INFERENCE_DB}"
echo "  Retriever:         ${INFERENCE_RETRIEVER_MODEL}"
echo "  Manifest:          ${MANIFEST}"
echo "  Cosine threshold:  ${COSINE_THRESHOLD}"
echo "  Top-K depth:       ${TOP_K_DEPTH}"
echo ""

python3 -u db_manifest.py verify \
    --db "${INFERENCE_DB}" \
    --manifest "${MANIFEST}" \
    --retriever_model "${INFERENCE_RETRIEVER_MODEL}" \
    --cosine-threshold "${COSINE_THRESHOLD}" \
    --top-k-depth "${TOP_K_DEPTH}"
