#!/bin/bash
# =============================================================================
# Verify this system's vector DB against a reference manifest.
#
# Usage (from repo root):
#   bash scripts/verify_db_manifest.sh MANIFEST [RETRIEVAL_THRESHOLD]
#
#   MANIFEST:            path to the reference manifest JSON (required).
#   RETRIEVAL_THRESHOLD: minimum mean reference-query top-K document set-overlap
#                        required to pass (default: 0.95).
#
# Configuration: see config.template.sh (uses INFERENCE_DB and
# INFERENCE_RETRIEVER_MODEL).
# =============================================================================

set -e

if [[ -z "$1" ]]; then
    echo "ERROR: manifest path required" >&2
    echo "Usage: $0 MANIFEST [RETRIEVAL_THRESHOLD]" >&2
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
RETRIEVAL_THRESHOLD="${2:-0.95}"

echo "=== Verifying DB against manifest ==="
echo "  DB:                   ${INFERENCE_DB}"
echo "  Retriever:            ${INFERENCE_RETRIEVER_MODEL}"
echo "  Manifest:             ${MANIFEST}"
echo "  Retrieval threshold:  ${RETRIEVAL_THRESHOLD}"
echo ""

python3 -u db_manifest.py verify \
    --db "${INFERENCE_DB}" \
    --manifest "${MANIFEST}" \
    --retriever_model "${INFERENCE_RETRIEVER_MODEL}" \
    --retrieval-threshold "${RETRIEVAL_THRESHOLD}"
