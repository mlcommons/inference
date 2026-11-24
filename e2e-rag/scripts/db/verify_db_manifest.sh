#!/bin/bash
# =============================================================================
# Verify this system's vector DB against a reference manifest.
#
# Usage (from repo root):
#   bash scripts/db/verify_db_manifest.sh MANIFEST [RETRIEVAL_THRESHOLD]
#
#   MANIFEST:            path to the reference manifest JSON (required).
#   RETRIEVAL_THRESHOLD: minimum mean reference-query top-K document set-overlap
#                        required to pass (default: 0.95).
#
# Configuration: see config.template.sh (uses DATABASE and EMBEDDING_MODEL).
# =============================================================================

set -e

if [[ -z "$1" ]]; then
    echo "ERROR: manifest path required" >&2
    echo "Usage: $0 MANIFEST [RETRIEVAL_THRESHOLD]" >&2
    exit 1
fi

# Run from e2e-rag/ (paths below are relative to it).
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

MANIFEST="$1"
RETRIEVAL_THRESHOLD="${2:-0.95}"

echo "=== Verifying DB against manifest ==="
echo "  DB:                   ${DATABASE}"
echo "  Embedding model:      ${EMBEDDING_MODEL}"
echo "  Manifest:             ${MANIFEST}"
echo "  Retrieval threshold:  ${RETRIEVAL_THRESHOLD}"
echo ""

python3 -u -m ingestion.db_manifest verify \
    --db "${DATABASE}" \
    --manifest "${MANIFEST}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --retrieval-threshold "${RETRIEVAL_THRESHOLD}"
