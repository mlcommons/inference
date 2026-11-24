#!/bin/bash
# =============================================================================
# Generate a reference DB manifest from this system's vector DB.
#
# Usage (from repo root):
#   bash scripts/write_db_manifest.sh [OUTPUT]
#
#   OUTPUT: optional path for the manifest JSON. Use .json.gz to compress.
#           default: db_manifest_$(hostname -s).json.gz
#
# Configuration: see config.template.sh (uses DATABASE and EMBEDDING_MODEL).
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

OUTPUT="${1:-db_manifest_$(hostname -s).json.gz}"

echo "=== Writing DB manifest ==="
echo "  DB:        ${DATABASE}"
echo "  Embedding model: ${EMBEDDING_MODEL}"
echo "  Output:    ${OUTPUT}"
echo ""

python3 -u db_manifest.py write \
    --db "${DATABASE}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --output "${OUTPUT}"
