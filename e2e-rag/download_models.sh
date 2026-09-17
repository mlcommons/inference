#!/bin/bash
# Downloads models required for the multi-shot RAG pipeline to /model/.
#
# Models:
#   - intfloat/e5-base-v2          → /model/e5-base-v2        (embeddings)
#   - colbert-ir/colbertv2.0       → /model/colbertv2.0       (reranker)
#
# Usage (run on host, not inside container):
#   bash download_models.sh                    # download both
#   bash download_models.sh e5-base-v2         # embeddings only
#   bash download_models.sh colbertv2.0        # reranker only
#
# Set DEST_DIR to override the default /model destination.

set -e

DEST_DIR="${DEST_DIR:-/model}"

# Disable SSL verification to work around corporate CA cert issues
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""

download_model() {
    local HF_REPO="$1"
    local LOCAL_DIR="$2"

    if [[ -d "${LOCAL_DIR}" && -n "$(ls -A "${LOCAL_DIR}" 2>/dev/null)" ]]; then
        echo "[skip] ${HF_REPO} already exists at ${LOCAL_DIR}"
        return 0
    fi

    echo ""
    echo "=== Downloading ${HF_REPO} → ${LOCAL_DIR} ==="
    mkdir -p "${LOCAL_DIR}"

    python3 - <<EOF
import os, sys
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub not installed; installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${HF_REPO}",
    local_dir="${LOCAL_DIR}",
    local_dir_use_symlinks=False,
)
print("Done: ${HF_REPO}")
EOF
}

TARGET="${1:-all}"

case "${TARGET}" in
    e5-base-v2|e5)
        download_model "intfloat/e5-base-v2" "${DEST_DIR}/e5-base-v2"
        ;;
    colbertv2.0|colbert|reranker)
        download_model "colbert-ir/colbertv2.0" "${DEST_DIR}/colbertv2.0"
        ;;
    all|"")
        download_model "intfloat/e5-base-v2" "${DEST_DIR}/e5-base-v2"
        download_model "colbert-ir/colbertv2.0" "${DEST_DIR}/colbertv2.0"
        ;;
    *)
        echo "Unknown target '${TARGET}'. Use: e5-base-v2 | colbertv2.0 | all"
        exit 1
        ;;
esac

echo ""
echo "=== All requested models downloaded to ${DEST_DIR} ==="
ls -lh "${DEST_DIR}"
