#!/bin/bash
# Main Docker launcher for gpt-oss-120b

set -e

# Default to vLLM if not specified
BACKEND=${MLPERF_BACKEND:-vllm}

echo "Launching Docker container for backend: $BACKEND"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$BACKEND" in
    sglang)
        exec bash "$SCRIPT_DIR/docker/launch_scripts/launch_sglang.sh" "$@"
        ;;
    vllm)
        exec bash "$SCRIPT_DIR/docker/launch_scripts/launch_vllm.sh" "$@"
        ;;
    *)
        echo "Error: Unknown backend '$BACKEND'"
        echo "Set MLPERF_BACKEND to: sglang or vllm"
        exit 1
        ;;
esac
