#!/bin/bash
# Main setup script for gpt-oss-120b

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect backend from MLPERF_BACKEND environment variable
if [ -z "$MLPERF_BACKEND" ]; then
    echo "Error: MLPERF_BACKEND environment variable not set"
    echo "Please set it to one of: sglang, vllm"
    echo "Example: export MLPERF_BACKEND=vllm"
    exit 1
fi

echo "Setting up backend: $MLPERF_BACKEND"

case "$MLPERF_BACKEND" in
    sglang)
        echo "Running SGLang setup..."
        bash "$SCRIPT_DIR/docker/setup_scripts/setup_sglang.sh" "$@"
        ;;
    vllm)
        echo "Running vLLM setup..."
        bash "$SCRIPT_DIR/docker/setup_scripts/setup_vllm.sh" "$@"
        ;;
    *)
        echo "Error: Unknown backend '$MLPERF_BACKEND'"
        echo "Supported backends: sglang, vllm"
        exit 1
        ;;
esac

echo ""
echo "=== Setup Complete for backend: $MLPERF_BACKEND ==="
