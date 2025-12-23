#!/bin/bash
# Setup script for vLLM Backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force-rebuild    Force rebuild MLPerf LoadGen"
            echo "  --help            Show help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Setting up vLLM Backend ==="

check_uv_installed

VENV_DIR="/work/.venv_vllm"
MLPERF_BACKEND="vllm"

setup_virtual_environment "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

install_build_dependencies

echo "Installing MLPerf LoadGen..."
install_mlperf_loadgen "$FORCE_REBUILD" "$MLPERF_BACKEND"

echo "Installing vLLM dependencies..."
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
    echo "vLLM available: $VLLM_VERSION"
else
    echo "Warning: vLLM not found"
fi

echo "Installing flash attention..."
VIRTUAL_ENV=$VENV_DIR uv pip install flashinfer || echo "Warning: flashinfer install failed"

print_setup_info "$VENV_DIR" "vllm"

echo ""
echo "=== vLLM Backend Setup Complete ==="
echo "Ready for vLLM inference"
