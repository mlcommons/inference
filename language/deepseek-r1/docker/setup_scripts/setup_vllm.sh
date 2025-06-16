#!/bin/bash
# Setup script for MLPerf DeepSeek evaluation environment - vLLM Backend
# This script sets up the vLLM backend with virtual environment activated

set -e  # Exit on error

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse command line arguments
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
            echo "  --force-rebuild    Force rebuild of MLPerf LoadGen from source"
            echo "  --help            Show this help message"
            echo ""
            echo "vLLM Backend Setup:"
            echo "- Creates and activates virtual environment for all operations"
            echo "- Installs accuracy evaluation dependencies"
            echo "- Sets up MLPerf LoadGen"
            echo "- Virtual environment remains active after setup"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Setting up MLPerf DeepSeek evaluation environment - vLLM Backend ==="
echo "=== Virtual environment will be active for all vLLM operations ==="

# Check if uv is installed
check_uv_installed

# Set the virtual environment directory
VENV_DIR="/work/.venv_vllm"

# Setup virtual environment
setup_virtual_environment "$VENV_DIR"

# Activate the virtual environment for setup and subsequent use
echo "Activating virtual environment for vLLM backend..."
source "$VENV_DIR/bin/activate"

# Install build dependencies
install_build_dependencies

# Apply patch to fix prm800k setup.py
patch_prm800k_setup

# Install evaluation requirements
install_evaluation_requirements

# Install MLPerf LoadGen
install_mlperf_loadgen "$FORCE_REBUILD" "$MLPERF_BACKEND"

# vLLM-specific setup
echo ""
echo "=== vLLM Backend-Specific Setup ==="

# Verify vLLM is available (should be from base image)
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
    echo "vLLM is available: version $VLLM_VERSION"
else
    echo "Warning: vLLM not found in the environment"
fi

# Install flash attention for improved performance
echo "Installing flash attention..."
VIRTUAL_ENV=$VENV_DIR uv pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl


# Verify torch is available for vLLM
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch is available: version $TORCH_VERSION"
    
    # Check CUDA availability
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        GPU_COUNT=$(python3 -c "import torch; print('GPU count:', torch.cuda.device_count())" 2>/dev/null || echo "GPU count: 0")
        echo "$GPU_COUNT"
    fi
else
    echo "Warning: PyTorch not found in the environment"
fi

# Print setup information (with venv activated)
print_setup_info "$VENV_DIR" "vllm" "true"

echo ""
echo "=== vLLM Backend Setup Complete ==="
echo "Virtual environment is now active and will remain active."
echo "Ready for vLLM inference and MLPerf runs." 