#!/bin/bash
# Setup script for MLPerf DeepSeek evaluation environment - PyTorch Backend
# This script sets up the PyTorch backend with virtual environment activated

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
            echo "PyTorch Backend Setup:"
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

echo "=== Setting up MLPerf DeepSeek evaluation environment - PyTorch Backend ==="
echo "=== Virtual environment will be active for all PyTorch operations ==="

# Check if uv is installed
check_uv_installed

# Set the virtual environment directory
VENV_DIR="/work/.venv_pytorch"

# Setup virtual environment
setup_virtual_environment "$VENV_DIR"

# Activate the virtual environment for setup and subsequent use
echo "Activating virtual environment for PyTorch backend..."
source "$VENV_DIR/bin/activate"

# Install build dependencies
install_build_dependencies

# Apply patch to fix prm800k setup.py
patch_prm800k_setup

# Install evaluation requirements
install_evaluation_requirements

# Install MLPerf LoadGen
install_mlperf_loadgen "$FORCE_REBUILD" "$MLPERF_BACKEND"

# PyTorch-specific setup
echo ""
echo "=== PyTorch Backend-Specific Setup ==="

VIRTUAL_ENV=$VENV_DIR uv pip install -r /opt/ref_dsinfer/inference/requirements.txt

# Verify PyTorch is available (should be from base image)
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "PyTorch is available: version $TORCH_VERSION"
else
    echo "Warning: PyTorch not found in the environment"
fi

# Set PYTHONPATH to include ref_dsinfer paths for checks
export PYTHONPATH="/opt:/opt/ref_dsinfer/inference:${PYTHONPATH}"

# Verify ref_dsinfer package is available
if python3 -c "import ref_dsinfer" 2>/dev/null; then
    echo "DeepSeek-V3 ref_dsinfer package is available"
else
    echo "Warning: ref_dsinfer package not found. Check PYTHONPATH and /opt/ref_dsinfer"
fi

# Check if kernel module is importable
if python3 -c "import sys; sys.path.append('/opt/ref_dsinfer/inference'); from ref_dsinfer.inference.model import Transformer, ModelArgs" 2>/dev/null; then
    echo "✓ ref_dsinfer inference model imports successfully"
else
    echo "✗ ref_dsinfer inference model import failed. Checking kernel module..."
    
    # Try to import kernel directly to get more detailed error
    if python3 -c "import sys; sys.path.append('/opt/ref_dsinfer/inference'); import kernel" 2>/dev/null; then
        echo "✓ kernel module is accessible directly"
    else
        echo "✗ kernel module import failed. This may be expected if CUDA extensions need compilation."
        echo "The kernel.py file should contain fallback implementations for CPU."
    fi
fi

# Model download and conversion
echo ""
echo "=== Model Download and Conversion ==="
echo "Downloading DeepSeek-R1 model..."
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download deepseek-ai/DeepSeek-R1 --revision 56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad --local-dir /raid/data/${USER}/models/deepseek-ai_DeepSeek-R1

echo "Converting model to inference format..."
# Check if converted model already exists
CONVERTED_MODEL_PATH="/raid/data/${USER}/models/deepseek-ai_DeepSeek-R1-Demo"
if [ -d "$CONVERTED_MODEL_PATH" ] && [ -f "$CONVERTED_MODEL_PATH/model0-mp8.safetensors" ] && [ -f "$CONVERTED_MODEL_PATH/tokenizer.json" ]; then
    echo "Converted model already exists at $CONVERTED_MODEL_PATH, skipping conversion..."
else
    echo "Converting model to inference format..."
    python /opt/ref_dsinfer/inference/convert.py --hf-ckpt-path /raid/data/${USER}/models/deepseek-ai_DeepSeek-R1 --save-path /raid/data/${USER}/models/deepseek-ai_DeepSeek-R1-Demo --n-experts 256 --model-parallel 8
fi

echo "Model download and conversion completed."

# Print setup information (with venv activated)
print_setup_info "$VENV_DIR" "pytorch" "true"

echo ""
echo "=== PyTorch Backend Setup Complete ==="
echo "Virtual environment is now active and will remain active."
echo "Ready for PyTorch distributed inference and MLPerf runs." 