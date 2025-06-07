#!/bin/bash
# Setup script for MLPerf DeepSeek evaluation environment - SGLang Backend
# This script sets up the SGLang backend with virtual environment activated

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
            echo "SGLang Backend Setup:"
            echo "- Creates and activates virtual environment for all operations"
            echo "- Installs accuracy evaluation dependencies"
            echo "- Sets up MLPerf LoadGen"
            echo "- Installs SGLang-specific optimizations"
            echo "- Virtual environment remains active after setup"
            echo ""
            echo "Note: SGLang server may take 20+ minutes to start for large models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Setting up MLPerf DeepSeek evaluation environment - SGLang Backend ==="
echo "=== Virtual environment will be active for all SGLang operations ==="

# Check if uv is installed
check_uv_installed

# Set the virtual environment directory
VENV_DIR="/work/.venv_sglang"

# Setup virtual environment
setup_virtual_environment "$VENV_DIR"

# Activate the virtual environment for setup and subsequent use
echo "Activating virtual environment for SGLang backend..."
source "$VENV_DIR/bin/activate"

# Install build dependencies
install_build_dependencies

# Apply patch to fix prm800k setup.py
patch_prm800k_setup

# Install evaluation requirements
install_evaluation_requirements

# Install MLPerf LoadGen
install_mlperf_loadgen "$FORCE_REBUILD" "$MLPERF_BACKEND"

# SGLang-specific setup
echo ""
echo "=== SGLang Backend-Specific Setup ==="

# Install core dependencies for SGLang
echo "Installing core dependencies for SGLang..."
VIRTUAL_ENV=$VENV_DIR uv pip install \
    transformers \
    pandas \
    numpy \
    tqdm \
    huggingface_hub \
    datasets \
    accelerate \
    openai \
    httpx \
    requests \
    torch==2.6.0 \
    sglang[all]==0.4.6.post5 

# Verify SGLang installation
if python3 -c "import sglang" 2>/dev/null; then
    SGLANG_VERSION=$(python3 -c "import sglang; print(sglang.__version__)")
    echo "SGLang installed successfully: version $SGLANG_VERSION"
else
    echo "Error: SGLang installation failed"
    exit 1
fi

# Install sgl-kernel for optimizations
echo "Installing sgl-kernel..."
VIRTUAL_ENV=$VENV_DIR uv pip install sgl-kernel --no-deps --force-reinstall || echo "Warning: sgl-kernel installation failed"

# Verify sgl_kernel installation
if python3 -c "import sgl_kernel" 2>/dev/null; then
    echo "sgl_kernel installed successfully"
else
    echo "Warning: sgl_kernel import failed - this optimization may not be available"
fi

# Verify torch is available for SGLang
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

# Create logs directory for server output
echo "Creating logs directory for SGLang server..."
mkdir -p /work/logs

# Print component installation summary
echo ""
echo "=== Component Installation Summary ==="
echo -n "SGLang: "
python3 -c "import sglang; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"

echo -n "FlashInfer: "
python3 -c "import flashinfer; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"

echo -n "sgl_kernel: "
python3 -c "import sgl_kernel; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"

echo -n "DeepGEMM: "
python3 -c "import deepgemm; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"

echo -n "MLPerf LoadGen: "
python3 -c "import mlperf_loadgen; print('✓ Installed')" 2>/dev/null || echo "✗ Not installed"

echo ""

# Print setup information (with venv activated)
print_setup_info "$VENV_DIR" "sglang" "true"

echo ""
echo "=== SGLang Backend Setup Complete ==="
echo "Virtual environment is now active and will remain active."
echo ""
echo "IMPORTANT: SGLang Server Startup Time"
echo "The SGLang server may take 20-30 minutes to start for large models like DeepSeek-R1."
echo "The backend will show a progress bar during server startup."
echo ""
echo "Server logs will be saved to: /work/logs/sglang_server_*.log"
echo ""
echo "Ready for SGLang inference and MLPerf runs." 