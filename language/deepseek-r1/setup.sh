#!/bin/bash
# Main setup script for MLPerf DeepSeek evaluation environment
# This script detects the backend and calls the appropriate setup script
#
# Backend detection methods (in order):
# 1. MLPERF_BACKEND environment variable (recommended)
# 2. Automatic detection based on installed packages
#
# Backend-specific behavior:
# - pytorch, vllm, sglang: Virtual environment activated after setup

set -e  # Exit on error

# Function to detect backend based on installed packages and environment
detect_backend() {
    echo "Detecting backend environment..." >&2
    
    # Check for backend-specific markers or environment variables first
    if [ -n "$MLPERF_BACKEND" ]; then
        echo "Using MLPERF_BACKEND environment variable: $MLPERF_BACKEND" >&2
        # Validate the backend value
        case "$MLPERF_BACKEND" in
            pytorch|vllm|sglang)
                echo "$MLPERF_BACKEND"
                return 0
                ;;
            *)
                echo "Warning: Invalid MLPERF_BACKEND value: $MLPERF_BACKEND" >&2
                echo "Valid values: pytorch, vllm, sglang" >&2
                echo "Continuing with automatic detection..." >&2
                ;;
        esac
    fi
    
    # Check for vLLM in system Python
    if python3 -c "import vllm" >/dev/null 2>&1; then
        echo "Detected vLLM in system Python" >&2
        echo "vllm"
        return 0
    fi
    
    # Check for PyTorch with ref_dsinfer (DeepSeek-V3 reference implementation)
    if python3 -c "import torch" >/dev/null 2>&1 && python3 -c "import ref_dsinfer" >/dev/null 2>&1; then
        echo "Detected PyTorch with ref_dsinfer in system Python" >&2
        echo "pytorch"
        return 0
    fi
    
    # For SGLang, check CUDA version and specific PyTorch version
    # SGLang containers typically have CUDA 12.6 and PyTorch 2.6.0
    if command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
        if [[ "$CUDA_VERSION" == "12.6" ]]; then
            # Check if PyTorch 2.6.0 is in system (SGLang specific)
            if python3 -c "import torch; exit(0 if torch.__version__.startswith('2.6.0') else 1)" >/dev/null 2>&1; then
                # Additional check: no vLLM
                if ! python3 -c "import vllm" >/dev/null 2>&1; then
                    echo "Detected SGLang environment (CUDA 12.6, PyTorch 2.6.0)" >&2
                    echo "sglang"
                    return 0
                fi
            fi
        fi
    fi
    
    # Check for general PyTorch environment
    if python3 -c "import torch" >/dev/null 2>&1; then
        echo "Detected PyTorch in system Python" >&2
        echo "pytorch"
        return 0
    fi
    
    # Only check virtual environments as a last resort
    # and verify they are actually valid
    for backend in sglang vllm pytorch; do
        VENV_DIR="/work/.venv_$backend"
        if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python3" ]; then
            # Check if the venv Python is actually working
            if "$VENV_DIR/bin/python3" -c "import sys" >/dev/null 2>&1; then
                echo "Found working $backend virtual environment" >&2
                echo "$backend"
                return 0
            else
                echo "Found broken $backend virtual environment at $VENV_DIR" >&2
            fi
        fi
    done
    
    # If no specific backend detected, return unknown
    echo "unknown"
    return 1
}

# Function to clean up broken virtual environments
cleanup_broken_venvs() {
    echo "Checking for broken virtual environments..." >&2
    for backend in pytorch vllm sglang; do
        VENV_DIR="/work/.venv_$backend"
        if [ -d "$VENV_DIR" ]; then
            if [ ! -x "$VENV_DIR/bin/python3" ] || ! "$VENV_DIR/bin/python3" -c "import sys" >/dev/null 2>&1; then
                echo "Found broken virtual environment at $VENV_DIR" >&2
                echo -n "Remove it? [y/N] " >&2
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    rm -rf "$VENV_DIR"
                    echo "Removed $VENV_DIR" >&2
                fi
            fi
        fi
    done
}

# Parse command line arguments
FORCE_REBUILD=false
BACKEND_OVERRIDE=""
CLEANUP_VENVS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --backend)
            BACKEND_OVERRIDE="$2"
            shift 2
            ;;
        --cleanup-venvs)
            CLEANUP_VENVS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --force-rebuild    Force rebuild of MLPerf LoadGen from source"
            echo "  --backend BACKEND  Override backend detection (pytorch|vllm|sglang)"
            echo "  --cleanup-venvs    Check and remove broken virtual environments"
            echo "  --help            Show this help message"
            echo ""
            echo "Backend Detection:"
            echo "  1. Set MLPERF_BACKEND environment variable (recommended):"
            echo "     export MLPERF_BACKEND=sglang"
            echo "     $0"
            echo ""
            echo "  2. Use --backend command line argument:"
            echo "     $0 --backend sglang"
            echo ""
            echo "  3. Automatic detection (may fail with multiple backends)"
            echo ""
            echo "Backend-specific behavior:"
            echo "  pytorch, vllm, sglang: Virtual environment activated after setup"
            echo ""
            echo "Supported backends: pytorch, vllm, sglang"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== MLPerf DeepSeek Environment Setup ==="

# Check if cleanup requested
if [ "$CLEANUP_VENVS" = "true" ]; then
    cleanup_broken_venvs
    echo ""
fi

# Detect or use override backend
if [ -n "$BACKEND_OVERRIDE" ]; then
    BACKEND="$BACKEND_OVERRIDE"
    echo "Using backend override: $BACKEND"
else
    BACKEND=$(detect_backend)
    if [ $? -eq 0 ]; then
        echo "Detected backend: $BACKEND"
    else
        echo "ERROR: Could not detect backend automatically."
        echo ""
        echo "This usually happens when:"
        echo "  - Multiple backends are installed in the same container"
        echo "  - The container environment doesn't match expected patterns"
        echo "  - Old virtual environments exist from previous runs"
        echo ""
        echo "SOLUTION: Set the MLPERF_BACKEND environment variable:"
        echo "  export MLPERF_BACKEND=sglang  # For SGLang backend"
        echo "  export MLPERF_BACKEND=vllm    # For vLLM backend"
        echo "  export MLPERF_BACKEND=pytorch # For PyTorch backend"
        echo ""
        echo "Or use the --backend option:"
        echo "  $0 --backend sglang"
        echo ""
        echo "To clean up broken virtual environments:"
        echo "  $0 --cleanup-venvs"
        exit 1
    fi
fi

# Validate backend
case "$BACKEND" in
    pytorch|vllm|sglang)
        echo "Setting up for $BACKEND backend..."
        ;;
    *)
        echo "Error: Unknown backend '$BACKEND'"
        echo "Supported backends: pytorch, vllm, sglang"
        exit 1
        ;;
esac

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_SCRIPTS_DIR="$SCRIPT_DIR/docker/setup_scripts"

# Check if setup scripts directory exists
if [ ! -d "$SETUP_SCRIPTS_DIR" ]; then
    echo "Error: Setup scripts directory not found at $SETUP_SCRIPTS_DIR"
    echo "Please ensure the setup_scripts directory exists and contains backend-specific scripts."
    exit 1
fi

# Call the appropriate backend setup script
BACKEND_SCRIPT="$SETUP_SCRIPTS_DIR/setup_$BACKEND.sh"

if [ ! -f "$BACKEND_SCRIPT" ]; then
    echo "Error: Backend setup script not found: $BACKEND_SCRIPT"
    echo "Available setup scripts:"
    ls -1 "$SETUP_SCRIPTS_DIR"/setup_*.sh 2>/dev/null || echo "  None found"
    exit 1
fi

echo "Running backend-specific setup: $BACKEND_SCRIPT"
echo ""

# Make sure the script is executable (only if we have permission to change it)
if [ ! -x "$BACKEND_SCRIPT" ]; then
    if chmod +x "$BACKEND_SCRIPT" 2>/dev/null; then
        echo "Made script executable: $BACKEND_SCRIPT"
    else
        echo "Note: Could not make script executable, but will run with bash"
    fi
fi

# Pass through the force rebuild flag if set
if [ "$FORCE_REBUILD" = "true" ]; then
    bash "$BACKEND_SCRIPT" --force-rebuild
else
    bash "$BACKEND_SCRIPT"
fi

echo ""
echo "=== Main Setup Complete ==="
echo "Backend: $BACKEND"

# Provide activation instructions for backends that use virtual environments
case "$BACKEND" in
    pytorch|vllm|sglang)
        echo ""
        echo "IMPORTANT: To activate the virtual environment, run:"
        echo "   source /work/.venv_${BACKEND}/bin/activate"
        echo ""
        echo ""
        if [ "$BACKEND" = "pytorch" ]; then
            echo "   (.venv_${BACKEND}) torchrun --nproc_per_node=8 run_eval_mpi.py ..."
            echo "   (.venv_${BACKEND}) torchrun --nproc_per_node=8 run_mlperf_mpi.py ..."
        else
            echo "   (.venv_${BACKEND}) python run_eval.py ..."
            echo "   (.venv_${BACKEND}) python run_mlperf.py ..."
        fi
        ;;
esac

echo ""
echo "For usage instructions, see the output above or run:"
echo "  $BACKEND_SCRIPT --help"