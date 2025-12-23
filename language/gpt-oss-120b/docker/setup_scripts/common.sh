#!/bin/bash
# Common setup functions for MLPerf gpt-oss evaluation environment

set -e

check_uv_installed() {
    if ! command -v uv &>/dev/null; then
        echo "Error: uv is not installed"
        exit 1
    fi
}

setup_virtual_environment() {
    local VENV_DIR="$1"
    if [ "$PWD" != "/work" ]; then
        cd /work
    fi
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment exists at $VENV_DIR"
    else
        echo "Creating UV virtual environment..."
        uv venv --system-site-packages "$VENV_DIR"
    fi
}

install_build_dependencies() {
    echo "Installing build dependencies..."
    VIRTUAL_ENV=$VENV_DIR uv pip install numpy setuptools
}

install_mlperf_loadgen() {
    local FORCE_REBUILD="$1"
    local BACKEND="$2"
    echo "=== Installing MLPerf LoadGen for backend: $BACKEND ==="
    
    MLPERF_BUILD_DIR="/inference/loadgen/build"
    WHEEL_DIR="$MLPERF_BUILD_DIR/wheels"
    mkdir -p $MLPERF_BUILD_DIR $WHEEL_DIR
    
    if [ "$FORCE_REBUILD" = "true" ]; then
        echo "Force rebuild - removing cached wheels"
        rm -f $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl
    fi
    
    if ls $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl 1>/dev/null 2>&1; then
        echo "Found cached wheel for $BACKEND"
        WHEEL_FILE=$(ls $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl | head -n1)
        WHEEL_NAME=$(basename "$WHEEL_FILE")
        ORIGINAL_NAME=$(echo "$WHEEL_NAME" | sed "s/_${BACKEND}\.whl$/.whl/")
        TEMP_WHEEL="$WHEEL_DIR/$ORIGINAL_NAME"
        cp "$WHEEL_FILE" "$TEMP_WHEEL"
        pip install --force-reinstall "$TEMP_WHEEL"
        rm -f "$TEMP_WHEEL"
        echo "Installed from cached wheel"
    else
        echo "Building MLPerf LoadGen from source..."
        if [ ! -d "/inference/loadgen" ]; then
            git clone --depth 1 https://github.com/mlcommons/inference.git /inference
        fi
        
        cd /inference/loadgen
        mkdir -p build && cd build
        cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)
        
        cd /inference/loadgen
        VIRTUAL_ENV=$VENV_DIR uv pip install setuptools wheel "pybind11>=2.11.1"
        CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel --dist-dir=$WHEEL_DIR
        
        ORIGINAL_WHEEL=$(ls $WHEEL_DIR/mlcommons_loadgen*.whl | head -n1)
        WHEEL_NAME=$(basename "$ORIGINAL_WHEEL")
        BACKEND_WHEEL="${WHEEL_NAME%.whl}_${BACKEND}.whl"
        mv "$ORIGINAL_WHEEL" "$WHEEL_DIR/$BACKEND_WHEEL"
        
        TEMP_WHEEL="$WHEEL_DIR/${WHEEL_NAME}"
        cp "$WHEEL_DIR/$BACKEND_WHEEL" "$TEMP_WHEEL"
        pip install --force-reinstall "$TEMP_WHEEL"
        rm -f "$TEMP_WHEEL"
        echo "Built and installed MLPerf LoadGen"
    fi
    cd /work
}

print_setup_info() {
    local VENV_DIR="$1"
    local BACKEND="$2"
    echo ""
    echo "=== Setup Complete ==="
    echo "Virtual environment: $VENV_DIR"
    echo "Backend: $BACKEND"
}
