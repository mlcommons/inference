#!/bin/bash
# Common setup functions for MLPerf DeepSeek evaluation environment
# This script contains shared functionality for all backends

set -e # Exit on error

# Common function to check if uv is installed
check_uv_installed() {
	if ! command -v uv &>/dev/null; then
		echo "Error: uv is not installed. Please ensure the Docker image has uv installed."
		exit 1
	fi
}

# Common function to setup virtual environment
setup_virtual_environment() {
	local VENV_DIR="$1"

	# Check if we're already in the /work directory
	if [ "$PWD" != "/work" ]; then
		echo "Changing to /work directory..."
		cd /work
	fi

	# Check if virtual environment already exists
	if [ -d "$VENV_DIR" ]; then
		echo "Virtual environment already exists at $VENV_DIR"
	else
		echo "Creating new UV virtual environment..."
		uv venv --system-site-packages "$VENV_DIR"
	fi
}

# Common function to install build dependencies
install_build_dependencies() {
	echo "Installing build dependencies in venv..."
	VIRTUAL_ENV=$VENV_DIR uv pip install numpy setuptools
}

# Common function to create missing __init__.py files in prm800k
create_prm800k_init_files() {
	echo "Creating missing __init__.py files in prm800k..."

	# List of directories that need __init__.py files
	local PRM800K_DIRS=(
		"/work/submodules/prm800k"
		"/work/submodules/prm800k/prm800k"
		"/work/submodules/prm800k/prm800k/grading"
		"/work/submodules/prm800k/prm800k/eval"
		"/work/submodules/prm800k/prm800k/data"
		"/work/submodules/prm800k/prm800k/instructions"
		"/work/submodules/prm800k/prm800k/math_splits"
	)

	for dir in "${PRM800K_DIRS[@]}"; do
		if [ -d "$dir" ] && [ ! -f "$dir/__init__.py" ]; then
			echo "Creating __init__.py in $dir"
			touch "$dir/__init__.py"
		fi
	done

	echo "Finished creating __init__.py files"
}

# Common function to patch prm800k setup.py
patch_prm800k_setup() {
	echo "Checking and patching prm800k setup.py if necessary..."

	# First, create missing __init__.py files
	create_prm800k_init_files

	PRM800K_SETUP="/work/submodules/prm800k/setup.py"
	if [ -f "$PRM800K_SETUP" ]; then
		echo "Found prm800k setup.py at: $PRM800K_SETUP"
		# Check if the file still has the problematic import
		if grep -q "import numpy" "$PRM800K_SETUP"; then
			echo "Patching prm800k setup.py to fix numpy import issue..."
			# Create a backup
			cp "$PRM800K_SETUP" "$PRM800K_SETUP.bak"
			# Remove the numpy import  line and add it to install_requires
			cat >"$PRM800K_SETUP" <<'EOF'
from setuptools import setup, find_packages

setup(
    name='prm800k',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
        'numpy',
    ],
)
EOF
			echo "Patch applied successfully!"
			echo "New content of setup.py:"
			cat "$PRM800K_SETUP"
		else
			echo "prm800k setup.py already patched or doesn't need patching."
		fi
	else
		echo "WARNING: prm800k setup.py not found at $PRM800K_SETUP"
	fi
}

# Common function to install evaluation requirements
install_evaluation_requirements() {
	echo "Installing evaluation requirements..."
	if [ -f "/work/docker/evaluation_requirements.txt" ]; then
		VIRTUAL_ENV=$VENV_DIR uv pip install -r /work/docker/evaluation_requirements.txt
		echo "Override datasets==3.0.0 (LiveCodeBench/code-generation-lite is not updated for datasets 3.2.0)..."
		VIRTUAL_ENV=$VENV_DIR uv pip install --upgrade "datasets==3.0.0"
		echo "Evaluation requirements installed successfully!"
	else
		echo "Warning: evaluation_requirements.txt not found at /work/docker/evaluation_requirements.txt"
		echo "Please ensure the workspace is properly mounted."
	fi
}

# Common function to check build tools
check_build_tools() {
	echo "Checking if required build tools are available..."
	for tool in cmake make git g++; do
		if ! command -v $tool &>/dev/null; then
			echo "Error: $tool is not installed. MLPerf LoadGen installation cannot proceed."
			echo "Please ensure build dependencies are installed in the Docker image."
			exit 1
		fi
	done
}

# Common function to install MLPerf LoadGen
install_mlperf_loadgen() {
	local FORCE_REBUILD="$1"
	local BACKEND="$2"

	echo ""
	echo "=== Installing MLPerf LoadGen for backend: $BACKEND ==="

	check_build_tools

	# Set build directory
	MLPERF_BUILD_DIR="/inference/loadgen/build"
	WHEEL_DIR="$MLPERF_BUILD_DIR/wheels"

	# Create build directories
	mkdir -p $MLPERF_BUILD_DIR
	mkdir -p $WHEEL_DIR

	# Check if force rebuild is requested
	if [ "$FORCE_REBUILD" = "true" ]; then
		echo "Force rebuild requested. Removing cached wheels for backend $BACKEND..."
		rm -f $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl
	fi

	# Check if we already have a built wheel for this backend
	if ls $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl 1>/dev/null 2>&1; then
		echo "Found existing MLPerf LoadGen wheel for backend $BACKEND in $WHEEL_DIR"

		# Get Python version info for compatibility check
		PYTHON_VERSION=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
		PLATFORM=$(python3 -c "import platform; print(platform.machine())")

		# Check if the wheel is compatible with current Python version
		WHEEL_FILE=$(ls $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl | head -n1)
		WHEEL_NAME=$(basename "$WHEEL_FILE")

		echo "Current Python version tag: $PYTHON_VERSION"
		echo "Current platform: $PLATFORM"
		echo "Found wheel: $WHEEL_NAME"

		# Temporarily rename wheel to remove backend suffix for installation
		ORIGINAL_WHEEL_NAME=$(echo "$WHEEL_NAME" | sed "s/_${BACKEND}\.whl$/.whl/")
		TEMP_WHEEL_PATH="$WHEEL_DIR/$ORIGINAL_WHEEL_NAME"

		echo "Temporarily renaming wheel for installation: $WHEEL_NAME -> $ORIGINAL_WHEEL_NAME"
		cp "$WHEEL_FILE" "$TEMP_WHEEL_PATH"
		pip install --force-reinstall "$TEMP_WHEEL_PATH"

		# Clean up temporary wheel (cached wheel with backend suffix is preserved)
		rm -f "$TEMP_WHEEL_PATH"
		echo "Preserved cached wheel: $WHEEL_NAME"

		# Verify installation
		if python3 -c "import mlperf_loadgen" 2>/dev/null; then
			echo "MLPerf LoadGen installed successfully from cached wheel for backend $BACKEND!"
			return 0
		else
			echo "Installation verification failed. Will rebuild..."
			rm -f $WHEEL_DIR/mlcommons_loadgen*_${BACKEND}.whl
			NEED_BUILD=true
		fi
	else
		echo "No cached wheel found for backend $BACKEND."
		NEED_BUILD=true
	fi

	# Build from source if needed
	if [ "${NEED_BUILD:-true}" = "true" ]; then
		echo "Building MLPerf LoadGen from source for backend $BACKEND..."

		# Check for MLPerf inference repository at /inference, clone if needed
		if [ ! -d "/inference" ] || [ ! -d "/inference/loadgen" ]; then
			echo "MLPerf inference repository not found or incomplete at /inference"
			echo "Cloning MLPerf inference repository from main branch..."

			# Remove existing incomplete directory if it exists
			if [ -d "/inference" ]; then
				echo "Removing incomplete /inference directory..."
				rm -rf /inference
			fi

			# Clone the repository
			echo "Cloning https://github.com/mlcommons/inference.git to /inference..."
			git clone --depth 1 --branch main https://github.com/mlcommons/inference.git /inference

			if [ $? -ne 0 ]; then
				echo "Error: Failed to clone MLPerf inference repository"
				echo "Please check your internet connection or clone manually:"
				echo "  git clone https://github.com/mlcommons/inference.git /inference"
				exit 1
			fi

			echo "Successfully cloned MLPerf inference repository"
		else
			echo "Using existing MLPerf inference repository at /inference"
		fi

		# Navigate to loadgen directory
		LOADGEN_DIR="/inference/loadgen"
		if [ ! -d "$LOADGEN_DIR" ]; then
			echo "Error: loadgen directory not found at $LOADGEN_DIR even after cloning"
			echo "The repository structure may have changed. Please check manually."
			exit 1
		fi

		cd $LOADGEN_DIR

		# Build loadgen C++ library first
		echo "Building loadgen C++ library..."
		mkdir -p build
		cd build
		cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release ..
		make -j$(nproc)

		# Go back to loadgen directory to build wheel
		cd $LOADGEN_DIR

		# Build the wheel using UV to avoid dependency conflicts
		echo "Building MLPerf LoadGen wheel for backend $BACKEND..."
		# First, ensure we have the build dependencies
		VIRTUAL_ENV=$VENV_DIR uv pip install setuptools wheel "pybind11>=2.11.1"

		# Build the wheel using python setup.py directly to avoid pip's dependency resolver conflicts
		CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel --dist-dir=$WHEEL_DIR

		# Rename the wheel to include backend suffix
		ORIGINAL_WHEEL=$(ls $WHEEL_DIR/mlcommons_loadgen*.whl | head -n1)
		if [ -f "$ORIGINAL_WHEEL" ]; then
			WHEEL_NAME=$(basename "$ORIGINAL_WHEEL")
			# Extract the base name and extension
			WHEEL_BASE="${WHEEL_NAME%.whl}"
			BACKEND_WHEEL="${WHEEL_BASE}_${BACKEND}.whl"
			BACKEND_WHEEL_PATH="$WHEEL_DIR/$BACKEND_WHEEL"

			echo "Renaming wheel from $WHEEL_NAME to $BACKEND_WHEEL"
			mv "$ORIGINAL_WHEEL" "$BACKEND_WHEEL_PATH"
			WHEEL_FILE="$BACKEND_WHEEL_PATH"
		else
			echo "Error: No wheel file found after build!"
			exit 1
		fi

		# Install the wheel using pip directly
		echo "Installing MLPerf LoadGen wheel for backend $BACKEND..."

		# Temporarily rename wheel to remove backend suffix for installation
		ORIGINAL_WHEEL_NAME=$(echo "$(basename "$WHEEL_FILE")" | sed "s/_${BACKEND}\.whl$/.whl/")
		TEMP_WHEEL_PATH="$WHEEL_DIR/$ORIGINAL_WHEEL_NAME"

		echo "Temporarily renaming wheel for installation: $(basename "$WHEEL_FILE") -> $ORIGINAL_WHEEL_NAME"
		cp "$WHEEL_FILE" "$TEMP_WHEEL_PATH"

		pip install --force-reinstall "$TEMP_WHEEL_PATH"

		# Clean up temporary wheel (backend-suffixed wheel is preserved for caching)
		rm -f "$TEMP_WHEEL_PATH"
		echo "Preserved cached wheel for future use: $(basename "$WHEEL_FILE")"

		# Verify installation
		if python3 -c "import mlperf_loadgen" 2>/dev/null; then
			echo "MLPerf LoadGen built and installed successfully for backend $BACKEND!"
		else
			echo "Error: MLPerf LoadGen installation verification failed!"
			exit 1
		fi
	fi

	# Return to work directory
	cd /work
}

# Common function to print final setup info
print_setup_info() {
	local VENV_DIR="$1"
	local BACKEND="$2"
	local ACTIVATE_VENV="$3"

	echo ""
	echo "=== Setup completed successfully ==="
	echo "Virtual environment created at: $VENV_DIR"
	echo "Backend: $BACKEND"
	echo ""
	echo "IMPORTANT USAGE NOTES:"
	echo "====================="

	if [ "$ACTIVATE_VENV" = "true" ]; then
		echo "Virtual environment has been activated for this backend."
		echo ""
		echo "For all commands with this backend, the virtual environment is active:"
		case "$BACKEND" in
		"pytorch")
			echo "   (.venv_pytorch) \$ torchrun --nproc_per_node=8 run_eval_mpi.py ..."
			echo "   (.venv_pytorch) \$ torchrun --nproc_per_node=8 run_mlperf_mpi.py ..."
			;;
		"vllm")
			echo "   (.venv_vllm) \$ python run_eval.py ..."
			echo "   (.venv_vllm) \$ python run_mlperf.py ..."
			;;
		"sglang")
			echo "   (.venv_sglang) \$ python run_eval.py ..."
			echo "   (.venv_sglang) \$ python run_mlperf.py ..."
			;;
		esac
		echo "   \$ python eval_accuracy.py ..."
	else
		echo "Virtual environment NOT activated for this backend ($BACKEND)."
		echo ""
		echo "For regular inference and MLPerf runs with this backend:"
		echo "   DO NOT activate the virtual environment. Run directly:"
		echo ""
		echo "For accuracy evaluation ONLY, activate the virtual environment:"
		echo "   \$ source $VENV_DIR/bin/activate"
		echo "   (.venv_BACKEND) \$ python eval_accuracy.py ..."
		echo "   (.venv_BACKEND) \$ deactivate"
	fi

	echo ""
	echo "MLPerf LoadGen installed is installed as system package."
	echo "Build artifacts cached at: /inference/loadgen/build"
	echo "Wheels cached at: /inference/loadgen/build/wheels"
}