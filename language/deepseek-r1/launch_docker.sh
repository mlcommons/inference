#!/bin/bash

# MLPerf DeepSeek Reference Implementation Docker Launcher
# Top-level script to launch different backend containers

set -e

# Default values
BACKEND=""
WORK_DIR=${WORK_DIR:-$(dirname "$(realpath "$0")")}
SCRIPT_DIR="$WORK_DIR/docker/launch_scripts"

# Available backends
AVAILABLE_BACKENDS=("pytorch" "vllm" "sglang")

# Additional mount directories (can be customized)
EXTRA_MOUNTS=${EXTRA_MOUNTS:-""}

# Function to display usage
show_usage() {
    echo "Usage: $0 --backend <backend> [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --backend BACKEND        Backend to use: pytorch, vllm, sglang"
    echo ""
    echo "Options:"
    echo "  --image-name NAME        Docker image name"
    echo "  --container-name NAME    Container name"
    echo "  --gpu-count COUNT        GPU count (default: all)"
    echo "  --model-cache-dir DIR    Model cache directory"
    echo "  --local-user 0|1         Enable local user setup (default: 1)"
    echo "  --extra-mounts MOUNTS    Additional mount directories (format: 'host1:container1,host2:container2')"
    echo "  --help                   Show this help message"
    echo ""
    echo "Available backends:"
    for backend in "${AVAILABLE_BACKENDS[@]}"; do
        echo "  - $backend"
    done
    echo ""
    echo "Examples:"
    echo "  $0 --backend pytorch"
    echo "  $0 --backend vllm --gpu-count 2"
    echo "  $0 --backend sglang --extra-mounts '/data:/data,/models:/models'"
}

# Function to check if backend is valid
is_valid_backend() {
    local backend="$1"
    for valid_backend in "${AVAILABLE_BACKENDS[@]}"; do
        if [[ "$backend" == "$valid_backend" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to get launch script path
get_launch_script() {
    local backend="$1"
    echo "$SCRIPT_DIR/launch_${backend}.sh"
}

# Parse command line arguments
BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"
            if ! is_valid_backend "$BACKEND"; then
                echo "Error: Invalid backend '$BACKEND'"
                echo "Available backends: ${AVAILABLE_BACKENDS[*]}"
                exit 1
            fi
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        --image-name|--container-name|--gpu-count|--model-cache-dir|--local-user|--extra-mounts)
            # Pass these arguments to the backend script
            BACKEND_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            # Pass any other arguments to the backend script
            BACKEND_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if backend was specified
if [[ -z "$BACKEND" ]]; then
    echo "Error: --backend is required"
    echo ""
    show_usage
    exit 1
fi

# Get the launch script for the specified backend
LAUNCH_SCRIPT=$(get_launch_script "$BACKEND")

# Check if launch script exists
if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "Error: Launch script not found: $LAUNCH_SCRIPT"
    exit 1
fi

# Make sure the launch script is executable
chmod +x "$LAUNCH_SCRIPT"

echo "MLPerf DeepSeek Reference Implementation Docker Launcher"
echo "======================================================="
echo "Backend: $BACKEND"
echo "Launch script: $LAUNCH_SCRIPT"
echo "Work directory: $WORK_DIR"

# Set work directory environment variable
export WORK_DIR

echo "======================================================="
echo ""

# Execute the backend-specific launch script with all arguments
exec "$LAUNCH_SCRIPT" "${BACKEND_ARGS[@]}"