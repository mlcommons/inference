#!/bin/bash

# MLPerf Text-to-Video Benchmark Docker Launcher
set -e

# Default values
IMAGE_NAME=${IMAGE_NAME:-text2video-mlperf}
IMAGE_TAG=${IMAGE_TAG:-latest}
CONTAINER_NAME=${CONTAINER_NAME:-text2video}
GPU_IDS=${GPU_IDS:-all}
WORK_DIR=$(dirname "$(realpath "$0")")
INFERENCE_ROOT=$(dirname "$WORK_DIR")

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  --build              Build the Docker image"
    echo "  --image-name NAME    Docker image name (default: text2video-mlperf)"
    echo "  --container-name NAME Container name (default: text2video)"
    echo "  --gpu-count COUNT    GPU count (default: all)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --build                    # Build image"
    echo "  $0                            # Run interactive shell"
    echo "  $0 python3 script.py          # Run specific command"
}

# Parse arguments
BUILD_ONLY=false
RUN_CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_ONLY=true
            shift
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_IDS="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            # Remaining args are the command to run
            RUN_CMD="$@"
            break
            ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

USER_ID=$(id -u)
GROUP_ID=$(id -g)
USER_NAME=$(id -un)

# Function to build Docker image
build_image() {
    echo "Building Docker image: $FULL_IMAGE"
    echo "User: $USER_NAME (UID: $USER_ID, GID: $GROUP_ID)"
    docker build \
        --build-arg USER_ID=$USER_ID \
        --build-arg GROUP_ID=$GROUP_ID \
        --build-arg USER_NAME=$USER_NAME \
        -f "$WORK_DIR/docker/Dockerfile" \
        -t "$FULL_IMAGE" \
        "$INFERENCE_ROOT"
    
    if [ $? -eq 0 ]; then
        echo "Successfully built image: $FULL_IMAGE"
        return 0
    else
        echo "Error: Failed to build Docker image"
        return 1
    fi
}

if [ "$BUILD_ONLY" = true ]; then
    build_image || exit 1
    exit 0
fi

if ! docker image inspect "$FULL_IMAGE" >/dev/null 2>&1; then
    echo "Image '$FULL_IMAGE' not found. Building..."
    build_image
fi

# Prepare GPU options
if [ "$GPU_IDS" = "all" ]; then
    GPU_OPTS="--gpus=all"
else
    GPU_OPTS="--gpus=$GPU_IDS"
fi

# Run container
echo "Starting container: $CONTAINER_NAME"
echo "Image: $FULL_IMAGE"
echo "Work directory: $INFERENCE_ROOT"

# If no command specified, use bash
if [ -z "$RUN_CMD" ]; then
    RUN_CMD="/bin/bash"
fi

# Determine if we should use interactive mode
DOCKER_FLAGS="--rm -i"
if [ -t 0 ]; then
    DOCKER_FLAGS="--rm -it"
fi

docker run $DOCKER_FLAGS \
    --name "$CONTAINER_NAME" \
    --hostname "$(hostname)-docker" \
    $GPU_OPTS \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=32g \
    -v "$INFERENCE_ROOT:/workspace" \
    -v "/home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch" \
    -w /workspace/wan2.2-t2v-14b \
    "$FULL_IMAGE" \
    $RUN_CMD
