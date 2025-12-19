#!/bin/bash

# vLLM Docker Launch Script
# Replicates the functionality of launch_docker.sh for vLLM backend

# Set default values (can be overridden by environment variables)
MOUNT_SELF_AS_LOADGEN=${MOUNT_SELF_AS_LOADGEN:-true}
IMAGE_NAME=${IMAGE_NAME:-mlperf-vllm}
IMAGE_TAG=${IMAGE_TAG:-latest}
LOCAL_USER=${LOCAL_USER:-1}
WORK_DIR=${WORK_DIR:-$(dirname "$(realpath "$0")")/../..}
CONTAINER_NAME=${CONTAINER_NAME:-vllm}
RUN_CMD=${RUN_CMD:-}

# Additional mount directories (can be customized)
EXTRA_MOUNTS=${EXTRA_MOUNTS:-""}

# Get user information
USER_ID=$(id --user)
USER_NAME=$(id --user --name)
GROUP_ID=$(id --group)
GROUP_NAME=$(id --group --name)

# Set image tag suffix for local user
if [ "$LOCAL_USER" = "1" ]; then
    IMAGE_TAG_SUFFIX="-$USER_NAME"
else
    IMAGE_TAG_SUFFIX=""
fi

# Base and final image names
BASE_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
FINAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}${IMAGE_TAG_SUFFIX}"

# Generate random number for container name
RANDOM_NUM=$RANDOM

# Set ccache directory
CCACHE_DIR=${CCACHE_DIR:-${WORK_DIR}/.ccache}

# Docker run options
DOCKER_RUN_OPTS="--rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
DOCKER_RUN_ARGS=""
GPU_OPTS="--gpus=all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-name)
            IMAGE_NAME="$2"
            BASE_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
            FINAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}${IMAGE_TAG_SUFFIX}"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --gpu-count)
            if [ "$2" != "all" ]; then
                GPU_OPTS="--gpus=$2"
            fi
            shift 2
            ;;
        --model-cache-dir)
            MODEL_CACHE_DIR="$2"
            shift 2
            ;;
        --local-user)
            LOCAL_USER="$2"
            if [ "$LOCAL_USER" = "1" ]; then
                IMAGE_TAG_SUFFIX="-$USER_NAME"
            else
                IMAGE_TAG_SUFFIX=""
            fi
            FINAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}${IMAGE_TAG_SUFFIX}"
            shift 2
            ;;
        --extra-mounts)
            EXTRA_MOUNTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --image-name NAME        Docker image name (default: mlperf-vllm)"
            echo "  --container-name NAME    Container name (default: vllm)"
            echo "  --gpu-count COUNT        GPU count (default: all)"
            echo "  --model-cache-dir DIR    Model cache directory"
            echo "  --local-user 0|1         Enable local user setup (default: 1)"
            echo "  --extra-mounts MOUNTS    Additional mount directories (format: 'host1:container1,host2:container2')"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            # Store remaining arguments for passing to container
            RUN_CMD="$RUN_CMD $1"
            shift
            ;;
    esac
done

echo "Launching vLLM Docker container..."
echo "Base image: $BASE_IMAGE"
echo "Final image: $FINAL_IMAGE"
echo "Work directory: $WORK_DIR"
echo "Container name: ${CONTAINER_NAME}-${RANDOM_NUM}-${USER_NAME}"
echo "User: $USER_NAME (UID: $USER_ID)"

# Build the base Docker image
echo "Building base Docker image..."
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Build the base image using the Dockerfile for vLLM
docker build -f "$SCRIPT_DIR/../Dockerfile.vllm" -t "$BASE_IMAGE" "$SCRIPT_DIR/../.."

# Check if base build was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to build base Docker image"
    exit 1
fi

echo "Successfully built base Docker image: $BASE_IMAGE"

# Create user-specific image if LOCAL_USER is enabled
if [ "$LOCAL_USER" = "1" ]; then
    echo "Building user-specific image with user: $USER_NAME"
    
    # Create a temporary Dockerfile.user
    USER_DOCKERFILE="$SCRIPT_DIR/Dockerfile.user.tmp"
    cat > "$USER_DOCKERFILE" << EOF
ARG BASE_IMAGE_WITH_TAG

FROM \${BASE_IMAGE_WITH_TAG} as base

# Alternative user
ARG USER_ID=0
ARG USER_NAME=root
ARG GROUP_ID=0
ARG GROUP_NAME=root

ENV PATH="\$PATH:/home/\${USER_NAME}/.local/bin"
RUN (getent group \${GROUP_ID} || groupadd --gid \${GROUP_ID} \${GROUP_NAME}) && \\
    (getent passwd \${USER_ID} || useradd --gid \${GROUP_ID} --uid \${USER_ID} --create-home --no-log-init --shell /bin/bash \${USER_NAME})

RUN adduser \${USER_NAME} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER \${USER_NAME}

# Install basic user utilities
RUN pip install --user --no-cache-dir ipython jupyter

EOF

    # Build user-specific image
    docker build \
        --build-arg BASE_IMAGE_WITH_TAG="$BASE_IMAGE" \
        --build-arg USER_ID="$USER_ID" \
        --build-arg USER_NAME="$USER_NAME" \
        --build-arg GROUP_ID="$GROUP_ID" \
        --build-arg GROUP_NAME="$GROUP_NAME" \
        --file "$USER_DOCKERFILE" \
        --tag "$FINAL_IMAGE" \
        "$SCRIPT_DIR"
    
    # Clean up temporary Dockerfile
    rm -f "$USER_DOCKERFILE"
    
    # Check if user build was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build user-specific Docker image"
        exit 1
    fi
    
    echo "Successfully built user-specific image: $FINAL_IMAGE"
else
    FINAL_IMAGE="$BASE_IMAGE"
fi

# Check if final image exists
if ! docker image inspect "$FINAL_IMAGE" >/dev/null 2>&1; then
    echo "Error: Docker image '$FINAL_IMAGE' not found."
    echo "Please check the build process above."
    exit 1
fi

echo "Starting container with image: $FINAL_IMAGE"

# Process extra mounts if specified
EXTRA_MOUNT_OPTS=""
if [[ -n "$EXTRA_MOUNTS" ]]; then
    echo "Processing extra mounts: $EXTRA_MOUNTS"
    # Split the comma-separated mount pairs
    IFS=',' read -ra MOUNT_PAIRS <<< "$EXTRA_MOUNTS"
    for mount_pair in "${MOUNT_PAIRS[@]}"; do
        # Trim whitespace
        mount_pair=$(echo "$mount_pair" | xargs)
        if [[ -n "$mount_pair" ]]; then
            # Check if the mount pair contains a colon
            if [[ "$mount_pair" == *":"* ]]; then
                EXTRA_MOUNT_OPTS="$EXTRA_MOUNT_OPTS -v $mount_pair"
                echo "  Adding mount: $mount_pair"
            else
                echo "  Warning: Invalid mount format '$mount_pair' (expected 'host:container')"
            fi
        fi
    done
fi

# Prepare inference mount if MOUNT_SELF_AS_LOADGEN is true
INFERENCE_MOUNT=""
if [ "$MOUNT_SELF_AS_LOADGEN" = "true" ]; then
    INFERENCE_ROOT="$(dirname "$(realpath "$0")")/../../../../"
    INFERENCE_MOUNT="-v ${INFERENCE_ROOT}:/inference"
    echo "Mounting self to /inference: ${INFERENCE_ROOT} -> /inference"
else
    echo "Cloning loadgen repository to /tmp/inference (MOUNT_SELF_AS_LOADGEN=false)"
    # Clone loadgen repository if it doesn't exist
    INFERENCE_TMP="/tmp/inference-${USER_NAME}-${RANDOM_NUM}"
    if [ ! -d "$INFERENCE_TMP" ]; then
        echo "Cloning MLPerf Inference repository..."
        git clone --depth 1 https://github.com/mlcommons/inference.git "$INFERENCE_TMP"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to clone MLPerf Inference repository"
            exit 1
        fi
        echo "Successfully cloned loadgen repository to: $INFERENCE_TMP"
    else
        echo "Using existing loadgen repository at: $INFERENCE_TMP"
    fi
    INFERENCE_MOUNT="-v ${INFERENCE_TMP}:/inference"
fi

# Setup model cache directory mount
# If --model-cache-dir is provided, mount it to /raid/data/$USER/
# If not provided, mount /raid/data/$USER/ from host
if [ -n "$MODEL_CACHE_DIR" ]; then
    MODEL_CACHE_MOUNT="-v ${MODEL_CACHE_DIR}:/raid/data/${USER_NAME}"
    echo "Model cache directory: ${MODEL_CACHE_DIR} -> /raid/data/${USER_NAME}"
else
    MODEL_CACHE_MOUNT="-v /raid/data/${USER_NAME}:/raid/data/${USER_NAME}"
    echo "Model cache directory: /raid/data/${USER_NAME} (host) -> /raid/data/${USER_NAME} (container)"
fi

# Run the Docker container with all mounts (same as main docker setup)
docker run $DOCKER_RUN_OPTS $DOCKER_RUN_ARGS \
    $GPU_OPTS \
    -v /home/mlperf_inference_storage:/home/mlperf_inference_storage \
    $MODEL_CACHE_MOUNT \
    $EXTRA_MOUNT_OPTS \
    -e HISTFILE="${WORK_DIR}/.bash_history" \
    --env "CCACHE_DIR=${CCACHE_DIR}" \
    --env "USER=${USER_NAME}" \
    --env "HF_HOME=/raid/data/${USER_NAME}/.cache" \
    --env "HF_HUB_CACHE=/raid/data/${USER_NAME}/.cache" \
    --env "HUGGINGFACE_HUB_CACHE=/raid/data/${USER_NAME}/.cache" \
    --workdir /work \
    -v "${WORK_DIR}:/work" \
    $INFERENCE_MOUNT \
    --hostname "$(hostname)-docker" \
    --name "${CONTAINER_NAME}-${RANDOM_NUM}-${USER_NAME}" \
    --tmpfs /tmp:exec \
    "$FINAL_IMAGE" $RUN_CMD 