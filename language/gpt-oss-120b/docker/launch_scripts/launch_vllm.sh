#!/bin/bash
# vLLM Docker Launch Script

set -e

MOUNT_SELF_AS_LOADGEN=${MOUNT_SELF_AS_LOADGEN:-true}
IMAGE_NAME=${IMAGE_NAME:-mlperf-gptoss-vllm}
IMAGE_TAG=${IMAGE_TAG:-latest}
LOCAL_USER=${LOCAL_USER:-1}
WORK_DIR=${WORK_DIR:-$(dirname "$(realpath "$0")")/../..}
CONTAINER_NAME=${CONTAINER_NAME:-gptoss-vllm}
RUN_CMD=${RUN_CMD:-}

USER_ID=$(id --user)
USER_NAME=$(id --user --name)
GROUP_ID=$(id --group)
GROUP_NAME=$(id --group --name)

if [ "$LOCAL_USER" = "1" ]; then
    IMAGE_TAG_SUFFIX="-$USER_NAME"
else
    IMAGE_TAG_SUFFIX=""
fi

BASE_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
FINAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}${IMAGE_TAG_SUFFIX}"

RANDOM_NUM=$RANDOM
CCACHE_DIR=${CCACHE_DIR:-${WORK_DIR}/.ccache}

DOCKER_RUN_OPTS="--rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
GPU_OPTS="--gpus=all"

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
            [ "$2" != "all" ] && GPU_OPTS="--gpus=$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --image-name NAME        Docker image name"
            echo "  --container-name NAME    Container name"
            echo "  --gpu-count COUNT        GPU count"
            echo "  --help                   Show help"
            exit 0
            ;;
        *)
            RUN_CMD="$RUN_CMD $1"
            shift
            ;;
    esac
done

echo "Launching vLLM Docker container..."
echo "Base image: $BASE_IMAGE"
echo "Work directory: $WORK_DIR"

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Build base image
docker build -f "$SCRIPT_DIR/../Dockerfile.vllm" -t "$BASE_IMAGE" "$SCRIPT_DIR/../.."

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker image"
    exit 1
fi

# Build user-specific image if needed
if [ "$LOCAL_USER" = "1" ]; then
    echo "Building user-specific image..."
    USER_DOCKERFILE="$SCRIPT_DIR/Dockerfile.user.tmp"
    cat > "$USER_DOCKERFILE" << INNEREOF
ARG BASE_IMAGE_WITH_TAG
FROM \${BASE_IMAGE_WITH_TAG}
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
RUN pip install --user ipython jupyter
INNEREOF
    
    docker build \
        --build-arg BASE_IMAGE_WITH_TAG="$BASE_IMAGE" \
        --build-arg USER_ID="$USER_ID" \
        --build-arg USER_NAME="$USER_NAME" \
        --build-arg GROUP_ID="$GROUP_ID" \
        --build-arg GROUP_NAME="$GROUP_NAME" \
        --file "$USER_DOCKERFILE" \
        --tag "$FINAL_IMAGE" \
        "$SCRIPT_DIR"
    
    rm -f "$USER_DOCKERFILE"
else
    FINAL_IMAGE="$BASE_IMAGE"
fi

# Setup inference mount
INFERENCE_MOUNT=""
if [ "$MOUNT_SELF_AS_LOADGEN" = "true" ]; then
    INFERENCE_ROOT="$(dirname "$(realpath "$0")")/../../../../"
    INFERENCE_MOUNT="-v ${INFERENCE_ROOT}:/inference"
fi

# Run container
docker run $DOCKER_RUN_OPTS \
    $GPU_OPTS \
    -v /home/mlperf_inference_storage:/home/mlperf_inference_storage \
    -v /raid/data:/raid/data \
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
