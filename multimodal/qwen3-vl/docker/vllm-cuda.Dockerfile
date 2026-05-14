# Start from a vLLM image that was built from 
#   https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
#
# ============================================================================
# USAGE EXAMPLES
# ============================================================================
#
# 1. Install from default git URL (remote):
#      docker build -t myimage .
#
# 2. Install from a different git URL or branch:
#      docker build --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL=git+https://github.com/USER/REPO.git@BRANCH#subdirectory=multimodal/qwen3-vl \
#                   -t myimage .
#
# 3. Install from local directory (build from repo root with git auto-detection):
#    (Version number will be auto-detected from git if the build context includes .git)
#      docker build --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL=multimodal/qwen3-vl \
#                   -f multimodal/qwen3-vl/docker/vllm-cuda.Dockerfile \
#                   -t myimage .
#
# 4. Install from local directory (build from multimodal/qwen3-vl subdirectory):
#    (No .git in subdirectory, will use fallback version "0.0.0.dev0")
#      docker build --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL=. \
#                   -f multimodal/qwen3-vl/docker/vllm-cuda.Dockerfile \
#                   -t myimage multimodal/qwen3-vl
#
# 5. Install from local directory when pwd is already multimodal/qwen3-vl:
#    (No .git in subdirectory, will use fallback version "0.0.0.dev0")
#      cd multimodal/qwen3-vl
#      docker build --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL=. \
#                   -f docker/vllm-cuda.Dockerfile \
#                   -t myimage .
#
# 6. Install from local directory with a custom fallback version:
#    (Override the default "0.0.0.dev0" version when git is not available)
#      cd multimodal/qwen3-vl
#      docker build --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL=. \
#                   --build-arg MLPERF_INF_MM_Q3VL_VERSION=1.0.0 \
#                   -f docker/vllm-cuda.Dockerfile \
#                   -t myimage .
#
# 7. Use a custom vLLM base image:
#      docker build --build-arg BASE_IMAGE_URL=my-custom-vllm:latest \
#                   -t myimage .
#
# ============================================================================

ARG BASE_IMAGE_URL=vllm/vllm-openai:v0.12.0
FROM ${BASE_IMAGE_URL}

# MLPERF_INF_MM_Q3VL_INSTALL_URL can be either:
#   1. A git URL (default): git+https://github.com/...
#   2. A local directory path relative to the build context (e.g., multimodal/qwen3-vl)
#      Note: The build context is the directory you pass to `docker build` (the final arg).
#            MLPERF_INF_MM_Q3VL_INSTALL_URL must be a valid path inside that build context.
ARG MLPERF_INF_MM_Q3VL_INSTALL_URL=git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl

# LOADGEN_INSTALL_URL can be:
#   1. A local directory path relative to the build context (e.g., loadgen/). Note: The
#      build context is the directory you pass to `docker build` (the final arg).
#      LOADGEN_INSTALL_URL must be a valid path inside that build context.
#   2. A git URL: git+https://github.com/...
#   3. If empty (default), the loadgen in the dependencies of the pyproject.toml will
#      be used (i.e., from PyPI).
ARG LOADGEN_INSTALL_URL=""

# Temporary directory inside the container where the build context will be copied
# Only used when installing from a local directory path
ARG BUILD_CONTEXT_DIR=/tmp/mm_q3vl_build_context

# Fallback version to use when building from local directory without git metadata
# setuptools-scm will first try to detect version from .git, and use this as fallback
# Must be a valid PEP 440 version string (e.g., "0.0.0.dev0", "1.0.0", "0.1.0.dev1")
# Can be overridden at build time with --build-arg
ARG MLPERF_INF_MM_Q3VL_VERSION=0.0.0.dev0

# Install
# - git (required for installing "git+..." dependencies to work)
# - tmux (for `vllm serve` and `mlperf-inf-mm-q3vl` in different tmux sessions)
# - vim (for editing files in the container)
RUN apt-get update && \
    apt-get install -y git tmux vim && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables.
# Setting LD_LIBRARY_PATH here ensures it works during the build process
# and persists when you run the container later.
#ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Copy build context.
# This will be used only if MLPERF_INF_MM_Q3VL_INSTALL_URL is a local path.
COPY . ${BUILD_CONTEXT_DIR}/

# Install the mlcommons-loadgen package if LOADGEN_INSTALL_URL is not empty.
RUN if [ -n "${LOADGEN_INSTALL_URL}" ]; then \
        if echo "${LOADGEN_INSTALL_URL}" | grep -q "^git+"; then \
            echo "Installing from git URL: ${LOADGEN_INSTALL_URL}"; \
            uv pip install --system --no-cache --verbose "${LOADGEN_INSTALL_URL}"; \
        else \
            echo "Installing from local path: ${LOADGEN_INSTALL_URL}"; \
            uv pip install --system --no-cache --verbose "${BUILD_CONTEXT_DIR}/${LOADGEN_INSTALL_URL}"; \
        fi; \
    else \
        echo "Using mlcommons-loadgen from dependencies of pyproject.toml (i.e., from PyPI)"; \
    fi;

# Install the mlperf-inference-multimodal-q3vl package.
# We use --system to install into the container's global python environment.
# Detect if MLPERF_INF_MM_Q3VL_INSTALL_URL is a git URL or a local path:
RUN if echo "${MLPERF_INF_MM_Q3VL_INSTALL_URL}" | grep -q "^git+"; then \
        echo "Installing from git URL: ${MLPERF_INF_MM_Q3VL_INSTALL_URL}"; \
        uv pip install --system --no-cache --verbose "${MLPERF_INF_MM_Q3VL_INSTALL_URL}"; \
    else \
        echo "Installing from local path: ${MLPERF_INF_MM_Q3VL_INSTALL_URL}"; \
        # Check if the package directory is inside a git repository \
        if cd "${BUILD_CONTEXT_DIR}/${MLPERF_INF_MM_Q3VL_INSTALL_URL}" && git rev-parse --git-dir > /dev/null 2>&1; then \
            echo "Git repository detected, setuptools-scm will detect version automatically"; \
        else \
            echo "Not in a git repository, using fallback version: ${MLPERF_INF_MM_Q3VL_VERSION}"; \
            export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MLPERF_INF_MM_Q3VL="${MLPERF_INF_MM_Q3VL_VERSION}"; \
        fi; \
        uv pip install --system --no-cache --verbose "${BUILD_CONTEXT_DIR}/${MLPERF_INF_MM_Q3VL_INSTALL_URL}"; \
    fi && \
    rm -rf "${BUILD_CONTEXT_DIR}"

# Set the entrypoint to bash so it opens a shell by default
ENTRYPOINT ["/bin/bash"]