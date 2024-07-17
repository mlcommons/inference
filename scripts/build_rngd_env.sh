#!/bin/bash

# Install Rust https://rustup.rs/
install_rust() {
    if ! command -v rustup &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    else
        echo "Rust is already installed"
    fi
}

# Install necessary packages
# https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#a13025628209440c917a424ea173efe5
install_packages() {
    echo "Updating package list and installing packages..."
    apt update && apt install -y \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        clang \
        clang-format-11 \
        clang-tidy \
        curl \
        dpkg-dev \
        fakeroot \
        gcc-aarch64-linux-gnu \
        g++ \
        git \
        gnupg \
        libboost-dev \
        libboost-regex-dev \
        libc6-dev-arm64-cross \
        libcapstone-dev \
        libelf-dev \
        libssl-dev \
        libtbb-dev \
        libtool \
        libyaml-cpp-dev \
        lsb-release \
        pkg-config \
        python3-dev \
        rpm \
        rsync \
        software-properties-common \
        unzip \
        uuid-dev \
        wget
}

# https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#f29089b7b13842f097fcd27bb04a7e8e
# protoc
install_protoc() {
    local PROTOC_VERSION=22.0
    echo "Installing Protobuf compiler version ${PROTOC_VERSION}..."
    curl -Lo protoc.zip "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip"
    unzip -q protoc.zip bin/protoc -d /usr/local
    chmod a+x /usr/local/bin/protoc
    rm protoc.zip
}

# cmake
install_cmake() {
    local CMAKE_VERSION=3.26.3
    echo "Installing CMake version ${CMAKE_VERSION}..."
    curl -Lo cmake.tar.gz "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
    tar xzvf cmake.tar.gz -C /usr/local --strip-components=1
    rm cmake.tar.gz
}

# Install Miniconda
install_miniconda() {
    if ! command -v conda &> /dev/null; then
        echo "Installing Miniconda..."
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh
        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        source ~/.bashrc
    else
        echo "Miniconda is already installed"
    fi
}

# https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#85365fb447894f8f9952a824c135c197
setup_conda_env() {
    local ENV_NAME=llm
    echo "Creating Conda environment ${ENV_NAME}..."
    conda create -n ${ENV_NAME} python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
}

# https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#02c925d7dc3348be8e30ad6d102bafa3
install_dvc() {
    echo "Installing DVC with S3 support..."
    pip install 'dvc[s3]'
}

# https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#847dd27d9857405a8b3c26b4538c3656
setup_furiosa_runtime() {
    local COMMIT_HASH=$1
    if [ ! -d "furiosa-runtime" ]; then
        echo "Cloning furiosa-runtime repository..."
        git submodule add https://github.com/furiosa-ai/furiosa-runtime.git
    fi
    cd furiosa-runtime
    git submodule update --init --recursive
    git checkout ${COMMIT_HASH}
    (cd furiosa-llm-models-artifacts && dvc pull -r origin)
}

# https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#fe8f53d80d464606a87bb342211b5665
install_furiosa_llm() {
    # https://furiosa-ai.slack.com/archives/C05G3K8BVG8/p1721177782811749?thread_ts=1721145137.454689&cid=C05G3K8BVG8
    export CARGO_NET_GIT_FETCH_WITH_CLI=true
    # https://www.notion.so/furiosa/RNGD-SW-Stack-runtime-7df73fb4d92241e09a2721612ebd9c3d?pvs=4#fe8f53d80d464606a87bb342211b5665
    echo "Installing Furiosa runtime dependencies..."
    make install-python-deps
    make install-llm-full
}

main() {
    local DEFAULT_COMMIT_HASH="ea5957c2929151fe73c263269b5df38b11e324a4"
    local COMMIT_HASH="${1:-$DEFAULT_COMMIT_HASH}"

    install_rust
    install_packages
    install_protoc
    install_cmake
    setup_conda_env
    install_dvc
    setup_furiosa_runtime ${COMMIT_HASH}
    install_furiosa_llm
    echo "Setup complete."
}

main "$@"
