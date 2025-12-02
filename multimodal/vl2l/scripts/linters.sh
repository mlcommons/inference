#!/bin/bash

set -eux
set -o pipefail

PROJECT_ROOT=$(dirname "${BASH_SOURCE[0]}")/../
PROJECT_ROOT=$(realpath "${PROJECT_ROOT}")

function _exit_with_help_msg() {
  cat <<EOF
Run linters for "${PROJECT_ROOT}".

Usage: ${BASH_SOURCE[0]}
  [-h | --help]     Print this help message.
EOF
  if [ -n "$1" ]; then
    echo "$(tput bold setab 1)$1$(tput sgr0)"
  fi
  exit "$2"
}

while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --help)
    _exit_with_help_msg "" 0
    ;;
  *)
    _exit_with_help_msg "[ERROR] Unknown option: $1" 1
    ;;
  esac
done

echo "Running ruff..."
ruff check --fix "${PROJECT_ROOT}"/src/

echo "Running mypy..."
mypy --config-file="${PROJECT_ROOT}"/pyproject.toml \
     --install-types \
     "${PROJECT_ROOT}"/src/

echo "Running shellcheck..."
find "${PROJECT_ROOT}" -type f -name "*.sh" -exec shellcheck -ax {} +

echo "Running trufflehog..."
docker run --rm \
           -it \
           -v "${PROJECT_ROOT}":/to-scan \
           trufflesecurity/trufflehog:latest \
           filesystem /to-scan
