#!/bin/bash
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# Start one of the benchmark's vLLM servers.
#
# Usage (from e2e-rag/):
#   [DEVICES=<ids>] bash scripts/launch_llm_server.sh <query|grader|judge> [extra vllm serve args...]
#
#   query   gpt-oss-120b: query generation, sufficiency check, final answer
#   grader  gpt-oss-20b:  document grading
#   judge   Llama-3.1-8B: answer scoring (accuracy runs only)
#
# The model comes from <ROLE>_MODEL and the port from <ROLE>_SERVICE_URL, the
# same variables the run scripts read, so client and server always match.
#
# DEVICES picks the accelerators; the script detects the vendor and exports
# the matching variable (VENDOR=nvidia|amd|intel|cpu overrides detection):
#   nvidia  CUDA_VISIBLE_DEVICES     amd  HIP_VISIBLE_DEVICES
#   intel   ZE_AFFINITY_MASK         cpu  (DEVICES ignored)
# Without -tp/--tensor-parallel-size, tensor parallelism = number of DEVICES.
#   DEVICES=0,1,2,3 bash scripts/launch_llm_server.sh query
#   DEVICES=4,5     bash scripts/launch_llm_server.sh grader
#   DEVICES=6       bash scripts/launch_llm_server.sh judge

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Settings: env var > config.sh (optional) > config.default.sh
set -a
CONFIG="${CONFIG:-config.sh}"
if [ -f "${CONFIG}" ]; then
    source "${CONFIG}"
fi
source config.default.sh
set +a

ROLE="${1:-}"
shift || true

case "${ROLE}" in
    query)
        MODEL="${QUERY_MODEL}"
        URL="${QUERY_SERVICE_URL}" ;;
    grader)
        MODEL="${GRADER_MODEL}"
        URL="${GRADER_SERVICE_URL}" ;;
    judge)
        MODEL="${JUDGE_MODEL}"
        URL="${JUDGE_SERVICE_URL}" ;;
    *)
        echo "Usage: bash scripts/launch_llm_server.sh <query|grader|judge> [extra vllm serve args...]" >&2
        exit 1 ;;
esac

PORT=$(echo "${URL}" | sed -E 's#^[a-z]+://[^/:]+:([0-9]+).*#\1#')
if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: no port in ${ROLE} URL: ${URL}" >&2
    exit 1
fi

detect_vendor() {
    if command -v nvidia-smi >/dev/null 2>&1; then echo nvidia
    elif command -v amd-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then echo amd
    elif command -v xpu-smi >/dev/null 2>&1 || command -v sycl-ls >/dev/null 2>&1; then echo intel
    else echo cpu
    fi
}
VENDOR="${VENDOR:-$(detect_vendor)}"

DEVICE_INFO="${VENDOR}: all devices"
if [[ -n "${DEVICES:-}" ]]; then
    case "${VENDOR}" in
        nvidia) export CUDA_VISIBLE_DEVICES="${DEVICES}"; DEVICE_VAR=CUDA_VISIBLE_DEVICES ;;
        amd)    export HIP_VISIBLE_DEVICES="${DEVICES}";  DEVICE_VAR=HIP_VISIBLE_DEVICES ;;
        intel)  export ZE_AFFINITY_MASK="${DEVICES}";     DEVICE_VAR=ZE_AFFINITY_MASK ;;
        cpu)    DEVICE_VAR="" ;;
        *) echo "ERROR: VENDOR must be nvidia, amd, intel or cpu (got '${VENDOR}')" >&2; exit 1 ;;
    esac
    if [[ -n "${DEVICE_VAR}" ]]; then
        DEVICE_INFO="${VENDOR}: ${DEVICE_VAR}=${DEVICES}"
        # Tensor parallelism defaults to the number of devices.
        if [[ " $* " != *" -tp "* && " $* " != *" --tensor-parallel-size"* ]]; then
            N_DEVICES=$(echo "${DEVICES}" | tr ',' '\n' | grep -c .)
            set -- "$@" --tensor-parallel-size "${N_DEVICES}"
        fi
    else
        DEVICE_INFO="cpu: DEVICES ignored"
    fi
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/vllm_${ROLE}.log"

echo "[${ROLE}] ${DEVICE_INFO}"
echo "[${ROLE}] vllm serve ${MODEL} --port ${PORT} $* -> ${LOG}"
nohup vllm serve "${MODEL}" --port "${PORT}" "$@" > "${LOG}" 2>&1 &
PID=$!

# Model load can take several minutes for the 120B.
TIMEOUT="${TIMEOUT:-1800}"
for ((i = 0; i < TIMEOUT; i += 5)); do
    if curl -sf -o /dev/null --noproxy '*' "http://127.0.0.1:${PORT}/v1/models"; then
        echo "[${ROLE}] ready on port ${PORT} (pid ${PID})"
        exit 0
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "ERROR: [${ROLE}] server exited; see ${LOG}" >&2
        exit 1
    fi
    sleep 5
done
echo "ERROR: [${ROLE}] not ready after ${TIMEOUT}s; see ${LOG}" >&2
exit 1
