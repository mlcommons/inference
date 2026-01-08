#!/bin/bash

set -eux
set -o pipefail

DEFAULT_CONTAINER_IMAGE=""
container_image=${DEFAULT_CONTAINER_IMAGE}

DEFAULT_MODEL_REPO_ID=Qwen/Qwen3-VL-235B-A22B-Instruct
model_repo_id=${DEFAULT_MODEL_REPO_ID}

DEFAULT_SCENARIO=offline
scenario=${DEFAULT_SCENARIO}

DEFAULT_MODE=accuracy_only
mode=${DEFAULT_MODE}

DEFAULT_SERVER_TARGET_QPS=5
server_target_qps=${DEFAULT_SERVER_TARGET_QPS}

DEFAULT_TENSOR_PARALLEL_SIZE=8
tensor_parallel_size=${DEFAULT_TENSOR_PARALLEL_SIZE}

DEFAULT_CACHE_HOST_DIR=""
cache_host_dir=${DEFAULT_CACHE_HOST_DIR}

DEFAULT_OUTPUT_HOST_DIR=$(pwd)/outputs
output_host_dir=${DEFAULT_OUTPUT_HOST_DIR}

DEFAULT_SLURM_ACCOUNT=""
slurm_account=${DEFAULT_SLURM_ACCOUNT}

DEFAULT_BENCHMARK_SLURM_PARTITION=""
benchmark_slurm_partition=${DEFAULT_BENCHMARK_SLURM_PARTITION}

DEFAULT_EVALUATE_SLURM_PARTITION=""
evaluate_slurm_partition=${DEFAULT_EVALUATE_SLURM_PARTITION}

function _exit_with_help_msg() {
    cat <<EOF
Submit a benchmarking (and optionally, an evaluation) job(s) for the Qwen3-VL (Q3VL) benchmark.

Usage: ${BASH_SOURCE[0]}
    [-ci  | --container-image]           Container image to run the benchmark (default: ${DEFAULT_CONTAINER_IMAGE}).
    [-mri | --model-repo-id]             HuggingFace repo ID of the model to benchmark (default: ${DEFAULT_MODEL_REPO_ID}).
    [-s | --scenario]                    Benchmark scenario (default: ${DEFAULT_SCENARIO}).
    [-m | --mode]                        Benchmark mode (default: ${DEFAULT_MODE}).
    [-stq | --server-target-qps]         The target QPS for the server scenario (default: ${DEFAULT_SERVER_TARGET_QPS}).
    [-tps | --tensor-parallel-size]      Tensor parallelism size for the model deployment (default: ${DEFAULT_TENSOR_PARALLEL_SIZE}).
    [-chd | --cache-host-dir]            Host directory of the ".cache" directory to which HuggingFace will dump the dataset and the model checkpoint, and vLLM will dump compilation artifacts (default: ${DEFAULT_CACHE_HOST_DIR}).
    [-ohd | --output-host-dir]           Host directory to which the benchmark and evaluation results will be dumped (default: ${DEFAULT_OUTPUT_HOST_DIR}).
    [-sa | --slurm-account]              Slurm account for submitting the benchmark and evaluation jobs (default: ${DEFAULT_SLURM_ACCOUNT}).
    [-bsp | --benchmark-slurm-partition] Slurm partition for submitting the benchmarking job; usually a partition with nodes that have GPUs (default: ${DEFAULT_BENCHMARK_SLURM_PARTITION}).
    [-esp | --evaluate-slurm-partition]  Slurm partition for submitting the evaluation job; usually a partition with nodes that have CPUs only (default: ${DEFAULT_EVALUATE_SLURM_PARTITION}).
    [-h | --help]     Print this help message.
EOF
    if [ -n "$1" ]; then
        echo "$(tput bold setab 1)$1$(tput sgr0)"
    fi
    exit "$2"
}

while [[ $# -gt 0 ]]; do
    case $1 in
    -ci | --container-image)
        container_image=$2
        shift
        shift
        ;;
    -ci=* | --container-image=*)
        container_image=${1#*=}
        shift
        ;;
    -mri | --model-repo-id)
        model_repo_id=$2
        shift
        shift
        ;;
    -mri=* | --model-repo-id=*)
        model_repo_id=${1#*=}
        shift
        ;;
    -s | --scenario)
        scenario=$2
        shift
        shift
        ;;
    -s=* | --scenario=*)
        scenario=${1#*=}
        shift
        ;;
    -m | --mode)
        mode=$2
        shift
        shift
        ;;
    -m=* | --mode=*)
        mode=${1#*=}
        shift
        ;;
    -seq | --server-expected-qps)
        server_target_qps=$2
        shift
        shift
        ;;
    -seq=* | --server-expected-qps=*)
        server_target_qps=${1#*=}
        shift
        ;;
    -tps | --tensor-parallel-size)
        tensor_parallel_size=$2
        shift
        shift
        ;;
    -tps=* | --tensor-parallel-size=*)
        tensor_parallel_size=${1#*=}
        shift
        ;;
    -chd | --cache-host-dir)
        cache_host_dir=$2
        shift
        shift
        ;;
    -chd=* | --cache-host-dir=*)
        cache_host_dir=${1#*=}
        shift
        ;;
    -ohd | --output-host-dir)
        output_host_dir=$2
        shift
        shift
        ;;
    -ohd=* | --output-host-dir=*)
        output_host_dir=${1#*=}
        shift
        ;;
    -sa | --slurm-account)
        slurm_account=$2
        shift
        shift
        ;;
    -sa=* | --slurm-account=*)
        slurm_account=${1#*=}
        shift
        ;;
    -bsp | --benchmark-slurm-partition)
        benchmark_slurm_partition=$2
        shift
        shift
        ;;
    -bsp=* | --benchmark-slurm-partition=*)
        benchmark_slurm_partition=${1#*=}
        shift
        ;;
    -esp | --evaluate-slurm-partition)
        evaluate_slurm_partition=$2
        shift
        shift
        ;;
    -esp=* | --evaluate-slurm-partition=*)
        evaluate_slurm_partition=${1#*=}
        shift
        ;;
    -h | --help)
        _exit_with_help_msg "" 0
        ;;
    *)
        _exit_with_help_msg "[ERROR] Unknown option: $1" 1
        ;;
    esac
done

if [[ -z "${container_image}" ]]; then
    _exit_with_help_msg "[ERROR] -ci or --container-image is required." 1
fi

if [[ -z "${cache_host_dir}" ]]; then
    _exit_with_help_msg "[ERROR] -chd or --cache-host-dir is required." 1
fi

if [[ -z "${slurm_account}" ]]; then
    _exit_with_help_msg "[ERROR] -sa or --slurm-account is required." 1
fi

if [[ -z "${benchmark_slurm_partition}" ]]; then
    _exit_with_help_msg "[ERROR] -bsp or --benchmark-slurm-partition is required." 1
fi

if [[ -z "${evaluate_slurm_partition}" ]]; then
    _exit_with_help_msg "[ERROR] -esp or --evaluate-slurm-partition is required." 1
fi

cache_container_dir=/root/.cache
output_container_dir=/outputs

mkdir -p "${output_host_dir}"

benchmark_job_id=$(
    CACHE_HOST_DIR="${cache_host_dir}" \
    CACHE_CONTAINER_DIR="${cache_container_dir}" \
    OUTPUT_HOST_DIR="${output_host_dir}" \
    OUTPUT_CONTAINER_DIR="${output_container_dir}" \
    CONTAINER_IMAGE="${container_image}" \
    SCENARIO="${scenario}" \
    MODE="${mode}" \
    SERVER_TARGET_QPS="${server_target_qps}" \
    TENSOR_PARALLEL_SIZE="${tensor_parallel_size}" \
    MODEL_REPO_ID="${model_repo_id}" \
    sbatch --parsable \
        --account="${slurm_account}" \
        --partition="${benchmark_slurm_partition}" \
        --gres=gpu:"${tensor_parallel_size}" \
        benchmark.sh
)

if [[ "${mode}" == "accuracy_only" ]]; then
    CACHE_HOST_DIR="${cache_host_dir}" \
    CACHE_CONTAINER_DIR="${cache_container_dir}" \
    OUTPUT_HOST_DIR="${output_host_dir}" \
    OUTPUT_CONTAINER_DIR="${output_container_dir}" \
    CONTAINER_IMAGE="${container_image}" \
    BENCHMARK_JOB_ID="${benchmark_job_id}" \
    NVIDIA_VISIBLE_DEVICES=void \
    sbatch \
        --dependency=afterok:"${benchmark_job_id}" \
        --account="${slurm_account}" \
        --partition="${evaluate_slurm_partition}" \
        evaluate.sh
fi