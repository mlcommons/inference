#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --tasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=benchmark-slurm-output-%j.txt
#SBATCH --error=benchmark-slurm-error-%j.txt

set -eux
set -o pipefail

mkdir -p "${OUTPUT_HOST_DIR}"/"${SLURM_JOB_ID}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CACHE_HOST_DIR}":"${CACHE_CONTAINER_DIR}","${OUTPUT_HOST_DIR}":"${OUTPUT_CONTAINER_DIR}" \
    --no-container-mount-home \
    mlperf-inf-mm-q3vl benchmark vllm \
        --settings.test.scenario="${SCENARIO}" \
        --settings.test.mode="${MODE}" \
        --settings.test.server_target_qps="${SERVER_TARGET_QPS}" \
        --vllm.model.repo_id="${MODEL_REPO_ID}" \
        --vllm.cli=--async-scheduling \
        --vllm.cli=--max-model-len=32768 \
        --vllm.cli=--limit-mm-per-prompt.video=0 \
        --vllm.cli=--tensor-parallel-size="${TENSOR_PARALLEL_SIZE}" \
        --vllm.cli=--no-enable-prefix-caching \
        --settings.logging.log_output.outdir="${OUTPUT_CONTAINER_DIR}"/"${SLURM_JOB_ID}" 