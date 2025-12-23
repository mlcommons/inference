#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=evaluate-slurm-output-%j.txt
#SBATCH --error=evaluate-slurm-error-%j.txt

set -eux
set -p pipefail

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CACHE_HOST_DIR}":"${CACHE_CONTAINER_DIR}","${OUTPUT_HOST_DIR}":"${OUTPUT_CONTAINER_DIR}" \
    --no-container-mount-home \
    --container-env=NVIDIA_VISIBLE_DEVICES \
    mlperf-inf-mm-q3vl evaluate \
        --filename="${OUTPUT_CONTAINER_DIR}"/"${BENCHMARK_JOB_ID}"/mlperf_log_accuracy.json