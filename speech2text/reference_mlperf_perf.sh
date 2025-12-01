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
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

#/bin/bash

echo "Time Start: $(date +%s)"
export WORKSPACE_DIR="/workspace"
export DATA_DIR="/data"
export MANIFEST_FILE="${DATA_DIR}/dev-all-repack.json"
export RUN_LOGS=${WORKSPACE_DIR}/run_output
export SCENARIO="Offline"

export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
export CORES_PER_INST=$((${NUM_CORES} / ${NUM_NUMA_NODES}))
export OMP_NUM_THREADS=${CORES_PER_INST}
export INSTS_PER_NODE=1
export NUM_INSTS=$((${NUM_NUMA_NODES} * ${INSTS_PER_NODE}))

export START_CORES=$(lscpu | grep "NUMA node.* CPU.*" | awk "{print \$4}" | cut -d "-" -f 1 | paste -s -d ',')

echo "CORES_PER_INST: ${CORES_PER_INST}"
echo "NUM_INSTS: ${NUM_INSTS}"
echo "START_CORES: ${START_CORES}"

python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir ${RUN_LOGS} \
    --num_workers ${NUM_INSTS} 

echo "Time Stop: $(date +%s)"
