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

# #/bin/bash

export WORKSPACE_DIR=/workspace
export DATA_DIR=/data
export LIBRISPEECH_DIR=${DATA_DIR}/LibriSpeech
export UTILS_DIR=${WORKSPACE_DIR}/utils
mkdir -p ${LIBRISPEECH_DIR}

#### Librispeech dev-other ####
python ${UTILS_DIR}/download_librispeech.py \
    ${UTILS_DIR}/librispeech-inference_other.csv \
    ${LIBRISPEECH_DIR} \
    -e ${DATA_DIR}

cd ${WORKSPACE_DIR}
python ${UTILS_DIR}/convert_librispeech.py \
    --input_dir ${LIBRISPEECH_DIR}/dev-other \
    --dest_dir ${DATA_DIR}/dev-other-wav \
    --output_json ${DATA_DIR}/dev-other-wav.json
