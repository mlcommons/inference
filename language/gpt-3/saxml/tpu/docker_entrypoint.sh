#!/bin/bash
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

if [[ "$1" = "admin-server" ]]; then

  shift 1

  admin_config \
    --sax_cell=/sax/test \
    --fs_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-fs-root \
    --alsologtostderr;

  admin_server \
    --sax_cell=/sax/test \
    --port=10000 \
    --alsologtostderr;


elif [[ "$1" = "model-server" ]]; then

  shift 1

  server \
      --sax_cell=/sax/test \
      --port=10001 \
      --platform_chip=${PLATFORM_CHIP} \
      --platform_topology=${PLATFORM_TOPOLOGY} \
      --alsologtostderr ;

else

    eval "$@"

fi

tail -f /dev/null
