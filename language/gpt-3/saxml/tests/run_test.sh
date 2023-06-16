#! /bin/bash
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

set -ex;

GPT3_CHECKPOINT_PATH="gs://jwyang-cloud-tpu-tutorial/gpt3-cnndm/checkpoint_00011000"

function find_docker {

  docker_name=$1

  container_id=$(docker ps | grep "${docker_name}" | awk '{print $1}')

  if [ -z ${container_id} ]; then

    echo "Docker ${docker_name} is not running, please check"
    exit 1

  fi

  echo "Found Docker with CONTAINER ID: ${container_id}"

}


function run_lmtest {

  model_name=$1
  model_config_path=$2
  wait_time=$3

  docker exec \
    --privileged \
    ${SAX_ADMIN_SERVER_DOCKER_NAME} \
    python /mlperf_inference/language/gpt-3/saxml/tests/models/lmtest.py --model_name=${model_name} --model_config_path=${model_config_path} --wait=${wait_time};

}

function run_gpt3 {

  model_name=$1
  model_config_path=$2
  wait_time=$3

  checkpoint_path=${GPT3_CHECKPOINT_PATH}

  docker exec \
    --privileged \
    ${SAX_ADMIN_SERVER_DOCKER_NAME} \
    python /server/tests/models/lmtest.py --model_name=${model_name} --model_config_path=${model_config_path} --checkpoint_path=${checkpoint_path} --wait=${wait_time};

}

find_docker ${SAX_ADMIN_SERVER_DOCKER_NAME}


if [ ${HOST} == 1-2 ]; then
  run_lmtest lmcloudspmd2b4test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4Test 60
fi

if [ ${HOST} == v4-4 ]; then
  run_lmtest lmcloudspmd175b32test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32Test 120
  run_gpt3 gpt3175b32 saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP32 120
fi

if [ ${HOST} == 8 ]; then
  run_lmtest lmcloudspmd175b32test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32Test 120
  run_gpt3 gpt3175b32 saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP32 120
fi

if [ ${HOST} == 16 ]; then
  run_lmtest lmcloudspmd175b64test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64Test 120
  run_gpt3 gpt3175b64 saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP 120
fi

if [ ${HOST} == 32 ]; then
  run_lmtest lmcloudspmd175b128test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B128Test 120
  run_gpt3 gpt3175b128 saxml.server.pax.lm.params.c4.C4SpmdGpt3AdamOrgHP128 120
fi
