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

GPT3_CHECKPOINT_PATH="gs://mlperf-llm-public2/gpt3-cnndm/checkpoint_00011000"
TEST_FILE_PATH="/mlperf_inference/language/gpt-3/saxml/tests/models/lmtest.py"

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
  tokenized=$4

  docker exec \
    --privileged \
    ${SAX_ADMIN_SERVER_DOCKER_NAME} \
    python ${TEST_FILE_PATH} --model_name=${model_name} --model_config_path=${model_config_path} --wait=${wait_time} --tokenized=${tokenized};

}


find_docker ${SAX_ADMIN_SERVER_DOCKER_NAME}


if [ ${HOST} == vlp-1-2 ]; then
  run_lmtest lmcloudspmd2b4test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4Test 45 False
  run_lmtest lmcloudspmd2b4testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4TestTokenized 45 True
fi

if [ ${HOST} == a100-1-8 ]; then
  run_lmtest lmcloudspmd2b8test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B8Test 120 False
  run_lmtest lmcloudspmd2b8testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B8TestTokenized 120 True
fi

if [ ${HOST} == a100-1-16 ]; then
  run_lmtest lmcloudspmd2b16test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B16Test 120 False
  run_lmtest lmcloudspmd2b16testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B16TestTokenized 120 True
fi

if [ ${HOST} == v4-1 ]; then
  run_lmtest lmcloudspmd2b4test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4Test 45 False
  run_lmtest lmcloudspmd2b4testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd2B4TestTokenized 45 True
fi

if [ ${HOST} == v4-4 ]; then
  run_lmtest lmcloudspmd175b16test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B16Test 120 False
  run_lmtest lmcloudspmd175b16testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B16TestTokenized 120 True
fi

if [ ${HOST} == v4-8 ]; then
  run_lmtest lmcloudspmd175b32test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32Test 120 False
  run_lmtest lmcloudspmd175b32testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32TestTokenized 120 True
fi

if [ ${HOST} == v4-16 ]; then
  run_lmtest lmcloudspmd175b64test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64Test 120 False
  run_lmtest lmcloudspmd175b64testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64TestTokenized 120 True
fi

if [ ${HOST} == vlp-8 ]; then
  run_lmtest lmcloudspmd175b32test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32Test 120 False
  run_lmtest lmcloudspmd175b32testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B32TestTokenized 120 True
fi

if [ ${HOST} == vlp-16 ]; then
  run_lmtest lmcloudspmd175b64test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64Test 120 False
  run_lmtest lmcloudspmd175b64testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B64TestTokenized 120 True
fi

if [ ${HOST} == vlp-32 ]; then
  run_lmtest lmcloudspmd175b128test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B128Test 120 False
  run_lmtest lmcloudspmd175b128testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B128TestTokenized 120 True
fi

if [ ${HOST} == vlp-64 ]; then
  run_lmtest lmcloudspmd175b256test saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B256Test 120 False
  run_lmtest lmcloudspmd175b256testtokenized saxml.server.pax.lm.params.lm_cloud.LmCloudSpmd175B256TestTokenized 120 True
fi
