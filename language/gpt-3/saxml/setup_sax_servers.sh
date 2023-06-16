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


function find_image {

  image_name=$1

  if [ -z $(docker images -q ${image_name}) ]; then
    echo "Docker image ${image_name} doesn't exist, please create or pull"
    exit 1
  fi

  echo "Find image ${image_name}"

}


function stop_docker {

  docker=$1

  echo "Stop Docker: "
  docker stop -t0 ${docker};

  sleep 20
}


function cleanup_docker {

  docker_name=$1

  container_id=$(docker ps | grep "${docker_name}" | awk '{print $1}')

  if [ ! -z ${container_id} ]; then
    stop_docker ${container_id}
  fi
}


function start_admin_server {

  image_name=$1
  docker_name=$2
  admin_bucket_name=$3

  docker run \
    --privileged  \
    --shm-size 16G \
    --name ${docker_name} \
    -it \
    -d \
    --rm \
    --network host \
    --env SAX_ADMIN_STORAGE_BUCKET=${admin_bucket_name} \
    --env SAX_ROOT=gs://${admin_bucket_name}/sax-root \
    --mount=type=bind,source=$(pwd),target=/mlperf_inference/language/gpt-3/saxml/ \
    ${image_name}

}

function start_model_server {

  platform_chip=$1
  platform_topology=$2
  image_name=$3
  docker_name=$4
  admin_bucket_name=$5

  docker run \
    --privileged  \
    --shm-size 16G \
    --name ${docker_name} \
    -it \
    -d \
    --rm \
    --network host \
    --env SAX_ADMIN_STORAGE_BUCKET=${admin_bucket_name} \
    --env SAX_ROOT=gs://${admin_bucket_name}/sax-root \
    --env PLATFORM_CHIP=${platform_chip} \
    --env PLATFORM_TOPOLOGY=${platform_topology} \
    ${image_name}
}

function setup_admin_server {

  find_image ${SAX_ADMIN_SERVER_IMAGE_NAME}
  cleanup_docker ${SAX_ADMIN_SERVER_DOCKER_NAME}
  start_admin_server ${SAX_ADMIN_SERVER_IMAGE_NAME} ${SAX_ADMIN_SERVER_DOCKER_NAME} ${SAX_ADMIN_STORAGE_BUCKET}

}

function setup_model_server {

  find_image ${SAX_MODEL_SERVER_IMAGE_NAME}
  cleanup_docker ${SAX_MODEL_SERVER_DOCKER_NAME}
  start_model_server ${PLATFORM_CHIP} ${PLATFORM_TOPOLOGY} ${SAX_MODEL_SERVER_IMAGE_NAME} ${SAX_MODEL_SERVER_DOCKER_NAME} ${SAX_ADMIN_STORAGE_BUCKET}

}


if [ ${SERVER_TYPE} == "admin" ] ; then

  echo "Setup SAX Admin Server ... "
  setup_admin_server;

fi


if [ ${SERVER_TYPE} == "model" ] ; then

  echo "Setup SAX Model Server ... "
  setup_model_server;

fi
