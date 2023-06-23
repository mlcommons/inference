
# ./language/gpt-3/saxml/tpu/setup_tpu.sh
set -e;

_LOCAL_WORK_DIR="/usr/local/google/home/morgandu/projects/mlcommons-inference/language/gpt-3/saxml"
_WORK_DIR="gpt-3/saxml"
_BUILD_DIR="tpu"
_TEST_DIR="tests"

IMAGE_PROJECT_ID="tpu-prod-env-one-vm"

RUNTIME_VERSION="v2-alpha-tpuv5-lite"
PLATFORM_CHIP=tpuv4 # No need to change across v4 and vlp

# "v4-4" # test mode pass, checkpoint loading memory issue
# "v4-8" # test mode pass, checkpoint loading memory issue
# "v4-16" # testing now
# "vlp-1-2" # build succeeded, single host test mode pass (mainly for build)
# "vlp-8" # can not access all workers, firewall issue
# "vlp-16" # can not access all workers, firewall issue

HOST_TYPE="v4"
HOST_COUNT="16"

# HOST_TYPE="vlp"
# HOST_COUNT="1-2"

HOST=${HOST_TYPE}-${HOST_COUNT}
echo "HOST: ${HOST}"

_TPU_USER="root@"

run_create_tpu="no"
run_delete_queued_resource="no"
run_delete_tpu="no"
run_get_main_worker="yes"
run_create_disk="no"
run_mount_disk="no"
run_create_bucket="no"
run_delete_bucket="no"

run_scp="yes"

run_build_image="no"
run_docker_push_for_test="no"
run_pull_for_test="no"

run_setup_sax_servers="yes"
run_test="no"


if [ ${HOST} == v4-4 ]; then

  PROJECT_ID="tpu-prod-env-one-vm"
  _SERVICE_ACCOUNT="630405687483-compute@developer.gserviceaccount.com"
  ACCELERATOR_TYPE="v4-32"
  ZONE="us-central2-b"
  PLATFORM_TOPOLOGY="2x2x4"

elif [ ${HOST} == v4-8 ]; then

  PROJECT_ID="tpu-prod-env-one-vm"
  _SERVICE_ACCOUNT="630405687483-compute@developer.gserviceaccount.com"
  ACCELERATOR_TYPE="v4-64"
  ZONE="us-central2-b"
  PLATFORM_TOPOLOGY="2x4x4"

elif [ ${HOST} == v4-16 ]; then

  PROJECT_ID="tpu-prod-env-one-vm"
  _SERVICE_ACCOUNT="630405687483-compute@developer.gserviceaccount.com"
  ACCELERATOR_TYPE="v4-128"
  ZONE="us-central2-b"
  PLATFORM_TOPOLOGY="4x4x4"
  _TPU_NAME="morgandu-tpu-vm-4"


elif [ ${HOST} == vlp-1-2 ]; then

  PROJECT_ID="tpu-prod-env-small"
  _SERVICE_ACCOUNT="463402977885-compute@developer.gserviceaccount.com"
  ACCELERATOR_TYPE="v5litepod-4"
  ZONE="us-west4-a"
  PLATFORM_TOPOLOGY="2x2"
  _TPU_NAME="morgandu-tpu-vm-4"

elif [ ${HOST} == vlp-8 ]; then

  PROJECT_ID="tpu-prod-env-large-cont"
  _SERVICE_ACCOUNT="641569443805-compute@developer.gserviceaccount.com"
  ZONE="us-east1-c"
  ACCELERATOR_TYPE="v5litepod-32"
  PLATFORM_TOPOLOGY="4x8"

elif [ ${HOST} == vlp-16 ]; then

  PROJECT_ID="tpu-prod-env-large-cont"
  _SERVICE_ACCOUNT="641569443805-compute@developer.gserviceaccount.com"
  ZONE="us-east1-c"
  ACCELERATOR_TYPE="v5litepod-64"
  PLATFORM_TOPOLOGY="8x8"

elif [ ${HOST} == vlp-32 ]; then

  PROJECT_ID="tpu-prod-env-large-cont"
  _SERVICE_ACCOUNT="641569443805-compute@developer.gserviceaccount.com"
  ZONE="us-east1-c"
  ACCELERATOR_TYPE="v5litepod-128"
  PLATFORM_TOPOLOGY="8x16"

fi

echo "_TPU_NAME: ${_TPU_NAME}"

echo "PROJECT_ID=${PROJECT_ID}"
echo "ZONE=${ZONE}"
echo "ACCELERATOR_TYPE=${ACCELERATOR_TYPE}"
echo "RUNTIME_VERSION=${RUNTIME_VERSION}"
echo "PLATFORM_CHIP=${PLATFORM_CHIP}"
echo "PLATFORM_TOPOLOGY=${PLATFORM_TOPOLOGY}"


# sax
_SAX_ADMIN_SERVER_IMAGE_NAME="sax-gpt3-admin-server-tpu"
_SAX_MODEL_SERVER_IMAGE_NAME="sax-gpt3-model-server-tpu"
echo "_SAX_ADMIN_SERVER_IMAGE_NAME: ${_SAX_ADMIN_SERVER_IMAGE_NAME}"
echo "_SAX_MODEL_SERVER_IMAGE_NAME: ${_SAX_MODEL_SERVER_IMAGE_NAME}"

_SAX_ADMIN_SERVER_DOCKER_NAME=${_SAX_ADMIN_SERVER_IMAGE_NAME}
_SAX_MODEL_SERVER_DOCKER_NAME=${_SAX_MODEL_SERVER_IMAGE_NAME}
echo "_SAX_ADMIN_SERVER_DOCKER_NAME: ${_SAX_ADMIN_SERVER_DOCKER_NAME}"
echo "_SAX_MODEL_SERVER_DOCKER_NAME: ${_SAX_MODEL_SERVER_DOCKER_NAME}"


_SAX_ADMIN_STORAGE_BUCKET="sax-admin-host-${HOST}"
echo "_SAX_ADMIN_STORAGE_BUCKET: ${_SAX_ADMIN_STORAGE_BUCKET}"


_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME="gcr.io/${IMAGE_PROJECT_ID}/${_SAX_ADMIN_SERVER_IMAGE_NAME}"
_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME="gcr.io/${IMAGE_PROJECT_ID}/${_SAX_MODEL_SERVER_IMAGE_NAME}"
CLOUD_TPU_SAX_TEST_TAG="latest"
echo "_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME: ${_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME}"
echo "_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME: ${_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME}"
echo "CLOUD_TPU_SAX_TEST_TAG: ${CLOUD_TPU_SAX_TEST_TAG}"


if [ ${run_create_tpu} == "yes" ]; then

  if [ ${HOST_TYPE} == v4 ]; then

    echo "Creating a TPU VM ${_TPU_NAME} ...";
    gcloud alpha compute tpus tpu-vm create ${_TPU_NAME} \
      --project=${PROJECT_ID} \
      --zone=${ZONE} \
      --accelerator-type=${ACCELERATOR_TYPE} \
      --version=${RUNTIME_VERSION};

    echo "Waiting for the newly created TPU VM to be ready to connect. (should be less than 2 mins)";
    _=$(gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
        --project=${PROJECT_ID} \
        --zone=${ZONE} \
        --command="echo ready" 2>&1);

  fi

  if [ ${HOST_TYPE} == vlp ]; then

    if [ ${run_delete_queued_resource} == "yes" ]; then
      echo "Deleting a previously queued resource ${_TPU_NAME} ...";
      gcloud alpha compute tpus queued-resources delete -q ${_TPU_NAME} \
        --project ${PROJECT_ID} \
        --zone=${ZONE}
    fi

    echo "Creating a queued resource ${_TPU_NAME} ...";
    gcloud alpha compute tpus queued-resources create ${_TPU_NAME} \
          --node-id ${_TPU_NAME} \
          --project=${PROJECT_ID} \
          --zone=${ZONE} \
          --accelerator-type=${ACCELERATOR_TYPE} \
          --runtime-version=${RUNTIME_VERSION} \
          --reserved;

    echo "Describing a queued resource ${_TPU_NAME} ...";
    gcloud alpha compute tpus queued-resources describe ${_TPU_NAME} \
      --project ${PROJECT_ID} \
      --zone=${ZONE};

  fi

else

  echo "Did not create a new TPU VM, reusing a persistent one ...";

fi


if [ ${run_get_main_worker} == "yes" ]; then

  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --worker=all \
    --command="hostname";

  echo "Acquiring the main tpu worker name ...";
  _TPU_WORKER_NAME=$(gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="hostname");
  echo "_TPU_WORKER_NAME: ${_TPU_WORKER_NAME}"

fi

if [ ${run_create_disk} == "yes" ]; then

  _DISK_NAME="ssd-${_TPU_WORKER_NAME}"
  _DISK_ZONE="${ZONE}"
  _DISK_SIZE="100G"
  _DISK_TYPE="pd-ssd"
  echo "_DISK_NAME: ${_DISK_NAME}"
  echo "_DISK_ZONE: ${_DISK_ZONE}"
  echo "_DISK_SIZE: ${_DISK_SIZE}"
  echo "_DISK_TYPE: ${_DISK_TYPE}"

  echo "Creating a persistent disk ..."
  gcloud compute disks create ${_DISK_NAME} \
    --project=${PROJECT_ID} \
    --size ${_DISK_SIZE}  \
    --type ${_DISK_TYPE} \
    --zone ${_DISK_ZONE};

  echo "Attaching the persistent disk to the main tpu worker ... "
  gcloud alpha compute instances attach-disk ${_TPU_WORKER_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --disk=${_DISK_NAME} \
    --mode=rw;

  echo "Setting the persistent disk to auto delete ... "
  gcloud compute instances set-disk-auto-delete ${_TPU_WORKER_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --auto-delete \
    --disk=${_DISK_NAME};

else

  echo "Did not create a persistent disk, reusing a previously created one ..."

fi


set -e


if [ ${run_scp} == "yes" ]; then

  echo "Removing the saxml gpt3 directory on the TPU VM";
  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --worker=all \
    --command="
      set ex;
      rm -rf ~/${_WORK_DIR};
      mkdir -p ~/${_WORK_DIR};
      ls -ltr";

  echo "Copying the sax directory to the TPU VM";
  gcloud compute tpus tpu-vm scp ${_LOCAL_WORK_DIR}/* ${_TPU_USER}${_TPU_NAME}:${_WORK_DIR} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --worker=all \
    --recurse;

else

  echo "Did not copy new code to the TPU VM ... ";

fi


if [ ${run_mount_disk} == "yes" ]; then

  echo "Mounting the disk and restart docker daemon to run on the main tpu worker ... "
  _PERSISTENT_SSD_DISK="${_PERSISTENT_SSD_DISK:="sdb"}"
  gcloud compute ssh ${_TPU_USER}${_TPU_WORKER_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set -ex;
      cd ~/${_WORK_DIR}/${_BUILD_DIR};
      PERSISTENT_SSD_DISK=${_PERSISTENT_SSD_DISK} ./init_ssd.sh;";

else

  echo "Did not mount SSD disk to the main tpu worker, already mounted ...";

fi


if [ ${run_build_image} == "yes" ]; then

  echo "Building image on the main tpu worker: ${_TPU_WORKER_NAME} ... ";
  gcloud compute ssh ${_TPU_USER}${_TPU_WORKER_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set -ex;
      gcloud auth configure-docker --quiet;
      cd ~/${_WORK_DIR}/${_BUILD_DIR};
      SAX_ADMIN_SERVER_IMAGE_NAME=${_SAX_ADMIN_SERVER_IMAGE_NAME} \
        SAX_MODEL_SERVER_IMAGE_NAME=${_SAX_MODEL_SERVER_IMAGE_NAME} \
        ./build_image.sh;
      docker tag ${_SAX_ADMIN_SERVER_IMAGE_NAME} ${_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG};
      docker tag ${_SAX_MODEL_SERVER_IMAGE_NAME} ${_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG};";

else

  echo "Did not build the image on the main tpu worker ...";

fi


if [ ${run_docker_push_for_test} == "yes" ]; then

  echo "Pushing the newly built admin and model server images to GCR for testing ... ";
  gcloud compute ssh ${_TPU_USER}${_TPU_WORKER_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set -ex;
      gcloud auth configure-docker --quiet;
      gcloud config set account ${_SERVICE_ACCOUNT};
      docker push ${_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG};
      docker push ${_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG};";

else

  echo "Did not push the newly built admin and model server images to GCR for testing ...";

fi


if [ ${run_pull_for_test} == "yes" ]; then

  echo "Pulling the admin server docker image from GCR for testing ... ";
  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set -ex;
      gcloud auth configure-docker --quiet;
      docker pull ${_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG};";

  echo "Pulling the model server docker image from GCR for testing ... ";
  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --worker=all \
    --command="
      set -ex;
      gcloud auth configure-docker --quiet;
      docker pull ${_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG};";

else

  echo "Did not pull the docker image from GCP for test ..."

fi


if [ ${run_create_bucket} == "yes" ]; then

  echo "Creating a new storage bucket ...";
  gcloud storage buckets create gs://${_SAX_ADMIN_STORAGE_BUCKET} --project=${PROJECT_ID}

else

  echo "Did not create a new storage bucket, reusing an existing one ...";

fi


if [ ${run_setup_sax_servers} == "yes" ]; then

  echo "Setting up TPU VM admin server for testing ... ";
  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set -ex;
      cd ~/${_WORK_DIR};
      SERVER_TYPE=admin \
        SAX_ADMIN_SERVER_IMAGE_NAME=${_CLOUD_TPU_SAX_ADMIN_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG} \
        SAX_ADMIN_SERVER_DOCKER_NAME=${_SAX_ADMIN_SERVER_DOCKER_NAME} \
        SAX_ADMIN_STORAGE_BUCKET=${_SAX_ADMIN_STORAGE_BUCKET} \
        ./setup_sax_servers.sh;";

  echo "Setting up TPU VM model server for testing ... ";
  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --worker=all \
    --command="
      set -ex;
      cd ~/${_WORK_DIR};
      SERVER_TYPE=model \
        SAX_MODEL_SERVER_IMAGE_NAME=${_CLOUD_TPU_SAX_MODEL_SERVER_IMAGE_NAME}:${CLOUD_TPU_SAX_TEST_TAG} \
        SAX_MODEL_SERVER_DOCKER_NAME=${_SAX_MODEL_SERVER_DOCKER_NAME} \
        SAX_ADMIN_STORAGE_BUCKET=${_SAX_ADMIN_STORAGE_BUCKET} \
        PLATFORM_CHIP=${PLATFORM_CHIP} \
        PLATFORM_TOPOLOGY=${PLATFORM_TOPOLOGY} \
        ./setup_sax_servers.sh;";

else

  echo "Did not run set up servers on TPU VM ..."

fi

if [ ${run_test} == "yes" ]; then

  echo "Running test on the TPU VM ... ";
  gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set -ex;
      cd ~/${_WORK_DIR}/${_TEST_DIR};
      HOST=${HOST} \
        SAX_ADMIN_SERVER_DOCKER_NAME=${_SAX_ADMIN_SERVER_DOCKER_NAME} \
        ~/gpt-3/saxml/tests/run_test.sh;";
  echo "Test run on TPU VM completed Successfully !"

else

  echo "Did not run test on TPU VM ..."

fi


if [ ${run_delete_tpu} == "yes" ]; then

  echo "Cleaning up the TPU VM ... "
  gcloud compute tpus tpu-vm delete -q ${_TPU_NAME} \
      --project=${PROJECT_ID} \
      --zone=${ZONE};

else

  echo "Did not delete the TPU VM ... "

fi


if [ ${run_delete_bucket} == "yes" ]; then

  echo "Deleting the storage bucket ..."
  gcloud storage rm -r gs://${_SAX_ADMIN_STORAGE_BUCKET} --project=${PROJECT_ID}

else

  echo "Did not delete the storage bucket ..."

fi