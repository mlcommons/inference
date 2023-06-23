# GPT-3 Reference Implementation

git clone https://github.com/mlcommons/inference.git
cd inference

CWD=$(pwd)

## Model
The benchmark reference model GPT-3 was re-implemented in PaxML with the best efforts to match available details from the paper:

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, et. al. Language Models are Few-Shot Learners. arXiv:2005.14165

## Dataset
This benchmark uses the cnn_dailymail/3.0.0 dataset from TensorFlow Dataset.

## Benchmark

### Setup

#### TPU

```
nano ./language/gpt-3/saxml/tpu/setup_tpu.sh
```

edit:
```
run_setup_sax_servers="yes"
```

```
./language/gpt-3/saxml/tpu/setup_tpu.sh
```

```
docker exec \
    --privileged \
    -it sax-gpt3-admin-server-tpu \
    ls /mlperf_inference/language/gpt-3/saxml
```

##### Test Mode
```
docker exec \
    --privileged \
    -it sax-gpt3-admin-server-tpu \
    /bin/bash
```
```
python /mlperf_inference/language/gpt-3/saxml/publish_sax_model.py --model_name lmcloudspmd175b64test --wait 180 --unpublish True
```
##### Checkpoint Mode
```
CHECKPOINT_PATH="gs://mlperf-llm-public2/gpt3-cnndm/checkpoint_00011000"
python /mlperf_inference/language/gpt-3/saxml/publish_sax_model.py --model_name gpt3175b64 --checkpoint_path=${CHECKPOINT_PATH} --wait 1200
```

#### GPU

GPU_NAME=morgandu-a100
PROJECT_ID=tpu-prod-env-one-vm
ZONE=europe-west4-a

##### Create a Google Cloud Bucket
SAX_ADMIN_STORAGE_BUCKET="sax-admin-mlperf-gp3" # tpu
SAX_ADMIN_STORAGE_BUCKET="sax-admin-mlperf-gpt-3" # gpu
gcloud storage buckets create gs://${SAX_ADMIN_STORAGE_BUCKET} --project=${PROJECT_ID}


##### Create a Google Cloud GPU
gcloud compute instances create ${GPU_NAME} \
   --project ${PROJECT_ID} \
   --zone ${ZONE} \
   --machine-type a2-ultragpu-1g \
   --maintenance-policy TERMINATE \
   --restart-on-failure \
   --image-family tf2-ent-latest-gpu \
   --image-project deeplearning-platform-release \
   --boot-disk-size 300GB \
   --metadata "install-nvidia-driver=True,proxy-mode=project_editors" \
   --scopes https://www.googleapis.com/auth/cloud-platform

gcloud compute ssh ${GPU_NAME} \
   --project ${PROJECT_ID} \
   --zone ${ZONE}


##### Install Bazel

BAZEL_VERSION=5.4.0
mkdir ~/bazel
cd ~/bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
chmod +x bazel-*.sh
sudo su
BAZEL_VERSION=5.4.0
./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
exit

bazel version
WARNING: Invoking Bazel in batch mode since it is not invoked from within a workspace (below a directory having a WORKSPACE file).
Build label: 5.4.0
Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
Build time: Thu Dec 15 16:14:25 2022 (1671120865)
Build timestamp: 1671120865
Build timestamp as int: 1671120865


###### Build Loadgen
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
cd ../..


###### Reference Implementation DIR

_LOCAL_WORK_DIR="/usr/local/google/home/morgandu/projects/mlcommons-inference/language/gpt-3/saxml"
_WORK_DIR="mlperf_inference/language/gpt-3/saxml"
gcloud compute ssh ${GPU_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --command="
      set ex;
      rm -rf ~/${_WORK_DIR};
      mkdir -p ~/${_WORK_DIR};";


gcloud compute scp ${_LOCAL_WORK_DIR}/* ${GPU_NAME}:${_WORK_DIR} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --recurse;


cd ~
#_REF_WORK_DIR="mlperf_inference/language/gpt-3/saxml"
_REF_WORK_DIR="gpt-3/saxml"
cd ${_REF_WORK_DIR}
REF_CWD=$(pwd)
echo ${REF_CWD}


###### Build and Run SAXML
cd ~
git clone https://github.com/google/saxml.git
cd saxml
git fetch && git checkout r0.2.1

rm -rf saxml/server/torch

###### cp for SAXML build

cd ~
_SAXML_DIR="saxml"
cd ${_SAXML_DIR}
SAX_CWD=$(pwd)
echo ${SAX_CWD}


cp ${REF_CWD}/tpu/saxml/saxml/server/pax/lm/params/BUILD ${SAX_CWD}/saxml/server/pax/lm/params/BUILD
cp ${REF_CWD}/tpu/saxml/saxml/server/pax/lm/params/c4.py ${SAX_CWD}/saxml/server/pax/lm/params/c4.py


##### init
cd ${SAX_CWD}
saxml/tools/init_cloud_vm.sh

SAX_ADMIN_STORAGE_BUCKET="sax-admin-mlperf-gp3" #tpu
SAX_ADMIN_STORAGE_BUCKET="sax-admin-mlperf-gpt-3" #gpu

##### start admin server
cd ${SAX_CWD}
bazel run saxml/bin:admin_config -- \
  --sax_cell=/sax/test \
  --sax_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
  --fs_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-fs-root \
  --alsologtostderr


bazel run saxml/bin:admin_server -- \
  --sax_cell=/sax/test \
  --sax_root=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
  --port=10000 \
  --alsologtostderr

##### start model server
cp requirements-cuda.txt requirements.txt

SAX_ROOT=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root \
bazel run saxml/server:server -- \
  --sax_cell=/sax/test \
  --port=10001 \
  --platform_chip=a100 \
  --platform_topology=4 \
  --jax_platforms=cuda \
  --alsologtostderr


##### install client

cd ~/saxml
mkdir -p ~/bazel/sax_client/output/
bazel --output_base=~/bazel/sax_client/output/ build \
        --color=yes \
        --curses=yes \
        --compile_one_dependency \
        saxml/client/python:sax.cc
mv bazel-bin ~/sax_client

cd ~
git clone https://github.com/pybind/pybind11_abseil.git
cd pybind11_abseil
bazel build \
        --color=yes \
        --curses=yes \
        pybind11_abseil/status.so
cp bazel-bin/pybind11_abseil/status.so ~/sax_client/saxml/client/python/

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get -qq update
sudo apt-get -qq install -y --no-install-recommends python3.8

export PYTHONPATH=~/sax_client/saxml/client/python


docker exec \
    --privileged \
    -it sax-admin-server-mlperf-gpt3 \
    ls /mlperf_inference/language/gpt-3/saxml


docker exec \
    --privileged \
    -it sax-admin-server-mlperf-gpt3 \
    python /mlperf_inference/language/gpt-3/saxml/publish_sax_model.py --model_name lmcloudspmd2b4test


docker exec \
    --privileged \
    -it sax-admin-server-mlperf-gpt3 \
    python /mlperf_inference/language/gpt-3/saxml/publish_sax_model.py --model_name lmcloudspmd175b64test

###### publish model

export SAX_ROOT=gs://${SAX_ADMIN_STORAGE_BUCKET}/sax-root

cd ${REF_CWD}
python3.8 publish_sax_model.py


###### Running the Benchmark

### Run Benchmark

### Run Evaluation


## License:
Apache License Version 2.0.
