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

### TPU

#### Setup

```
nano ./language/gpt-3/saxml/tpu/setup_tpu.sh
```

set
```
run_setup_sax_servers="yes"
```

```
./language/gpt-3/saxml/tpu/setup_tpu.sh
```

#### SSH to TPUVM

```
gcloud compute tpus tpu-vm ssh ${_TPU_USER}${_TPU_NAME} \
        --project=${PROJECT_ID} \
        --zone=${ZONE}
```
#### Get logs from the container
```
docker logs -f sax-gpt3-model-server-tpu
```
```
docker logs -f sax-gpt3-admin-server-tpu
```

#### Execute Inside `sax-gpt3-admin-server-tpu` Container

```
docker exec \
    --privileged \
    -it sax-gpt3-admin-server-tpu \
    /bin/bash

cd /mlperf_inference/language/gpt-3/saxml/

```

#### Some Paths
```
CNN_TOKENIZED_DATASET_PATH=gs://cnn_dailymail_public/mlperf/tokenized_cnn_dailymail_3.0.0/cnn_dailymail-validation.tfrecord-00000-of-00001
GPT3_SPM_PATH=gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model
GPT3_CHECKPOINT_PATH=gs://mlperf-llm-public2/gpt3-cnndm/checkpoint_00011000

GPT3_MODEL_PATH=/sax/test/gpt3175b64tokenized
GPT3_MODEL_PATH=/sax/test/gpt3175b64tokenizedgreedy

GPT3_MODEL_NAME=gpt3175b64tokenized
GPT3_MODEL_NAME=gpt3175b64tokenizedgreedy

```
#### Publish

##### Publish Model in TestMode
```
python publish_sax_model.py \
  --model_name lmcloudspmd2b4testtokenized \
  --wait 60 \
  --unpublish True

python publish_sax_model.py \
  --model_name lmcloudspmd175b64testtokenized \
  --wait 360 \
  --unpublish True
```

##### Publish Model with GPT3 Checkpoint

```
CHECKPOINT_PATH=${GPT3_CHECKPOINT_PATH}
MODEL_NAME=${GPT3_MODEL_NAME}

python publish_sax_model.py \
  --model_name ${MODEL_NAME} \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --wait 1000
```

#### Loadgen

##### Test Loadgen

```
# with TestMode Models
MODEL_PATH=/sax/test/lmcloudspmd2b4test
MODEL_PATH=/sax/test/lmcloudspmd175b64test

# with GPT3 Model
MODEL_PATH=${GPT3_MODEL_PATH}

```


###### Server

```
python main.py \
  --scenario Server \
  --model-path ${MODEL_PATH} \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --accuracy \
  --max-examples 20 \
  --perf-examples 20 \
  --log-interval 5 \
  --log-path /mlperf_inference/language/gpt-3/saxml/test_loadgen_logs

```
###### Offline

```
python main.py \
  --scenario Offline \
  --model-path ${MODEL_PATH} \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --batch-size 2 \
  --accuracy \
  --max-examples 10 \
  --perf-examples 10 \
  --log-interval 2 \
  --log-path /mlperf_inference/language/gpt-3/saxml/test_loadgen_logs

```
##### Run Loadgen with GPT3 Model

```
MODEL_PATH=${GPT3_MODEL_PATH}
```

###### Server
```
python main.py \
  --scenario Server \
  --model-path ${MODEL_PATH} \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --accuracy
```

###### Offline
```
BATCH_SIZE=4
python main.py \
  --scenario Offline \
  --model-path ${MODEL_PATH} \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --batch-size ${BATCH_SIZE} \
  --accuracy
```

```
ls /mlperf_inference/language/gpt-3/saxml/loadgen_logs
```
#### Evaluation


##### Test Evaluation
```
# with TestMode Models
SPM_PATH=gs://cnn_dailymail_public/mlperf/vocab/test_model.model

# with GPT3 Model
SPM_PATH=${GPT3_SPM_PATH}

```

###### Server

```
python evaluation.py \
  --mlperf-accuracy-file /mlperf_inference/language/gpt-3/saxml/test_loadgen_logs/Server/accuracy/mlperf_log_accuracy.json \
  --spm-path ${SPM_PATH} \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --log-dir /mlperf_inference/language/gpt-3/saxml/test_evaluation_logs/Server

```
###### Offline

```
python evaluation.py \
  --mlperf-accuracy-file /mlperf_inference/language/gpt-3/saxml/test_loadgen_logs/Offline/accuracy/mlperf_log_accuracy.json \
  --spm-path ${SPM_PATH} \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --log-dir /mlperf_inference/language/gpt-3/saxml/test_evaluation_logs/Offline

```
##### Run Evaluation with GPT3 Model

###### Server

```
python evaluation.py \
  --spm-path ${GPT3_SPM_PATH} \
  --mlperf-accuracy-file /mlperf_inference/language/gpt-3/saxml/loadgen_logs/Server/accuracy/mlperf_log_accuracy.json \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --log-dir /mlperf_inference/language/gpt-3/saxml/evaluation_logs/Server

```
###### Offline

```
python evaluation.py \
  --spm-path ${SPM_PATH} \
  --mlperf-accuracy-file /mlperf_inference/language/gpt-3/saxml/loadgen_logs/Offline/accuracy/mlperf_log_accuracy.json \
  --dataset-path ${CNN_TOKENIZED_DATASET_PATH} \
  --log-dir /mlperf_inference/language/gpt-3/saxml/evaluation_logs/Offline
```

```
gsutil cp -r /mlperf_inference/language/gpt-3/saxml/evaluation_logs gs://cnn_dailymail_public/mlperf/evaluation_logs

```

### GPU

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
