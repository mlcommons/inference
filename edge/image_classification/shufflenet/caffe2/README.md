# Caffe2 ShuffleNet Evaluation with [MLModelScope](http://docs.mlmodelscope.org)

## Model Manifest

A model manifest is defined in YAML format and contains all the information needed to reproduce a model’s evaluation results. It tells MLModelScope the HW/SW stack to instantiate and how to evaluate the model.

The manifest for this model is at [ShuffleNet_Caffe2](https://github.com/rai-project/caffe2/blob/master/builtin_models/ShuffleNet_Caffe2.yml).

## Evaluation

### MlModelScope Configration

You must have a carml config file called .carml_config.yml under your home directory. Please refer to [carml_config.yml](https://docs.mlmodelscope.org/installation/configuration/).

### Tracer and Database

Please refer to [external_services](https://docs.mlmodelscope.org/installation/source/external_services/)

### Software Environment

[Base](https://github.com/rai-project/dlframework/blob/master/dockerfiles/base/Dockerfile.amd64_cpu)
[Caffe2](https://github.com/rai-project/go-caffe2/blob/master/dockerfiles/Dockerfile.amd64_cpu)

### Evalution

```
#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=shufflenet
MODEL_NAME=ShuffleNet_Caffe2
MODEL_VERSION=1.0
NUM_FILE_PARTS=-1
BATCH_SIZE=48
TRACE_LEVEL=MODEL_TRACE

docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=true \
      --publish_predictions=false \
      --gpu=0 \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_address=$DATABASE_ADDRESS \
      --database_name=$DATABASE_NAME \
      --trace_level=$TRACE_LEVEL
```
