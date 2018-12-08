# Caffe2 ShuffleNet Evaluation with [MLModelScope](http://docs.mlmodelscope.org)

## Model Information and Environment

A model manifest is defined in YAML format and contains all the information needed to reproduce a model’s evaluation results. It tells MLModelScope the HW/SW stack to instantiate and how to evaluate the model.

The manifest for this model is at [ShuffleNet_Caffe2](https://github.com/rai-project/caffe2/blob/master/builtin_models/ShuffleNet_Caffe2.yml).

The environment is run within the [Caffe2](https://github.com/rai-project/go-caffe2/blob/master/dockerfiles/Dockerfile.amd64_cpu) dockerfile.

## Evaluation

The following command outputs the available evaluation commands in MLModelScope.

```
docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest predict -h
```

The example comands are in [example.sh](example.sh).

### Performance

You can evaluate the performance on CPU using the following script:

```bash
#!/bin/bash

DATABASE_ADDRESS=X.X.X.X # the ip of database to publish traces to
DATABASE_NAME=shufflenet_model_trace # the name of database to publish traces to
MODEL_NAME=ShuffleNet_Caffe2 # model name
MODEL_VERSION=1.0 # model version
NUM_FILE_PARTS=100 # number of batches to be processed, set to -1 to evalute the entire ImageNet
BATCH_SIZE=1 # batch size
TRACE_LEVEL=MODEL_TRACE # trace level

docker run --network host -t -v $HOME:/root -u `id -u`:`id -g` carml/caffe2-agent:amd64-cpu-latest predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=true \ # weather to publish to database
      --publish_predictions=false \ # weather to publish the prediction  of each inputto database
      --gpu=0 \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --trace_level=$TRACE_LEVEL
```

If you need to run with gpu enabled then you need to use `nvidia-docker` instead of `docker`, change the docker image tag from `amd64-cpu-latest` to `amd64-gpu-latest` and then set the `gpu` option to `1`.

### Accuracy

Similar to the script above but set `NUM_FILE_PARTS` to `-1` and config the tracing and publishing behavior as needed.

- The evaluation process outputs the top1 and top5 accuracy.
- You can also publish the top1 and top5 accuracy to the database by setting `publish` to `True`.
- To also publish the prediction probabilies of each input image, set `publish_predicitons` to `True`.

## Reporting

The following command outputs the available reporting commands in MLModelScope.

```
docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation -h
```

```
Get evaluation information from MLModelScope

Usage:
  Caffe2-agent info evaluation [command]

Available Commands:
  accuracy    Get accuracy summary from MLModelScope
  all         Get all evaluation information from MLModelScope
  cuda_launch Get evaluation kernel launch information from MLModelScope
  duration    Get evaluation duration summary from MLModelScope
  eventflow   Get evaluation trace in event_flow format from MLModelScope
  latency     Get evaluation latency or throughput information from MLModelScope
  layer_tree  Get evaluation layer tree information from MLModelScope
  layers      Get evaluation layer information from MLModelScope

Flags:
      --append                     append the output
      --arch string                architecture of machine to filter
      --batch_size int             the batch size to filter
      --database_address string    address of the database
      --database_name string       name of the database to query
  -f, --format string              print format to use (default "table")
      --framework_name string      frameworkName
      --framework_version string   frameworkVersion
  -h, --help                       help for evaluation
      --hostname string            hostname of machine to filter
      --limit int                  limit the evaluations (default -1)
      --model_name string          modelName (default "BVLC-AlexNet")
      --model_version string       modelVersion (default "1.0")
      --no_header                  show header labels for output
  -o, --output string              output file name
      --overwrite                  if the file or directory exists, then they get deleted

Global Flags:
      --config string   config file (default is $HOME/.carml_config.yaml)
  -d, --debug           Toggle debug mode.
  -l, --local           Listen on local address.
      --profile         Enable profile mode.
  -s, --secret string   The application secret.
  -v, --verbose         Toggle verbose mode.

Use "Caffe2-agent info evaluation [command] --help" for more information about a command.
```

Available output formats are table, json and csv.

### Performance Information

To get latency information, run the following command.

```
docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation latency \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS
```

### Accuracy Information

To get accuracy information, run the following command.

```
docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation acurracy\
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS
```
