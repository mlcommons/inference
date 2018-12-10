# Caffe2/ONNX ShuffleNet v1.3 Evaluation with [MLModelScope](http://docs.mlmodelscope.org)

## Model Information and Environment

A model manifest is defined in YAML format and contains all the information needed to reproduce a model’s evaluation results. It tells MLModelScope the HW/SW stack to instantiate and how to evaluate the model.

The manifest for this model is at [ShuffleNet_v1.3_ONNX](https://github.com/rai-project/caffe2/blob/master/builtin_models/ShuffleNet_v1.3_ONNX.yml).

The environment is run within the [Caffe2](https://github.com/rai-project/go-caffe2/blob/master/dockerfiles/Dockerfile.amd64_cpu) dockerfile.

## Evaluation

The following command outputs the available evaluation commands in MLModelScope.

```
docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest predict -h
```

```
Predicts using the MLModelScope agent

Usage:
  Caffe2-agent predict [command]

Available Commands:
  dataset     Evaluates the dataset using the specified model and framework
  max-qps     Finds the maximun qps of the system using the specified model, framework and workload
  urls        Evaluates the urls using the specified model and framework
  workload    Evaluates the workload using the specified model and framework

Flags:
  -b, --batch_size int                        the batch size to use while performing inference (default 64)
      --database_address database.endpoints   the address of the mongo database to store the results. By default the address in the config database.endpoints is used
      --database_name app.name                the name of the database to publish the evaluation results to. By default the app name in the config app.name is used
      --fail_on_error                         turning on causes the process to terminate/exit upon first inference error. This is useful since some inferences will result in an error because they run out of memory
      --gpu                                   whether to enable the gpu. An error is returned if the gpu is not available
  -h, --help                                  help for predict
      --model_name string                     the name of the model to use for prediction (default "BVLC-AlexNet")
      --model_version string                  the version of the model to use for prediction (default "1.0")
  -p, --partition_list_size int               the chunk size to partition the input list. By default this is the same as the batch size
      --publish                               whether to publish the evaluation to database. Turning this off will not publish anything to the database. This is ideal for using carml within profiling tools or performing experiments where the terminal output is sufficient.
      --publish_predictions                   whether to publish prediction results to database. This will store all the probability outputs for the evaluation in the database which could be a few gigabytes of data for one dataset
      --trace_level string                    the trace level to use while performing evaluations (default "APPLICATION_TRACE")
      --tracer_address string                 the address of the jaeger or the zipking trace server (default "localhost:16686")

Global Flags:
      --config string   config file (default is $HOME/.carml_config.yaml)
  -d, --debug           Toggle debug mode.
  -l, --local           Listen on local address.
      --profile         Enable profile mode.
  -s, --secret string   The application secret.
  -v, --verbose         Toggle verbose mode.

Use "Caffe2-agent predict [command] --help" for more information about a command.
```

The example comands are in [example.sh](example.sh).

### Performance

You can evaluate the performance on CPU using the following script:

```bash
#!/bin/bash

DATABASE_ADDRESS=localhost # the ip of database to publish traces to
DATABASE_NAME=shufflenet_model_trace # the name of database to publish traces to
MODEL_NAME=ShuffleNet_Caffe2 # model name
MODEL_VERSION=1.0 # model version
NUM_FILE_PARTS=100 # number of batches to be processed, set to -1 to evalute the entire ImageNet
BATCH_SIZE=1 # batch size
TRACE_LEVEL=MODEL_TRACE # trace level
TRACER_ADDRESS=localhost # the ip of tracer

docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=true \
      --publish_predictions=false \
      --gpu=false \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS \
      --trace_level=$TRACE_LEVEL \
      --tracer_address-$TRACER_ADDRESS
```

If you need to run with gpu enabled then you need to use `nvidia-docker` instead of `docker`, change the docker image tag from `amd64-cpu-latest` to `amd64-gpu-latest` and then set the `gpu` option to `true`.

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
docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest info evaluation accuracy\
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_name=$DATABASE_NAME \
      --database_address=$DATABASE_ADDRESS
```
