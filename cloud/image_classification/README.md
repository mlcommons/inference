# Mlperf cloud inference benchmark for image classification

This is the Mlperf cloud inference benchmark for image classification. 
The benchmark uses the resnet50 v1.5 model from [Mlperf training](https://github.com/mlperf/training/tree/master/image_classification)
which identical to the [official tensorflow resnet model](https://github.com/tensorflow/models/tree/master/official/resnet).
Models are provided for tensorflow as frozen graph (in NHWC data format) and as [ONNX](http://onnx.ai) model.

More information on resnet50 v1.5 can be found [here](https://github.com/tensorflow/models/tree/master/official/resnet).

We use the validation set from [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012/) do validate model accuracy.

Cloud inference benchmarks are looking for latency bound throughput: we are looking for the maximum qps the system can do while still
meeting a target latency in the 99 percentile.

## Disclaimer
This is an early version of the benchmark to get feedback from others.
Do expect some changes.

The benchmark is a reference implementation that is not meant to be the fastest implementation possible.
It is written in python which might make it less suitable for lite models like mobilenet or large number of cpu's.
We are thinking to provide a c++ implementation with identical functionality in the near future.

## Benchmark target numbers
```
Accuracy: 72.6% (TODO: check, it is suppose to be 76.47%)
99 pcercentile @ 10ms, 50ms, 100ms, 200ms
```

## Prerequisites and Installation
We support [tensorfow](https://github.com/tensorflow/tensorflow) and [onnxruntime](https://github.com/Microsoft/onnxruntime) backend's with the same benchmark tool.
Support for other backends can be easily added.

The following steps are only needed if you run the benchmark ```without docker```.

We require python 3.5 or 3.6 and recommend to use anaconda (See [Dockerfile](Dockerfile.cpu) for a minimal anaconda install).

Install the desired backend.
For tensorflow:
```
pip install tensorflow
or 
pip install tensorflow-gpu
```
For onnxruntime:
```
pip install onnxruntime
or
pip install onnxruntime-gpu
```

## Running the benchmark
### One time setup

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
The same dataset is used for other mlperf inference benchmarks that are using imagenet.
```
pip install ck
ck pull  repo:ck-env
ck install package:imagenet-2012-val-min
ck install package:imagenet-2012-aux
cp $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt
$HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
```

Download our pre-trained models. For tensorflow use:

https://zenodo.org/record/2535873/files/resnet50_v1.pb

This model uses the data format NHWC. If you need NCHW, please file an issue on github.

For ONNX use:

https://zenodo.org/record/2541184/files/resnet50_v1.onnx

This model uses ONNX opset 8. If you need other opset's, please file an issue on github. The model is directly converted
from the tensorflow model via https://github.com/onnx/tensorflow-onnx.

Both local and docker environment need to 2 environment variables set:
```
export MODEL_DIR=YourModelFileLocation
export DATA_DIR=YourImageNetLocation
```

### Run local
```
./run_local.sh backend cpu|gpu  # backend can be tf or onnxruntime

For example:

./run_local.sh tf gpu
```

### Run as Docker container
```
./run_and_time.sh backend cpu|gpu  # backend can be tf or onnxruntime

For example:

./run_and_time.sh tf gpu
```
This will build and run the benchmark.


### Results
The benchmark scans for the maximum qps and will take a couple of minutes to finish. 

There are 3 different stages the tool runs through:

1. Run the first 500 images to validate the target accuracy.
2. Find the maximum qps for the given latency in the 99 percentile using equal spacing between requests.
3. Find the maximum qps for the given latency in the 99 percentile using a exponential distribution that simulates realistic traffic patterns for a cloud environment.

Logs are written to stdout, while the benchmark is running. The have the form:
```
check_accuracy qps=39.07, mean=0.202488, time=12.82, tiles=50:0.0593,80:0.0613,90:0.0635,95:0.0677,99:8.9926,99.9:9.1235
check_accuracy accuacy=72.46, good_items=363, total_items=501
linear/0.01/100 qps=99.81, mean=0.008619, time=5.02, tiles=50:0.0085,80:0.0093,90:0.0095,95:0.0096,99:0.0098,99.9:0.0105
^taken
...
linear/0.01/129 qps=128.72, mean=0.008433, time=3.89, tiles=50:0.0083,80:0.0088,90:0.0092,95:0.0097,99:0.0105,99.9:0.0109
0.01 qps=127.73, mean=0.008219, time=3.92, tiles=50:0.0082,80:0.0084,90:0.0086,95:0.0089,99:0.0095,99.9:0.0097
===RESULT: linear target_latency=0.01 measured_latency=0.009488344192504883 qps=127
...
===FINAL_RESULT: {'linear': {'0.01': 128, '0.05': 142, '0.1': 142, '0.2': 142}, 'exponential': {'0.01': -1, '0.05': 104, '0.1': 126, '0.2': 132}}
```
Entries that start with ```check_accuracy``` report the accuracy.

Entries that start with ```linear``` report qps scans with equal spacing of requsts. ```linear/0.01/129 qps=128.72``` for example means tried we tried 129 requests/sec
in the 99 percentile and reached 128.72 requests/sec.

Entries that start with ```exponential``` report qps scans with exponential distribution that simulate a realistic traffic pattern.

```===RESULT``` will report the final result for each target latency.

```===FINAL_RESULT``` will summarize the results.

In the out directory we create a results.json file with the following format:
```
{
    "check_accuracy": {
        "accuacy": 72.45508982035928,
        ...
    },
    "results": {
        "exponential": {
            "0.05": {
                ...
                "qps": 115.00442176867026,
            },
            ...
        },
        ...
    },
    "scan": {
        ... more details about each scan we tried
    }
}
```

### Usage
```
usage: main.py [-h] [--dataset {imagenet}] --dataset-path DATASET_PATH
               [--dataset-list DATASET_LIST] [--data-format {NCHW,NHWC}]
               [--profile {defaults,resnet50-tf,resnet50-onnxruntime}] --model
               MODEL [--inputs INPUTS] [--output OUTPUT] [--outputs OUTPUTS]
               [--backend BACKEND] [--batch_size BATCH_SIZE]
               [--threads THREADS] [--time TIME] [--count COUNT]
               [--max-latency MAX_LATENCY] [--cache CACHE]
```
For example to run a quick test on 200ms in the 99 percentile for tensorflow you would do:
```
python python/main.py --profile resnet50-tf --count 500 --time 10 --model models/resnet50_v1.pb --dataset-path imagenet2012 --output results.json --max-latency 0.2
```

```--dataset```
use the specified dataset. Currently we only support imagenet.

```--dataset-path```
path to the dataset.

```--dataset-list DATASET_LIST```
the list of image names to be used. By default we look for val_map.txt in the dataset-path.

```--data-format {NCHW,NHWC}```
data-format of the model

```--profile {resnet50-tf,resnet50-onnxruntime}```
this fills in default command line options for the specific benchmark. Additional command line options may override the defautls.

```--model MODEL```
the model file.

```--inputs INPUTS```
comma seperated input name list in case the model format does not provide the input names. This is needed for tensorflow since the graph does not specify the inputs.

```--outputs OUTPUTS```
comma seperated output name list in case the model format does not provide the output names. This is needed for tensorflow since the graph does not specify the outputs.

```--output OUTPUT]```
location of the json output.

```--backend BACKEND```
which backend to use. Currently supported is tensorflow, onnxruntime, pytorch and tflite.
```--batch_size BATCH_SIZE```
batch size to use. Defaults to 1.

```--threads THREADS```
number of worker threads to use. This defaults to the number of processors in the system.

```--time TIME```
time how long we run each epoch. Defaults to 30 seconds.

```--count COUNT```
Number of images for each epoch. If not given we use the number of images in the dataset.

```--max-latency MAX_LATENCY```
comma seperated list of Which latencies (in seconds) we try to reach in the 99 percentile.
The deault is 0.010,0.050,0.100,0.200,0.400.

```--cache CACHE```
1 if we should pre-load images.


## License

[Apache License 2.0](LICENSE)
