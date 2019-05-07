# MobileNet via TensorFlow (C++)

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [Reference accuracy](#accuracy)
4. [Further information](#further-info)


<a name="installation"></a>
## Installation instructions

Please follow the [common installation instructions](../README.md#installation) first.

### Install TensorFlow (C++)

Install TensorFlow (C++) v1.13.1 from source:
```bash
$ ck install package:lib-tensorflow-1.13.1-src-static [--target_os=android23-arm64]
```

### Install the MobileNet model for TensorFlow (C++)

To select interactively from one of the non-quantized and quantized MobileNets-v1-1.0-224 models:
```
$ ck install package --tags=model,tf,mlperf,mobilenet
```

To install the [non-quantized model](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) directly:
```bash
$ ck install package --tags=model,tf,mlperf,mobilenet,non-quantized
```

To install the [quantized model](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) directly:
```bash
$ ck install package --tags=model,tf,mlperf,mobilenet,quantized
```

#### Bonus

##### Install the ResNet model

To install the ResNet50-v1.5 model:
```bash
$ ck install package --tags=model,tf,mlperf,resnet
```
You can use this model exactly in the same way as the MobileNet one.
Just replace `mobilenet` with `resnet` in the [benchmarking instructions](#benchmarking) below.

##### Install other MobileNets models

You can also install any other MobileNets model compatible with TensorFlow (C++) as follows:
```bash
$ ck install package --tags=tensorflowmodel,mobilenet,frozen --no_tags=mobilenet-all
```

### Compile the TensorFlow (C++) Image Classification client
```bash
$ ck compile program:image-classification-tf-cpp [--target_os=android23-arm64]
```

### Run the TensorFlow (C++) Image Classification client

Run the client (if required, connect an Android device to your host machine via USB):

- with the non-quantized model:
```bash
$ ck run program:image-classification-tf-cpp [--target_os=android23-arm64]
...
*** Dependency 3 = weights (TensorFlow model and weights):
...
    Resolved. CK environment UID = f934f3a3faaf4d73 (version 1_1.0_224_2018_02_22)
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.84 - (65) n01751748 sea snake
0.08 - (58) n01737021 water snake
0.04 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.037618s
All images loaded in 0.002786s
All images classified in 0.316360s
Average classification time: 0.316360s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

- with the quantized model:
```bash
$ ck run program:image-classification-tf-cpp [--target_os=android23-arm64]
...
*** Dependency 3 = weights (TensorFlow model and weights):
...
    Resolved. CK environment UID = b18ad885d440dc77 (version 1_1.0_224_quant_2018_08_02)
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.72 - (65) n01751748 sea snake
0.16 - (58) n01737021 water snake
0.05 - (54) n01729322 hognose snake, puff adder, sand viper
0.03 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (50) n01698640 American alligator, Alligator mississipi...
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.096074s
All images loaded in 0.004774s
All images classified in 0.568562s
Average classification time: 0.568562s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```


<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance
```
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--record --record_repo=local --record_uoa=mlperf-image-classification-mobilenet-tf-cpp-performance \
--tags=mlperf,image-classification,mobilenet,tf-cpp,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
...
Processing batches...

Batch 1 of 2
Batch loaded in 0.00341696 s
Batch classified in 0.355268 s

Batch 2 of 2
Batch loaded in 0.00335902 s
Batch classified in 0.0108837 s
...
Summary:
-------------------------------
Graph loaded in 0.053440s
All images loaded in 0.006776s
All images classified in 0.366151s
Average classification time: 0.010884s
Accuracy top 1: 0.5 (1 of 2)
Accuracy top 5: 1.0 (2 of 2)
--------------------------------
```

### Benchmark the accuracy
```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-image-classification-mobilenet-tf-cpp-accuracy \
--tags=mlperf,image-classification,mobilenet,tf-cpp,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** For the `imagenet-2012-val-min` dataset, change `--env.CK_BATCH_COUNT=50000`
to `--env.CK_BATCH_COUNT=500` (or drop completely to test on a single image as if
with `--env.CK_BATCH_COUNT=1`).


<a name="accuracy"></a>
## Reference accuracy

### ImageNet validation dataset (50,000 images)

#### 87.5% cropping (default)
```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-image-classification-tf-cpp-accuracy \
--tags=mlperf,image-classification,tf-cpp,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

##### MobileNet non-quantized
```
"accuracy_top1": 0.69966
"accuracy_top5": 0.89366
```

##### MobileNet quantized
```
"accuracy_top1": 0.68978
"accuracy_top5": 0.88628
```

##### ResNet
```
"accuracy_top1": 0.73288
"accuracy_top5": 0.91606
```

#### 100.0% cropping (makes the accuracy worse!)
```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 --env.CK_CROP_PERCENT=100 \
--record --record_repo=local --record_uoa=mlperf-image-classification-tf-cpp-accuracy-crop100 \
--tags=mlperf,image-classification,tf-cpp,accuracy,crop100 \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

##### MobileNet non-quantized
```
"accuracy_top1": 0.67698
"accuracy_top5": 0.87814
```

##### MobileNet quantized
```
"accuracy_top1": 0.66598
"accuracy_top5": 0.87096
```

##### ResNet
```bash
"accuracy_top1": 0.71360
"accuracy_top5": 0.90266
```


<a name="further-info"></a>
## Further information

### Using Collective Knowledge
See the [common instructions](../README.md) for information on how to use Collective Knowledge
to learn about [the anatomy of a benchmark](../README.md#anatomy), or
to inspect and visualize [experimental results](../README.md#results).

### Using the client program
See [`ck-tensorflow:program:image-classification-tf-cpp`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-cpp) for more details about the client program.
