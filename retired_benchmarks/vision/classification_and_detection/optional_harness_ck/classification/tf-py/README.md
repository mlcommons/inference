# Image Classification via TensorFlow (Python)

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [Reference accuracy](#accuracy)
4. [Further information](#further-info)


<a name="installation"></a>
## Installation instructions

Please follow the [common installation instructions](../README.md#installation) first.

### Install TensorFlow (Python)

Install TensorFlow (Python) from an `x86_64` binary package:
```
$ ck install package:lib-tensorflow-1.13.1-cpu
```
or from source:
```
$ ck install package:lib-tensorflow-1.13.1-src-cpu
```

### Install the MobileNet model for TensorFlow (Python)

To select interactively from one of the non-quantized and quantized MobileNets-v1-1.0-224 models:
```
$ ck install package --tags=model,tf,mlperf,mobilenet
```

To install the [non-quantized model](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) directly:
```
$ ck install package --tags=model,tf,mlperf,mobilenet,non-quantized
```

To install the [quantized model](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz) directly:
```
$ ck install package --tags=model,tf,mlperf,mobilenet,quantized
```

#### Bonus

##### Install other MobileNets models
You can also install any other MobileNets model compatible with TensorFlow (Python) as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet --no_tags=mobilenet-all
```
**NB:** This excludes "uber" packages which can be used to install all models in the sets `v1-2018-02-22` (16 models), `v1[-2018-06-14]` (16 models) and `v2` (22 models) in one go:
```
$ ck search package --tags=tensorflowmodel,mobilenet-all
ck-tensorflow:package:tensorflowmodel-mobilenet-v1-2018_02_22
ck-tensorflow:package:tensorflowmodel-mobilenet-v2
ck-tensorflow:package:tensorflowmodel-mobilenet-v1
```

### Run the TensorFlow (Python) Image Classification client

Run the client:

- with the non-quantized model:
```
$ ck run program:image-classification-tf-py
...
*** Dependency 4 = weights (TensorFlow-Python model and weights):
...
    Resolved. CK environment UID = f934f3a3faaf4d73 (version 1_1.0_224_2018_02_22)
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.82 - (65) n01751748 sea snake
0.10 - (58) n01737021 water snake
0.04 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.855126s
All images loaded in 0.001089s
All images classified in 0.116698s
Average classification time: 0.116698s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

- with the quantized model:
```
$ ck run program:image-classification-tf-py
...
*** Dependency 4 = weights (TensorFlow-Python model and weights):
...
    Resolved. CK environment UID = b18ad885d440dc77 (version 1_1.0_224_quant_2018_08_02)
...
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.16 - (60) n01740131 night snake, Hypsiglena torquata
0.10 - (600) n03532672 hook, claw
0.07 - (58) n01737021 water snake
0.05 - (398) n02666196 abacus
0.05 - (79) n01784675 centipede
---------------------------------------

Summary:
-------------------------------
Graph loaded in 1.066851s
All images loaded in 0.001507s
All images classified in 0.178281s
Average classification time: 0.178281s
Accuracy top 1: 0.0 (0 of 1)
Accuracy top 5: 0.0 (0 of 1)
--------------------------------
```

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance
```
$ ck benchmark program:image-classification-tf-py \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--record --record_repo=local --record_uoa=mlperf-image-classification-mobilenet-tf-py-performance \
--tags=mlperf,image-classification,mobilenet,tf-py,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```bash
$ ck benchmark program:image-classification-tf-py \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
...
Weights loaded in 0.293122s

Batch 1 of 2
Batch loaded in 0.001036s
Batch classified in 0.121501s

Batch 2 of 2
Batch loaded in 0.001257s
Batch classified in 0.013995s
...
Summary:
-------------------------------
Graph loaded in 1.115745s
All images loaded in 0.002293s
All images classified in 0.013995s
Average classification time: 0.013995s
Accuracy top 1: 0.5 (1 of 2)
Accuracy top 5: 1.0 (2 of 2)
--------------------------------
```

### Benchmark the accuracy
```bash
$ ck benchmark program:image-classification-tf-py \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-image-classification-mobilenet-tf-py-accuracy \
--tags=mlperf,image-classification,mobilenet,tf-py,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** For the `imagenet-2012-val-min` dataset, change `--env.CK_BATCH_COUNT=50000`
to `--env.CK_BATCH_COUNT=500` (or drop completely to test on a single image as if
with `--env.CK_BATCH_COUNT=1`).


<a name="accuracy"></a>
## Reference accuracy
**TODO**

<a name="further-info"></a>
## Further information

### Using Collective Knowledge
See the [common instructions](../README.md) for information on how to use Collective Knowledge
to learn about [the anatomy of a benchmark](../README.md#anatomy), or
to inspect and visualize [experimental results](../README.md#results).

### Using the client program
See [`ck-tensorflow:program:image-classification-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-py) for more details about the client program.
