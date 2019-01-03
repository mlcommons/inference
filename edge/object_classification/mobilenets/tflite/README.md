# MobileNets via TensorFlow Lite

Please follow the common [installation instructions](../README.md#installation) first.

**NB:** See [`ck-tensorflow:program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite) for more details.

### Install TensorFlow Lite (TFLite)

Install TFLite from source:
```
$ ck install package:lib-tflite-0.1.7-src-static [--target_os=android23-arm64]
```

You can also install TFLite from a prebuilt binary package for your target e.g.:
```
$ ck list package:lib-tflite-prebuilt*
lib-tflite-prebuilt-0.1.7-linux-aarch64
lib-tflite-prebuilt-0.1.7-linux-x64
lib-tflite-prebuilt-0.1.7-android-arm64
$ ck install package:lib-tflite-prebuilt-0.1.7-android-arm64 [--target_os=android23-arm64]
```

### Install the MLPerf MobileNets model for TFLite

Install the MobileNets-v1-1.0-224 model adopted for MLPerf:
```
$ ck install package --tags=tensorflowmodel,mobilenet,mlperf
```

**NB:** You can also install any other MobileNets models compatible with TFLite as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet,tflite
```

### Compile the TFLite image classification client 
```
$ ck compile program:image-classification-tflite [--target_os=android23-arm64]
```

### Run the TFLite image classification client once

Run the client (if required, connect an Android device to your host machine via USB):
```
$ ck run program:image-classification-tflite [--target_os=android23-arm64]
...
*** Dependency 3 = weights (TensorFlow-Python model and weights):

    Resolved. CK environment UID = 4edbb2648a48d94d (version 1_1.0_224_2018_08_02)
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
Graph loaded in 0.001723s
All images loaded in 0.025555s
All images classified in 0.391207s
Average classification time: 0.391207s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

### Benchmark the performance
```
$ ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-v1-1.00-224-tflite-0.1.7-performance \
--tags=mlperf,image-classification,mobilenet-v1-1.0-224,tflite-0.1.7,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```
$ ck benchmark program:image-classification-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
...
Processing batches...

Batch 1 of 2

Batch loaded in 0.00802251 s
Batch classified in 0.16831 s

Batch 2 of 2

Batch loaded in 0.00776105 s
Batch classified in 0.0762354 s
...
Summary:
-------------------------------
Graph loaded in 0.000663s
All images loaded in 0.015784s
All images classified in 0.244545s
Average classification time: 0.076235s
Accuracy top 1: 0.5 (1 of 2)
Accuracy top 5: 1.0 (2 of 2)
--------------------------------
```

### Benchmark the accuracy
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-mobilenet-v1-1.00-224-tflite-0.1.7-accuracy \
--tags=mlperf,image-classification,mobilenet-v1-1.0-224,tflite-0.1.7,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
