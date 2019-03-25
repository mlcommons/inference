# MobileNets via TensorFlow (C++)

Please follow the common [installation instructions](../README.md#installation) first.

**NB:** See the [TFLIte instructions](../tflite/README.md) how to use Collective Knowledge to learn more about the anatomy of the benchmark.

**NB:** See [`ck-tensorflow:program:image-classification-tf-cpp`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-cpp) for more details about the client program.


### Install TensorFlow (C++)

Install TensorFlow (C++) from source:
```
$ ck install package:lib-tensorflow-1.13.1-src-static [--target_os=android23-arm64]
```

### Install the MobileNets model for TensorFlow (C++)

To select interactively from one of the non-quantized and quantized MobileNets-v1-1.0-224 models:
```
$ ck install package --tags=model,tf,mlperf,mobilenet
```

To install the non-quantized model directly:
```
$ ck install package --tags=model,tf,mlperf,mobilenet,non-quantized
```

To install the quantized model directly:
```
$ ck install package --tags=model,tf,mlperf,mobilenet,quantized
```

**NB:** You can also install any other MobileNets model compatible with TensorFlow (C++) as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet,frozen
```

### Compile the TensorFlow (C++) image classification client
```
$ ck compile program:image-classification-tf-cpp [--target_os=android23-arm64]
```

### Run the TensorFlow (C++) image classification client

Run the client (if required, connect an Android device to your host machine via USB):

- with the non-quantized model:
```
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
