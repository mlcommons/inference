# MobileNet via TensorFlow (Python)

Please follow the common [installation instructions](../README.md#installation) first.

**NB:** See the [TFLite instructions](../tflite/README.md) how to use Collective Knowledge to learn more about the anatomy of the benchmark.

**NB:** See [`ck-tensorflow:program:image-classification-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-py) for more details about the client program.


### Install TensorFlow (Python)

Install TensorFlow (Python) from an `x86_64` binary package (may still require system `protobuf`):
```
$ sudo python3 -m pip install -U protobuf
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


**NB:** You can also install any other MobileNets model compatible with TensorFlow (Python) as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet --no_tags=mobilenet-all
```
This excludes "uber" packages which can be used to install all models in the sets `v1-2018-02-22` (16 models), `v1[-2018-06-14]` (16 models) and `v2` (22 models) in one go:
```
$ ck search package --tags=tensorflowmodel,mobilenet-all
ck-tensorflow:package:tensorflowmodel-mobilenet-v1-2018_02_22
ck-tensorflow:package:tensorflowmodel-mobilenet-v2
ck-tensorflow:package:tensorflowmodel-mobilenet-v1
```

### Run the TensorFlow (Python) image classification client

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
