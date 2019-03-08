# MobileNets via TensorFlow (Python)

Please follow the common [installation instructions](../README.md#installation) first.

**NB:** See [`ck-tensorflow:program:image-classification-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-py) for more details.

### Install TensorFlow (Python)

Install TensorFlow (Python) from an `x86_64` binary package (requires system `protobuf`):
```
$ sudo python3 -m pip install -U protobuf
$ ck install package:lib-tensorflow-1.10.1-cpu
```
or from source:
```
$ ck install package:lib-tensorflow-1.10.1-src-cpu
```

### Install the MobileNets model for TensorFlow (Python)

Install the MobileNets-v1-1.0-224 model adopted for MLPerf Inference v0.5:
```
$ ck install package --tags=tensorflowmodel,mobilenet,mlperf
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
```
$ ck run program:image-classification-tf-py
...
*** Dependency 4 = weights (TensorFlow-Python model and weights):
    ...
    Resolved. CK environment UID = f934f3a3faaf4d73 (version 1_1.0_224_2018_02_22)
...
---------------------------------------
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
