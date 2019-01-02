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

### Run the TFLite image classification client 

Run the client (if required, connect an Android device to your host machine via USB):
```
$ ck run program:image-classification-tflite [--target_os=android23-arm64]
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
Graph loaded in 0.000379s
All images loaded in 0.003440s
All images classified in 0.226376s
Average classification time: 0.226376s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```
