# Image Classification via TensorFlow Lite (TFLite)

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [Reference accuracy](#accuracy)
4. [Further information](#further-info)


<a name="installation"></a>
## Installation instructions

Please follow the [common installation instructions](../README.md#installation) first.

### Install TFLite

Install TFLite v1.13.1 from source:
```
$ ck install package --tags=lib,tflite,v1.13.1,vsrc [--target_os=android23-arm64]
```
**NB:** TFLite v1.14.0 has [many known issues on Arm platforms](https://github.com/ctuning/ck-tensorflow/blob/master/package/lib-tflite-1.14.0-src-static/README.md), and does not work for Android yet.

You can also install TFLite v0.1.7 from a prebuilt binary package for your target e.g.:
```
$ ck list package:lib-tflite-prebuilt*
lib-tflite-prebuilt-0.1.7-linux-aarch64
lib-tflite-prebuilt-0.1.7-linux-x64
lib-tflite-prebuilt-0.1.7-android-arm64
$ ck install package:lib-tflite-prebuilt-0.1.7-android-arm64 [--target_os=android23-arm64]
```

**NB:** Please [let us know](info@dividiti.com) if you would like us to create
prebuilt packages for TFLite 1.13.1.


### Install the models for TFLite

#### ResNet

To install the ResNet50-v1.5 model:
```bash
$ ck install package --tags=model,tflite,mlperf,resnet

More than one package or version found:

 0) model-tflite-mlperf-resnet-no-argmax  Version 1.5  (afb43014ef38f646)
 1) model-tflite-mlperf-resnet  Version 1.5  (d60d4e9a84151271)
 2) model-tflite-convert-from-tf (35e84375ac48dcb1), Variations: resnet

Please select the package to install [ hit return for "0" ]:
```

Option 0 will download a TFLite model preconverted from the TF model.  During
the conversion, the `ArgMax` operator causing an
[issue](https://github.com/ARM-software/armnn/issues/150) with ArmNN v19.02
and v19.05 was excluded.

Option 1 will download a TFLite model preconverted from the TF model, but
including the `ArgMax` operator. This variant can be used with ArmNN once
the above issue is resolved.

Option 2 will download the TF model and convert it to TFLite, while excluding
the `ArgMax` operator.


#### MobileNet

To select interactively from one of the non-quantized and quantized
MobileNets-v1-1.0-224 models:
```
$ ck install package --tags=model,tflite,mlperf,mobilenet

More than one package or version found:

 0) model-tf-mlperf-mobilenet  Version 1_1.0_224_2018_08_02  (05c4dcbbbf872ecf)
 1) model-tf-mlperf-mobilenet-quantized  Version 1_1.0_224_quant_2018_08_02  (3013bdc96184bf3b)
 2) model-tflite-convert-from-tf (35e84375ac48dcb1), Variations: mobilenet

Please select the package to install [ hit return for "0" ]:
```
Options 0 and 1 will download the official non-quantized and quantized models.
Option 2 will download the official TF model and convert it to TFLite.

**NB:** Option 2 is only viable on x86 platforms, as it depends on using a
prebuilt version of TF.  While this constraint could be relaxed to use a
version of TF built from source, building TF from source takes a long time on
Arm platforms (as well as [not being officially
supported](https://github.com/tensorflow/tensorflow/issues/25607#issuecomment-466583730)).

##### MobileNet non-quantized

To install the non-quantized MobileNet model from:
- [zenodo.org](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) (default):
```bash
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized,from-zenodo
```
- [tensorflow.org](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)
```bash
$ ck install package --tags=model,tflite,mlperf,mobilenet,non-quantized,from-google
```

##### MobileNet quantized

To install the quantized MobileNet model from:
- [zenodo.org](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224_quant.tgz) (default):
```bash
$ ck install package --tags=model,tflite,mlperf,mobilenet,quantized,from-zenodo
```
- [tensorflow.org](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)
```bash
$ ck install package --tags=model,tflite,mlperf,mobilenet,quantized,from-google
```

#### Bonus: other MobileNets models
You can also install any other MobileNets model compatible with TFLite as follows:
```
$ ck install package --tags=tensorflowmodel,mobilenet,tflite
```

### Compile the TFLite Image Classification client

Compile the client. (For Android, append e.g. `--target_os=android23-arm64` to the command.)

```bash
$ ck compile program:image-classification-tflite --speed
```

### Run the TFLite Image Classification client

Run the client. (For Android, connect an Android device to your host machine via USB and append e.g. `--target_os=android23-arm64` to the command).

If you have preprocessed input data using more than one method (OpenCV, Pillow or TensorFlow), you need to select the particular preprocessed dataset. Note that the TensorFlow preprocessing method is not applicable to the MobileNet models.

#### ResNet

##### OpenCV preprocessing
```bash
$ ck run program:image-classification-tflite \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=resnet
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.95 - (65) n01751748 sea snake
0.01 - (58) n01737021 water snake
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (66) n01753488 horned viper, cerastes, sand viper, horn...
0.00 - (60) n01740131 night snake, Hypsiglena torquata
---------------------------------------
```

#### MobileNet non-quantized

##### OpenCV preprocessing
```bash
$ ck run program:image-classification-tflite \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.86 - (65) n01751748 sea snake
0.05 - (58) n01737021 water snake
0.04 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------
```

#### MobileNet quantized

##### OpenCV preprocessing
```bash
$ ck run program:image-classification-tflite \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.88 - (65) n01751748 sea snake
0.07 - (34) n01665541 leatherback turtle, leatherback, leather...
0.03 - (58) n01737021 water snake
0.00 - (54) n01729322 hognose snake, puff adder, sand viper
0.00 - (0) n01440764 tench, Tinca tinca
---------------------------------------
```
**NB:** The prediction from `tflite` differs from that from `tf-cpp`.

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```
$ ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
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

#### ResNet
```
$ ck benchmark program:image-classification-tflite --speed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-performance-using-opencv-resnet \
--tags=mlperf,image-classification,tflite,performance,using-opencv,resnet \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=resnet
```

#### MobileNet non-quantized
```
$ ck benchmark program:image-classification-tflite --speed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-performance-using-opencv-mobilenet-non-quantized \
--tags=mlperf,image-classification,tflite,performance,using-opencv,mobilenet,non-quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
```

#### MobileNet quantized
```
$ ck benchmark program:image-classification-tflite --speed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-performance-using-opencv-mobilenet-quantized \
--tags=mlperf,image-classification,tflite,performance,using-opencv,mobilenet,quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,quantized
```

### Benchmark the accuracy

**NB:** For the `imagenet-2012-val-min` dataset, change `--env.CK_BATCH_COUNT=50000`
to `--env.CK_BATCH_COUNT=500` (or drop completely to test on a single image as if
with `--env.CK_BATCH_COUNT=1`).

#### ResNet
```bash
$ ck benchmark program:image-classification-tflite --speed \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-accuracy-using-opencv-resnet \
--tags=mlperf,image-classification,tflite,accuracy,using-opencv,resnet \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=resnet
```

#### MobileNet non-quantized
```bash
$ ck benchmark program:image-classification-tflite --speed \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-accuracy-using-opencv-mobilenet-non-quantized \
--tags=mlperf,image-classification,tflite,accuracy,using-opencv,mobilenet,non-quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
```

#### MobileNet quantized
```bash
$ ck benchmark program:image-classification-tflite --speed \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-accuracy-using-opencv-mobilenet-quantized \
--tags=mlperf,image-classification,tflite,accuracy,using-opencv,mobilenet,quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,quantized
```


<a name="accuracy"></a>
## Reference accuracy

### Example: OpenCV preprocessing (default), MobileNet non-quantized
```bash
$ ck benchmark program:image-classification-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tflite-accuracy-using-opencv-mobilenet-non-quantized \
--tags=mlperf,image-classification,tflite,accuracy,using-opencv,mobilenet,non-quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
```

### ImageNet validation dataset (50,000 images)

| Model                   | Metric | Pillow  | OpenCV universal | OpenCV for MobileNet                                         | OpenCV for ResNet | TensorFlow |
|-|-|-|-|-|-|-|
| ResNet                  |  Top1  | 0.76170 | 0.76422          | N/A                                                          | 0.76456           | 0.76522    |
|                         |  Top5  | 0.92866 | 0.93074          | N/A                                                          | 0.93016           | 0.93066    |
| MobileNet non-quantized |  Top1  | 0.71226 | 0.71676          | 0.71676                                                      | N/A               | N/A        |
|                         |  Top5  | 0.89834 | 0.90118          | 0.90118                                                      | N/A               | N/A        |
| MobileNet quantized     |  Top1  | 0.70502 | 0.70762          | N/A ([bug?](https://github.com/ctuning/ck-mlperf/issues/40)) | N/A               | N/A        |
|                         |  Top5  | 0.89118 | 0.89266          | N/A ([bug?](https://github.com/ctuning/ck-mlperf/issues/40)) | N/A               | N/A        |


<a name="further-info"></a>
## Further information
### Using Collective Knowledge
See the [common instructions](../README.md) for information on how to use Collective Knowledge
to learn about [the anatomy of a benchmark](../README.md#anatomy), or
to inspect and visualize [experimental results](../README.md#results).

### Using the client program
See [`ck-tensorflow:program:image-classification-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tflite) for more details about the client program.
