# Image Classification via TensorFlow (C++)

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
**NB:** The ResNet model has a [known issue with v1.14.0](https://github.com/ctuning/ck-tensorflow/blob/master/package/lib-tensorflow-1.14.0-src-static/README.md).

### Install models for TensorFlow (C++)

#### ResNet

To install the [ResNet50-v1.5 model](https://zenodo.org/record/2535873):
```bash
$ ck install package --tags=model,tf,mlperf,resnet
```

#### MobileNet
To select interactively from one of the non-quantized and quantized MobileNets-v1-1.0-224 models:
```
$ ck install package --tags=model,tf,mlperf,mobilenet
```

##### MobileNet non-quantized

To install the non-quantized MobileNet model from:
- [zenodo.org](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224.tgz) (default):
```bash
$ ck install package --tags=model,tf,mlperf,mobilenet,non-quantized,from-zenodo
```
- [tensorflow.org](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)
```bash
$ ck install package --tags=model,tf,mlperf,mobilenet,non-quantized,from-google
```

##### MobileNet quantized

To install the quantized MobileNet model from:
- [zenodo.org](https://zenodo.org/record/2269307/files/mobilenet_v1_1.0_224_quant.tgz) (default):
```bash
$ ck install package --tags=model,tf,mlperf,mobilenet,quantized,from-zenodo
```
- [tensorflow.org](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)
```bash
$ ck install package --tags=model,tf,mlperf,mobilenet,quantized,from-google
```

##### Bonus: other MobileNets models

You can also install any other MobileNets model compatible with TensorFlow (C++) as follows:
```bash
$ ck install package --tags=tensorflowmodel,mobilenet,frozen --no_tags=mobilenet-all
```

### Compile the TensorFlow (C++) Image Classification client

Compile the client. (For Android, append e.g. `--target_os=android23-arm64` to the command.)

```bash
$ ck compile program:image-classification-tf-cpp --speed
```

### Run the TensorFlow (C++) Image Classification client

Run the client. (For Android, connect an Android device to your host machine via USB and append e.g. `--target_os=android23-arm64` to the command).

If you have preprocessed input data using more than one method (OpenCV, Pillow or TensorFlow), you need to select the particular preprocessed dataset. Note that the TensorFlow preprocessing method is not applicable to the MobileNet models.

#### ResNet

##### OpenCV preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
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

##### Pillow preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-pillow --dep_add_tags.weights=resnet
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.95 - (65) n01751748 sea snake
0.02 - (58) n01737021 water snake
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (60) n01740131 night snake, Hypsiglena torquata
0.01 - (66) n01753488 horned viper, cerastes, sand viper, horn...
---------------------------------------
```

##### TensorFlow preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-tf --dep_add_tags.weights=resnet
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.95 - (65) n01751748 sea snake
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (58) n01737021 water snake
0.01 - (66) n01753488 horned viper, cerastes, sand viper, horn...
0.00 - (60) n01740131 night snake, Hypsiglena torquata
---------------------------------------
```

#### MobileNet non-quantized

##### OpenCV preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
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

##### Pillow preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-pillow --dep_add_tags.weights=mobilenet,non-quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.87 - (65) n01751748 sea snake
0.06 - (34) n01665541 leatherback turtle, leatherback, leather...
0.04 - (58) n01737021 water snake
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------
```

##### TensorFlow preprocessing (**NOT APPLICABLE!**)
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-tf --dep_add_tags.weights=mobilenet,non-quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.67 - (616) n03627232 knot
0.08 - (584) n03476684 hair slide
0.06 - (488) n02999410 chain
0.02 - (792) n04208210 shovel
0.02 - (549) n03291819 envelope
---------------------------------------
```

#### MobileNet quantized

##### OpenCV preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.91 - (65) n01751748 sea snake
0.05 - (58) n01737021 water snake
0.03 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.00 - (57) n01735189 garter snake, grass snake
---------------------------------------
```

##### Pillow preprocessing
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-pillow --dep_add_tags.weights=mobilenet,quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.92 - (65) n01751748 sea snake
0.04 - (58) n01737021 water snake
0.02 - (34) n01665541 leatherback turtle, leatherback, leather...
0.00 - (390) n02526121 eel
0.00 - (54) n01729322 hognose snake, puff adder, sand viper
---------------------------------------
```

##### TensorFlow preprocessing (**NOT APPLICABLE!**)
```bash
$ ck run program:image-classification-tf-cpp \
--dep_add_tags.images=preprocessed,using-tf --dep_add_tags.weights=mobilenet,quantized
...
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.55 - (616) n03627232 knot
0.39 - (488) n02999410 chain
0.01 - (71) n01770393 scorpion
0.01 - (310) n02219486 ant, emmet, pismire
0.01 - (695) n03874599 padlock
---------------------------------------
```

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
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

#### ResNet
```
$ ck benchmark program:image-classification-tf-cpp --speed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-performance-using-opencv-resnet \
--tags=mlperf,image-classification,tf-cpp,performance,using-opencv,resnet \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=resnet
```

#### MobileNet non-quantized
```
$ ck benchmark program:image-classification-tf-cpp --speed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-performance-using-opencv-mobilenet-non-quantized \
--tags=mlperf,image-classification,tf-cpp,performance,using-opencv,mobilenet,non-quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
```

#### MobileNet quantized
```
$ ck benchmark program:image-classification-tf-cpp --speed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-performance-using-opencv-mobilenet-quantized \
--tags=mlperf,image-classification,tf-cpp,performance,using-opencv,mobilenet,quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,quantized
```


### Benchmark the accuracy

**NB:** For the `imagenet-2012-val-min` dataset, change `--env.CK_BATCH_COUNT=50000`
to `--env.CK_BATCH_COUNT=500` (or drop completely to test on a single image as if
with `--env.CK_BATCH_COUNT=1`).

#### ResNet
```bash
$ ck benchmark program:image-classification-tf-cpp --speed \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-accuracy-using-opencv-resnet \
--tags=mlperf,image-classification,tf-cpp,accuracy,using-opencv,resnet \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=resnet
```

#### MobileNet non-quantized
```bash
$ ck benchmark program:image-classification-tf-cpp --speed \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-accuracy-using-opencv-mobilenet-non-quantized \
--tags=mlperf,image-classification,tf-cpp,accuracy,using-opencv,mobilenet,non-quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
```

#### MobileNet quantized
```bash
$ ck benchmark program:image-classification-tf-cpp --speed \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-accuracy-using-opencv-mobilenet-quantized \
--tags=mlperf,image-classification,tf-cpp,accuracy,using-opencv,mobilenet,quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,quantized
```


<a name="accuracy"></a>
## Reference accuracy

### Example: universal OpenCV preprocessing (default), MobileNet non-quantized
```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--record_uoa=mlperf-image-classification-tf-cpp-accuracy-using-opencv-mobilenet-non-quantized \
--tags=mlperf,image-classification,tf-cpp,accuracy,using-opencv,mobilenet,non-quantized \
--dep_add_tags.images=preprocessed,using-opencv --dep_add_tags.weights=mobilenet,non-quantized
```

### ImageNet validation dataset (50,000 images)

```bash
$ ck benchmark program:image-classification-tf-cpp \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--skip_print_timers --skip_stat_analysis --process_multi_keys --record --record_repo=local \
--dep_add_tags.images=preprocessed,using-opencv \
--record_uoa=mlperf-image-classification-tf-cpp-accuracy \
--tags=mlperf,image-classification,tf-cpp,accuracy
```

| Model                   | Metric | Pillow  | OpenCV universal | OpenCV for MobileNet | OpenCV for ResNet | TensorFlow |
|-|-|-|-|-|-|-|
| ResNet                  |  Top1  | 0.76170 | 0.76422          | N/A                  | 0.76456           | 0.76522    |
|                         |  Top5  | 0.92866 | 0.93074          | N/A                  | 0.93016           | 0.93066    |
| MobileNet non-quantized |  Top1  | 0.71226 | 0.71676          | 0.71676              | N/A               | N/A        |
|                         |  Top5  | 0.89834 | 0.90118          | 0.90118              | N/A               | N/A        |
| MobileNet quantized     |  Top1  | 0.70348 | 0.70700          | 0.70694              | N/A               | N/A        |
|                         |  Top5  | 0.89376 | 0.89594          | 0.89594              | N/A               | N/A        |


<a name="further-info"></a>
## Further information

### Using Collective Knowledge
See the [common instructions](../README.md) for information on how to use Collective Knowledge
to learn about [the anatomy of a benchmark](../README.md#anatomy), or
to inspect and visualize [experimental results](../README.md#results).

### Using the client program
See [`ck-tensorflow:program:image-classification-tf-cpp`](https://github.com/ctuning/ck-tensorflow/tree/master/program/image-classification-tf-cpp) for more details about the client program.
