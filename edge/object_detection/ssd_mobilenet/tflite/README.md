# SSD-MobileNet via TensorFlow Lite

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [Reference accuracy](#accuracy)

<a name="installation"></a>
## Installation instructions

Please follow the common [installation instructions](../README.md#installation) first.

**NB:** See [`ck-tensorflow:program:object-detection-tflite`](https://github.com/ctuning/ck-tensorflow/tree/master/program/object-detection-tflite) for more details about the client program.

### Install TensorFlow Lite (TFLite)

Install TFLite from source:
```
$ ck install package:lib-tflite-1.13.1-src-static [--target_os=android23-arm64]
```

**NB:** Currently we have no TFLite 1.13.1 prebuilt packages.
Please [let us know](info@dividiti.com) if you would like us to create some.


### Install the SSD-MobileNet models for TFLite

To select interactively from one of the non-quantized and quantized SSD-MobileNets-v1-1.0-224 models adopted for MLPerf Inference v0.5:
```
$ ck install package --tags=model,tflite,object-detection,mlperf,ssd-mobilenet
```

To install the non-quantized model directly:
```
$ ck install package --tags=model,tflite,object-detection,mlperf,ssd-mobilenet,non-quantized
```

### Compile the TFLite Object Detection client
```
$ ck compile program:object-detection-tflite [--target_os=android23-arm64]
```

### Run the TFLite Object Detection client on 50 images

Run the client (if required, connect an Android device to your host machine via USB):
```
$ ck run program:object-detection-tflite --env.CK_BATCH_COUNT=50 \
[--target_os=android23-arm64]
...
********************************************************************************
* Process results
********************************************************************************

Convert results to coco ...

Evaluate metrics as coco ...
loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.12s).
Accumulating evaluation results...
DONE (t=0.22s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.288
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.063
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668

Summary:
-------------------------------
Graph loaded in 0.000000s
All images loaded in 0.000000s
All images detected in 0.000000s
Average detection time: 0.000000s
mAP: 0.2883553845372072
Recall: 0.3022676916450782
--------------------------------
```
**NB:** We are working on resolving the difference in mAP between the TF and
TFLite versions (31.5% vs. 28.8%), as well as resolving the timing issue (all
zeros).

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance
```bash
$ ck benchmark program:object-detection-tflite \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-performance \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** When using the batch count of **N**, the program runs object detection
on **N** images, but the slow first run is not taken into account when
computing the average detection time.

### Benchmark the accuracy
```bash
$ ck benchmark program:object-detection-tflite \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tflite-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tflite,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

<a name="accuracy"></a>
## Reference accuracy

### SSD-MobileNet non-quantized
```
********************************************************************************
* Process results
********************************************************************************

Convert results to coco ...

Evaluate metrics as coco ...
loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=13.98s).
Accumulating evaluation results...
DONE (t=2.73s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.190
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.446
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.172
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570

Summary:
-------------------------------
Graph loaded in 0.000000s
All images loaded in 0.000000s
All images detected in 0.000000s
Average detection time: 0.000000s
mAP: 0.1899981780161538
Recall: 0.2431880894097952
--------------------------------
```
**NB:** We are working on resolving the difference in mAP between the TF and
TFLite versions (23.1% vs. 19.0%), as well as resolving the timing issue (all
zeros).
