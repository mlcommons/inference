# Object Detection via TensorFlow (Python)

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [Reference accuracy](#accuracy)
4. [Further information](#further-info)


<a name="installation"></a>
## Installation instructions

Please follow the common [installation instructions](../README.md#installation) first.

### Install additional Python packages in user-space
```
$ python -m pip install gast --user
$ python -m pip install astor --user
$ python -m pip install termcolor --user
$ python -m pip install tensorflow-estimator==1.13.0 --user
$ python -m pip install keras_applications==1.0.4 --no-deps --user
$ python -m pip install keras_preprocessing==1.0.2 --no-deps --user
```

### Install TensorFlow (Python)

Install TensorFlow (Python) from an `x86_64` binary package:
```bash
$ ck install package:lib-tensorflow-1.13.1-cpu
```
or from source:
```bash
$ ck install package:lib-tensorflow-1.13.1-src-cpu
```

### Install the SSD-MobileNet model

To select interactively from one of the non-quantized and quantized SSD-MobileNet models:
```
$ ck install package --tags=model,tf,object-detection,mlperf,ssd-mobilenet
```

#### Install the [non-quantized model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) directly
```bash
$ ck install package --tags=model,tf,object-detection,mlperf,ssd-mobilenet,non-quantized
```

#### Install the quantized finetuned model (courtesy of [Habana](https://habana.ai/)) directly
```bash
$ ck install package --tags=model,tf,object-detection,mlperf,ssd-mobilenet,quantized,finetuned
```

### Run the TensorFlow (Python) Object Detection client on 50 images
```bash
$ ck run program:object-detection-tf-py --env.CK_BATCH_COUNT=50
...
********************************************************************************
* Process results
********************************************************************************

Convert results to coco ...

Evaluate metrics as coco ...
loading annotations into memory...
Done (t=0.55s)
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
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.439
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.184
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.066
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.187
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711

Summary:
-------------------------------
Graph loaded in 0.748657s
All images loaded in 12.052568s
All images detected in 1.251245s
Average detection time: 0.025536s
mAP: 0.3148934914889957
Recall: 0.3225293342489256
--------------------------------
```

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance
```bash
$ ck benchmark program:object-detection-tf-py \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tf-py-performance \
--tags=mlperf,object-detection,ssd-mobilenet,tf-py,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** When using the batch count of **N**, the program runs object detection
on **N** images, but the slow first run is not taken into account when
computing the average detection time e.g.:
```bash
$ ck benchmark program:object-detection-tf-py \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
...
Graph loaded in 0.7420s

Detect image: 000000000139.jpg (1 of 2)
Detected in 1.9351s

Detect image: 000000000285.jpg (2 of 2)
Detected in 0.0284s
...
Summary:
-------------------------------
Graph loaded in 0.741997s
All images loaded in 0.604377s
All images detected in 0.028387s
Average detection time: 0.028387s
mAP: 0.15445544554455443
Recall: 0.15363636363636363
--------------------------------
```

### Benchmark the accuracy
```bash
$ ck benchmark program:object-detection-tf-py \
--repetitions=1 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=5000 --env.CK_METRIC_TYPE=COCO \
--record --record_repo=local --record_uoa=mlperf-object-detection-ssd-mobilenet-tf-py-accuracy \
--tags=mlperf,object-detection,ssd-mobilenet,tf-py,accuracy \
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
Done (t=0.49s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=12.46s).
Accumulating evaluation results...
DONE (t=2.09s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.252
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.262
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.023
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604

Summary:
-------------------------------
Graph loaded in 0.753200s
All images loaded in 1193.981655s
All images detected in 123.461871s
Average detection time: 0.024697s
mAP: 0.23111107753357035
Recall: 0.26304841188725403
--------------------------------
```

### SSD-MobileNet quantized finetuned
```
********************************************************************************
* Process results
********************************************************************************

Convert results to coco ...

Evaluate metrics as coco ...
loading annotations into memory...
Done (t=0.48s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.18s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=12.74s).
Accumulating evaluation results...
DONE (t=2.13s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.546
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.268
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.269
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.025
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.618

Summary:
-------------------------------
Graph loaded in 1.091316s
All images loaded in 1150.996266s
All images detected in 122.103661s
Average detection time: 0.024426s
mAP: 0.23594222525632427
Recall: 0.26864982712779556
--------------------------------
```


<a name="further-info"></a>
## Further information

### Using Collective Knowledge
See the [common MobileNet instructions](../../../object_classification/mobilenets/README.md) for information on how to use Collective Knowledge
to learn about [the anatomy of a benchmark](../../../object_clasification/mobilenets/README.md#the-anatomy-of-a-benchmark), or
to inspect and visualize [experimental results](../../../object_clasification/mobilenets/README.md#inspecting-recorded-experimental-results).

### Using the client program
See [`ck-tensorflow:program:object-detection-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/object-detection-tf-py) for more details about the client program.
