# SSD-MobileNet-v1 via TensorFlow (Python)

Please follow the [installation instructions](../README.md#installation) first.

**NB:** See [`ck-tensorflow:program:object-detection-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/object-detection-tf-py) for more details.

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
$ ck install package --tags=model,object-detection,mlperf,ssd-mobilenet
```

To install the [non-quantized model](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) directly:
```bash
$ ck install package --tags=model,tf,object-detection,mlperf,ssd-mobilenet,non-quantized
```

To install the quantized finetuned model (courtesy of [Habana](https://habana.ai/)) directly:
```bash
$ ck install package --tags=model,tf,object-detection,mlperf,ssd-mobilenet,quantized,finetuned
```

### Run the TensorFlow (Python) Object Detection client
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
