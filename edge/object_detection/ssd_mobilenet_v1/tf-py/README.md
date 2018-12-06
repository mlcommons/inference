# SSD-MobileNet-v1 via TensorFlow (Python)

Please follow the [installation instructions](../README.md#installation) first.

**NB:** See [`ck-tensorflow:program:object-detection-tf-py`](https://github.com/ctuning/ck-tensorflow/tree/master/program/object-detection-tf-py) for more details.

### Install TensorFlow (Python)

Install TensorFlow (Python) from an `x86_64` binary package (requires system `protobuf`):
```bash
$ sudo python3 -m pip install -U protobuf
$ ck install package:lib-tensorflow-1.10.1-cpu
```
or from source:
```bash
$ ck install package:lib-tensorflow-1.10.1-src-cpu
```

### Run the TensorFlow (Python) object detection client
```bash
$ ck run program:object-detection-tf-py --env.CK_BATCH_COUNT=50
...
********************************************************************************
* Process results
********************************************************************************

Convert results to coco ...

Evaluate metrics as coco ...
loading annotations into memory...
Done (t=3.70s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.01s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.10s).
Accumulating evaluation results...
DONE (t=0.17s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.281
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.262
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.156
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.595

Summary:
-------------------------------
Graph loaded in 0.608890s
All images loaded in 8.702114s
All images detected in 2.062680s
Average detection time: 0.042096s
mAP: 0.28063198419129565
Recall: 0.28815323348218086
--------------------------------
```
