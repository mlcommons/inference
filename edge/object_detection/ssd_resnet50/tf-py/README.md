# SSD-ResNet50 via TensorFlow (Python)

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
Done (t=5.54s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.68s).
Accumulating evaluation results...
DONE (t=0.31s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.574
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.489
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708

Summary:
-------------------------------
Graph loaded in 1.428280s
All images loaded in 14.183870s
All images detected in 158.842161s
Average detection time: 3.241677s
mAP: 0.4203664230794217
Recall: 0.44792078946857344
--------------------------------
```

