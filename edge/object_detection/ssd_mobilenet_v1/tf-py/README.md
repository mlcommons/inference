# MobileNets via TensorFlow (Python)

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
$ ck run program:object-detection-tf-py --env.CK_BATCH_COUNT=5000 [--env.CK_METRIC_TYPE=COCO]
...
```
