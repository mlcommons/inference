# Image Classification via ONNX

1. [Installation instructions](#installation)
2. [Benchmarking instructions](#benchmarking)
3. [Reference accuracy](#accuracy)
4. [Further information](#further-info)


<a name="installation"></a>
## Installation instructions

Please follow the [common installation instructions](../README.md#installation) first.

### Install ONNX

Install the ONNX library and runtime:
```
$ ck install package --tags=lib,python-package,onnx
$ ck install package --tags=lib,python-package,onnxruntime
```

### Install the MobileNet model for ONNX

To select interactively from one of the non-quantized and quantized MobileNets-v1-1.0-224 models:
```
$ ck install package --tags=model,onnx,mlperf,mobilenet

More than one package or version found:

 0) model-onnx-mlperf-mobilenet  Version 1_1.0_224_2018_08_02  (b47f4980eefabffa)
 1) model-onnx-convert-from-tf (22b1d864174bf743), Variations: mobilenet

Please select the package to install [ hit return for "0" ]: 
```
Option 1 downloads the TF model and converts it to ONNX. Option 0 uses a pre-converted ONNX model.

We recommend [converting models on-the-fly](https://github.com/ctuning/ck-mlperf/blob/master/package/model-onnx-convert-from-tf/README.md), as you can additionally control the data layout as follows:
- NHWC:
```
$ ck install package --tags=onnx,model,mobilenet,converted,nhwc
```
- NCHW
```
$ ck install package --tags=onnx,model,mobilenet,converted,nchw
```
Note that without the layout tags (`nhwc` or `nchw`), the layout is selected nondeterministically.

#### Bonus

##### Install the ResNet model

You can similarly convert ResNet as follows:
- NHWC
```
$ ck install package --tags=onnx,model,resnet,converted,nhwc
```
- NCHW
```
$ ck install package --tags=onnx,model,resnet,converted,nchw
```

You can benchmark ResNet exactly in the same way as MobileNet.
Just replace `mobilenet` with `resnet` in the [benchmarking instructions](#benchmarking) below.


### Run the ONNX Image Classification client

#### MobileNet, NHWC
```
$ ck run program:image-classification-onnx-py
...
*** Dependency 3 = weights (ONNX model):

More than one environment found for "ONNX model" with tags="model,image-classification,onnx" and setup={"host_os_uoa": "linux-64", "target_os_uoa": "linux-64", "target_os_bits": "64"}:

 0) ONNX-from-TF model (MLPerf MobileNet) - v1_1.0_224_2018_08_02 (64bits,converted,converted-from-tf,host-os-linux-64,image-classification,mlperf,mobilenet,model,nhwc,onnx,target-os-linux-64,v1,v1.1,v1.1
.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2 (f18d48538fbfbd46))
                                       - Depends on "python" (env UOA=7c8bbf2343208d88, tags="compiler,python", version=3.6.7)
                                       - Depends on "lib-python-numpy" (env UOA=fe9d0436cbfd34c8, tags="lib,python-package,numpy", version=1.16.2)
                                       - Depends on "lib-tensorflow" (env UOA=9c34f3f9b9b8dfd4, tags="lib,tensorflow,vprebuilt", version=1.13.1)
                                       - Depends on "lib-python-onnx" (env UOA=c9a3c5ad5de9adcb, tags="lib,python-package,onnx", version=1.4.1)
                                       - Depends on "lib-python-tf2onnx" (env UOA=44dd6b520ae81482, tags="lib,python-package,tf2onnx", version=1.4.1)
                                       - Depends on "model-source" (env UOA=e5cf6f254447a629, tags="model,image-classification,tf", version=1_1.0_224_2018_08_02)

 1) ONNX-from-TF model (MLPerf MobileNet) - v1_1.0_224_2018_08_02 (64bits,converted,converted-from-tf,host-os-linux-64,image-classification,mlperf,mobilenet,model,nchw,onnx,target-os-linux-64,v1,v1.1,v1.1
.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2 (2e1b5534351b7e33))
                                       - Depends on "python" (env UOA=7c8bbf2343208d88, tags="compiler,python", version=3.6.7)
                                       - Depends on "lib-python-numpy" (env UOA=fe9d0436cbfd34c8, tags="lib,python-package,numpy", version=1.16.2)
                                       - Depends on "lib-tensorflow" (env UOA=9c34f3f9b9b8dfd4, tags="lib,tensorflow,vprebuilt", version=1.13.1)
                                       - Depends on "lib-python-onnx" (env UOA=c9a3c5ad5de9adcb, tags="lib,python-package,onnx", version=1.4.1)
                                       - Depends on "lib-python-tf2onnx" (env UOA=44dd6b520ae81482, tags="lib,python-package,tf2onnx", version=1.4.1)
                                       - Depends on "model-source" (env UOA=e5cf6f254447a629, tags="model,image-classification,tf", version=1_1.0_224_2018_08_02)


Select one of the options for "ONNX model" with tags="model,image-classification,onnx" and setup={"host_os_uoa": "linux-64", "target_os_uoa": "linux-64", "target_os_bits": "64"} [ hit return for "0" ]: 0

    Resolved. CK environment UID = f18d48538fbfbd46 (version 1_1.0_224_2018_08_02)
...
--------------------------------
Process results in predictions
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.84 - (65) n01751748 sea snake
0.08 - (58) n01737021 water snake
0.04 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.018571s
All images loaded in 0.001246s
All images classified in 0.188061s
Average classification time: 0.188061s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

#### MobileNet, NCHW
```
$ ck run program:image-classification-onnx-py
...
*** Dependency 3 = weights (ONNX model):

More than one environment found for "ONNX model" with tags="model,image-classification,onnx" and setup={"host_os_uoa": "linux-64", "target_os_uoa": "linux-64", "target_os_bits": "64"}:

 0) ONNX-from-TF model (MLPerf MobileNet) - v1_1.0_224_2018_08_02 (64bits,converted,converted-from-tf,host-os-linux-64,image-classification,mlperf,mobilenet,model,nhwc,onnx,target-os-linux-64,v1,v1.1,v1.1
.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2 (f18d48538fbfbd46))
                                       - Depends on "python" (env UOA=7c8bbf2343208d88, tags="compiler,python", version=3.6.7)
                                       - Depends on "lib-python-numpy" (env UOA=fe9d0436cbfd34c8, tags="lib,python-package,numpy", version=1.16.2)
                                       - Depends on "lib-tensorflow" (env UOA=9c34f3f9b9b8dfd4, tags="lib,tensorflow,vprebuilt", version=1.13.1)
                                       - Depends on "lib-python-onnx" (env UOA=c9a3c5ad5de9adcb, tags="lib,python-package,onnx", version=1.4.1)
                                       - Depends on "lib-python-tf2onnx" (env UOA=44dd6b520ae81482, tags="lib,python-package,tf2onnx", version=1.4.1)
                                       - Depends on "model-source" (env UOA=e5cf6f254447a629, tags="model,image-classification,tf", version=1_1.0_224_2018_08_02)

 1) ONNX-from-TF model (MLPerf MobileNet) - v1_1.0_224_2018_08_02 (64bits,converted,converted-from-tf,host-os-linux-64,image-classification,mlperf,mobilenet,model,nchw,onnx,target-os-linux-64,v1,v1.1,v1.1
.0,v1.1.0.224,v1.1.0.224.2018,v1.1.0.224.2018.8,v1.1.0.224.2018.8.2 (2e1b5534351b7e33))
                                       - Depends on "python" (env UOA=7c8bbf2343208d88, tags="compiler,python", version=3.6.7)
                                       - Depends on "lib-python-numpy" (env UOA=fe9d0436cbfd34c8, tags="lib,python-package,numpy", version=1.16.2)
                                       - Depends on "lib-tensorflow" (env UOA=9c34f3f9b9b8dfd4, tags="lib,tensorflow,vprebuilt", version=1.13.1)
                                       - Depends on "lib-python-onnx" (env UOA=c9a3c5ad5de9adcb, tags="lib,python-package,onnx", version=1.4.1)
                                       - Depends on "lib-python-tf2onnx" (env UOA=44dd6b520ae81482, tags="lib,python-package,tf2onnx", version=1.4.1)
                                       - Depends on "model-source" (env UOA=e5cf6f254447a629, tags="model,image-classification,tf", version=1_1.0_224_2018_08_02)


Select one of the options for "ONNX model" with tags="model,image-classification,onnx" and setup={"host_os_uoa": "linux-64", "target_os_uoa": "linux-64", "target_os_bits": "64"} [ hit return for "0" ]: 1

    Resolved. CK environment UID = 2e1b5534351b7e33 (version 1_1.0_224_2018_08_02)
...
--------------------------------
Process results in predictions
---------------------------------------
ILSVRC2012_val_00000001.JPEG - (65) n01751748 sea snake
0.84 - (65) n01751748 sea snake
0.08 - (58) n01737021 water snake
0.04 - (34) n01665541 leatherback turtle, leatherback, leather...
0.01 - (54) n01729322 hognose snake, puff adder, sand viper
0.01 - (57) n01735189 garter snake, grass snake
---------------------------------------

Summary:
-------------------------------
Graph loaded in 0.018411s
All images loaded in 0.001247s
All images classified in 0.189969s
Average classification time: 0.189969s
Accuracy top 1: 1.0 (1 of 1)
Accuracy top 5: 1.0 (1 of 1)
--------------------------------
```

<a name="benchmarking"></a>
## Benchmarking instructions

### Benchmark the performance
```
$ ck benchmark program:image-classification-onnx-py --cmd_key=preprocessed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2 \
--record --record_repo=local --record_uoa=mlperf-image-classification-mobilenet-onnx-py-performance \
--tags=mlperf,image-classification,mobilenet,onnx-py,performance \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```

**NB:** When using the batch count of **N**, the program classifies **N** images, but
the slow first run is not taken into account when computing the average
classification time e.g.:
```bash
$ ck benchmark program:image-classification-onnx-py --cmd_key=preprocessed \
--repetitions=10 --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=2
...
Batch 1 of 2
Batch loaded in 0.001307s
Batch classified in 0.186297s

Batch 2 of 2
Batch loaded in 0.000721s
Batch classified in 0.029533s
...
Summary:
-------------------------------
Graph loaded in 0.018409s
All images loaded in 0.002028s
All images classified in 0.029533s
Average classification time: 0.029533s
Accuracy top 1: 0.5 (1 of 2)
Accuracy top 5: 1.0 (2 of 2)
--------------------------------
```

### Benchmark the accuracy
```bash
$ ck benchmark program:image-classification-onnx-py --cmd_key=preprocessed \
--repetitions=1  --env.CK_BATCH_SIZE=1 --env.CK_BATCH_COUNT=50000 \
--record --record_repo=local --record_uoa=mlperf-image-classification-mobilenet-onnx-py-accuracy \
--tags=mlperf,image-classification,mobilenet,onnx-py,accuracy \
--skip_print_timers --skip_stat_analysis --process_multi_keys
```
**NB:** For the `imagenet-2012-val-min` dataset, change `--env.CK_BATCH_COUNT=50000`
to `--env.CK_BATCH_COUNT=500` (or drop completely to test on a single image as if
with `--env.CK_BATCH_COUNT=1`).


<a name="accuracy"></a>
## Reference accuracy
**TODO**

<a name="further-info"></a>
## Further information

### Using Collective Knowledge
See the [common instructions](../README.md) for information on how to use Collective Knowledge
to learn about [the anatomy of a benchmark](../README.md#anatomy), or
to inspect and visualize [experimental results](../README.md#results).

### Using the client program

See [`ck-mlperf:program:image-classification-onnx-py`](https://github.com/ctuning/ck-mlperf/tree/master/program/image-classification-onnx-py) for more details about the client program.
