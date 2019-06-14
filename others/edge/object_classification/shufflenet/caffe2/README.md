# Shufflenet MLPerf reference implementation

Owner: Fei Sun (feisun@fb.com)

Model architecture: shufflenet v1

Datasets used for evaluation: ImageNet 2012 validation dataset

Harness: [FAI-PEP](https://github.com/facebook/FAI-PEP)

Steps:
  * Download the imagenet validation dataset from [the official imagenet competition website](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar). Untar the file.
    * The download is too long so it is a separate step.
  * type: `run.sh <imagenet dir> [group1|group2|group3|group4|group8]`
    * Tested on a cleanly installed ubuntu 16.04 system.

## Model content
Five shufflenet v1 models are provided. They are different in the number of groups: one, two, three, four, or eight.

The models are saved under directory `shufflenet/model/group[x]`. The following files are present in the directory:

* model_dev.svg (svg file to show model structure)
* model_init_dev.svg (svg file to show the init model structure)
* model_init.pb (init model used by Caffe2 with all trained weights and biases)
* model_init.pbtxt (text file showing the weights and biases for all blobs)
* model.pb (model used by Caffe2)
* model.pbtxt (text file showing the model structure)
* model.pkl (model weights in pickle format)

## Model execution setup/pipeline

To run the shufflenet model on ImageNet validation dataset, we have created a script
to simplify the process. It works on the reference platform, taking charge of
installing all dependencies, building the framework, and invoking the harness
[FAI-PEP](https://github.com/facebook/FAI-PEP) to perform the benchmarking.

The future model submitters may need to understand the flow, and modify it for
new hardware platforms. However, the scripts and harness may hide some implementation
details on the benchmarking flow.

To make the flow obvious, the components of the benchmarking flow are described below:

* The overview of benchmarking on imagenet validation datatset can be found in the
[wiki](https://github.com/facebook/FAI-PEP/wiki/Run-Imagenet-validate-dataset)
* [build.sh](https://github.com/facebook/FAI-PEP/blob/master/specifications/frameworks/caffe2/host/incremental/build.sh)
builds the Pytorch/Caffe2 framework.
* [shufflenet_accuracy_imagenet.json](https://github.com/facebook/FAI-PEP/blob/master/specifications/models/caffe2/shufflenet/shufflenet_accuracy_imagenet.json)
 specifies the overall benchmarking flow.
* [imagenet_test_map.py](https://github.com/facebook/FAI-PEP/blob/master/libraries/python/imagenet_test_map.py)
transforms the imagenet validation dataset into a format that is consumable by the benchmarking process.
* [convert_image_to_tensor.cc](https://github.com/pytorch/pytorch/blob/master/binaries/convert_image_to_tensor.cc)
performs preprocessing of the images (scale, crop, normalize, etc.) and converts to the blob the framework can consume directly.
* [caffe2_benchmark.cc](https://github.com/pytorch/pytorch/blob/master/binaries/caffe2_benchmark.cc)
runs the model in various settings and output the result.
* [classification_compare.py](https://github.com/facebook/FAI-PEP/blob/master/libraries/python/classification_compare.py)
compares the model output with the golden values.
* Since the validation dataset is large and cannot be processed in one iteration, the images are processed in batches.
[aggregate_classification_results.py](https://github.com/facebook/FAI-PEP/blob/master/libraries/python/aggregate_classification_results.py) aggregates all the run results and forms the final result.

## Model accuracy

* group1: 66.468%
* group2: 67.376%
* group3: 67.506%
* group4: 67.455%
* group8: 67.710%

## Acknowledgement

Thanks a lot for the help from Carole-Jean Wu, Peizhao Zhang, and Xiaoliang Dai.
