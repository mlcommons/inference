# MLPerf Inference Reference Implementations

This is a repository of reference implementations for the MLPerf inference benchmark suite. 

Reference implementations are valid as only starting points for benchmark implementations. They are not fully optimized, and they are not intended to be used for "real" performance measurements of software frameworks or hardware platforms. The objective is for hardware and software vendors to take these reference implementations, optimize then, and submit them as optimized solutions to the MLPerf Inference call for submissions.

# Preliminary release (v0.5)

MLPerf inference release is very much an "alpha" release -- it could be improved in many ways. The benchmark suite is still being developed and refined. Please see the suggestions section below to learn how to contribute. We anticipate a significant round of updates after the v0.5 call for submission finishes. Much of the input would be taken into account for v0.6.

# Benchmarks

We provide reference implementations for each of the 5 benchmarks in the MLPerf inference suite:


| Area     | Task                 | Model             | Dataset            | Quality Target | Latency Constraint |
|----------|----------------------|-------------------|--------------------|----------------|--------------------|
| Vision   | Image classification | Resnet50-v1.5     | ImageNet (224x224) | TBD            | TBD                |
| Vision   | Image classification | MobileNets-v1 224 | ImageNet (224x224) | TBD            | TBD                |
| Vision   | Object detection     | SSD-ResNet34      | COCO (1200x1200)   | TBD            | TBD                |
| Vision   | Object detection     | SSD-MobileNets-v1 | COCO (300x300)     | TBD            | TBD                |
| Language | Machine translation  | GNMT              | WMT16              | TBD            | TBD                |


Each reference implementation provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container. The exception is GNMT.
* A script which downloads the appropriate dataset.
* A script which runs and times the model inference.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

You will need to install the _LoadGen_ and download and install the datasets and models.

## Load Generator

Please refer to README at under `/loadgen` directory to install LoadGen. Also see useful presentation material [here](https://docs.google.com/presentation/d/1QZmAYGbwZbNcrWUrxUNA4bNxuzCI27N8iV1AdJu4QIE/edit#slide=id.g5ae6850c4c_0_0).


## Models

### Vision

Please refer to README at https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection.

### Language

Please refer to README at https://github.com/mlperf/inference/tree/master/v0.5/translation/gnmt/tensorflow

### Test Setup

These benchmarks have been tested on the following machine configuration:

* Tested SW configurations:
  * Python 3
  * Cuda10.0, TF 1.14

* Tested HW configurations:
  * Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz with 12 threads (vCPUs) and one Titan XP
  * 64G DRAM
  * 200G Disk (recommmended)

So far, all models/scenarios except for SSD-resnet34 have been tested under the above hardware configuration.

# Suggestions

We are still in the early stages of developing MLPerf, and we are looking for areas to improve, partners, and contributors. If you have recommendations for new benchmarks, or otherwise would like to be involved in the process, please reach out to `info@mlperf.org`. For technical bugs or support, email `support@mlperf.org`.

# FAQ

Please search https://groups.google.com/forum/#!forum/mlperf-inference-submitters for frequently asked questions. There isn't any one link/resource in there as the questions are spread across the mailing list. Please search for the appropriate question you have in mind. If you cannot find a response, or are unclear please start a new thread by sending an email to mlperf-inference-submitters@googlegroups.com. 
