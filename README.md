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

These benchmarks have been tested on the following machine configuration:

~~* 16 CPUs, one Nvidia P100.
* Ubuntu 16.04, including docker with Nvidia support.
~~* 600GB of disk (though many benchmarks do require less disk).
* Either CPython 2 or CPython 3, depending on benchmark (see Dockerfiles for details).

Generally, a benchmark can be run with the following steps:

1. Setup docker & dependencies. There is a shared script (install_cuda_docker.sh) to do this. Some benchmarks will have additional setup, mentioned in their READMEs.
2. Download the dataset using `./download_dataset.sh`. This should be run outside of docker, on your host machine. This should be run from the directory it is in (it may make assumptions about CWD).
3. Optionally, run `verify_dataset.sh` to ensure the was successfully downloaded.
4. Build and run the docker image; the command to do this is included with each Benchmark. 

Each benchmark will run until the target quality is reached and then stop, printing timing results. 

Some these benchmarks are rather slow or take a long time to run on the reference hardware (i.e., 16 CPUs and one P100). We expect to see significant performance improvements with more hardware and optimized implementations. 

# Suggestions

We are still in the early stages of developing MLPerf, and we are looking for areas to improve, partners, and contributors. If you have recommendations for new benchmarks, or otherwise would like to be involved in the process, please reach out to `info@mlperf.org`. For technical bugs or support, email `support@mlperf.org`.

# FAQ

Please search https://groups.google.com/forum/#!forum/mlperf-inference-submitters for frequently asked questions. There isn't any one link/resource in there as the questions are spread across the mailing list. Please search for the appropriate question you have in mind. If you cannot find a response, or are unclear please start a new thread by sending an email to mlperf-inference-submitters@googlegroups.com. 
