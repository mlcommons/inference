---
hide:
  - toc
---

# Question and Answering using Bert Large for IndySCC 2024

## Introduction

This guide is designed for the [IndySCC 2024](https://sc24.supercomputing.org/students/indyscc/) to walk participants through running and optimizing the [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549) using [Bert Large](https://github.com/mlcommons/inference/tree/master/language/bert#supported-models) across various software and hardware configurations. The goal is to maximize system throughput (measured in samples per second) without compromising accuracy.

For a valid MLPerf inference submission, two types of runs are required: a performance run and an accuracy run. In this competition, we focus on the `Offline` scenario, where throughput is the key metric—higher values are better. The official MLPerf inference benchmark for Bert Large requires processing a minimum of 10833 samples in both performance and accuracy modes using the Squad v1.1 dataset. Setting up for Nvidia GPUs may take 2-3 hours but can be done offline. Your final output will be a tarball (`mlperf_submission.tar.gz`) containing MLPerf-compatible results, which you will submit to the SCC organizers for scoring.

## Scoring

In the SCC, your first objective will be to run a reference (unoptimized) Python implementation or a vendor-provided version (such as Nvidia's) of the MLPerf inference benchmark to secure a baseline score.

Once the initial run is successful, you'll have the opportunity to optimize the benchmark further by maximizing system utilization, applying quantization techniques, adjusting ML frameworks, experimenting with batch sizes, and more, all of which can earn you additional points.

Since vendor implementations of the MLPerf inference benchmark vary and are often limited to single-node benchmarking, teams will compete within their respective hardware categories (e.g., Nvidia GPUs, AMD GPUs). Points will be awarded based on the throughput achieved on your system.


!!! info
    Both MLPerf and CM automation are evolving projects.
    If you encounter issues or have questions, please submit them [here](https://github.com/mlcommons/cm4mlops/issues)

## Artifacts to submit to the SCC committee

You will need to submit the following files:

* `mlperf_submission_short.tar.gz` - automatically generated file with validated MLPerf results.
* `mlperf_submission_short_summary.json` - automatically generated summary of MLPerf results.
* `mlperf_submission_short.run` - CM commands to run MLPerf BERT inference benchmark saved to this file.
* `mlperf_submission_short.tstamps` - execution timestamps before and after CM command saved to this file.
* `mlperf_submission_short.md` - description of your platform and some highlights of the MLPerf benchmark execution.



=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "bert-99", "reference", extra_variation_tags=",_short", scenarios=["Offline"],categories=["Edge"], setup_tips=False) }}

=== "Nvidia"
    ## Nvidia MLPerf Implementation
{{ mlperf_inference_implementation_readme (4, "bert-99", "nvidia", extra_variation_tags=",_short", scenarios=["Offline"],categories=["Edge"], setup_tips=False, implementation_tips=False) }}
    

