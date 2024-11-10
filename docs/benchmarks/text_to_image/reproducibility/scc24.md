---
hide:
  - toc
---

# Text-to-Image with Stable Diffusion for Student Cluster Competition 2024

## Introduction

This guide is designed for the [Student Cluster Competition 2024](https://sc24.supercomputing.org/students/student-cluster-competition/) to walk participants through running and optimizing the [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549) using [Stable Diffusion XL 1.0](https://github.com/mlcommons/inference/tree/master/text_to_image#supported-models) across various software and hardware configurations. The goal is to maximize system throughput (measured in samples per second) without compromising accuracy. Since the model performs poorly on CPUs, it is essential to run it on GPUs.

For a valid MLPerf inference submission, two types of runs are required: a performance run and an accuracy run. In this competition, we focus on the `Offline` scenario, where throughput is the key metric—higher values are better. The official MLPerf inference benchmark for Stable Diffusion XL requires processing a minimum of 5,000 samples in both performance and accuracy modes using the COCO 2014 dataset. However, for SCC, we have reduced this and we also have two variants. `scc-base` variant has dataset size reduced to 50 samples, making it possible to complete both performance and accuracy runs in approximately 5-10 minutes. `scc-main` variant has dataset size of 500 and running it will fetch extra points as compared to running just the base variant. Setting up for Nvidia GPUs may take 2-3 hours but can be done offline. Your final output will be a tarball (`mlperf_submission.tar.gz`) containing MLPerf-compatible results, which you will submit to the SCC organizers for scoring.

## Scoring

In the SCC, your first objective will be to run `scc-base` variant for reference (unoptimized) Python implementation or a vendor-provided version (such as Nvidia's) of the MLPerf inference benchmark to secure a baseline score.

Once the initial run is successful, you'll have the opportunity to optimize the benchmark further by maximizing system utilization, applying quantization techniques, adjusting ML frameworks, experimenting with batch sizes, and more, all of which can earn you additional points.

Since vendor implementations of the MLPerf inference benchmark vary and are often limited to single-node benchmarking, teams will compete within their respective hardware categories (e.g., Nvidia GPUs, AMD GPUs). Points will be awarded based on the throughput achieved on your system.

Additionally, significant bonus points will be awarded if your team enhances an existing implementation, adds support for new hardware (such as an unsupported GPU), enables multi-node execution, or adds/extends scripts to [cm4mlops repository](https://github.com/mlcommons/cm4mlops/tree/main/script) supporting new devices, frameworks, implementations etc. All improvements must be made publicly available under the Apache 2.0 license and submitted alongside your results to the SCC committee to earn these bonus points, contributing to the MLPerf community.


!!! info
    Both MLPerf and CM automation are evolving projects.
    If you encounter issues or have questions, please submit them [here](https://github.com/mlcommons/cm4mlops/issues)

## Artifacts to submit to the SCC committee

You will need to submit the following files:

* `mlperf_submission.run` - CM commands to run MLPerf inference benchmark saved to this file.
* `mlperf_submission.md` - description of your platform and some highlights of the MLPerf benchmark execution.
* `<Team Name>` under which results are pushed to the github repository. 


## SCC interview

You are encouraged to highlight and explain the obtained MLPerf inference throughput on your system
and describe any improvements and extensions to this benchmark (such as adding new hardware backend
or supporting multi-node execution) useful for the community and [MLCommons](https://mlcommons.org).

## Run Commands

=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "sdxl", "reference", extra_variation_tags=",_short,_scc24-base", devices=["ROCm", "CUDA"],fixed_scenarios=["Offline"],categories=["Datacenter"], setup_tips=False, skip_test_query_count=True, extra_input_string="--precision=float16") }}

=== "Nvidia"
    ## Nvidia MLPerf Implementation
{{ mlperf_inference_implementation_readme (4, "sdxl", "nvidia", extra_variation_tags=",_short,_scc24-base", fixed_scenarios=["Offline"],categories=["Datacenter"], setup_tips=False, implementation_tips=False, skip_test_query_count=True) }}

!!! info
    Once the above run is successful, you can change `_scc24-base` to `_scc24-main` to run the main variant.

## Submission Commands

### Generate actual submission tree

```bash
cm run script --tags=generate,inference,submission \
   --clean \
   --run-checker \
   --tar=yes \
   --env.CM_TAR_OUTFILE=submission.tar.gz \
   --division=open \
   --category=datacenter \
   --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
   --run_style=test \
   --adr.submission-checker.tags=_short-run \
   --quiet \
   --submitter=<Team Name>
```

* Use `--hw_name="My system name"` to give a meaningful system name.


### Push Results to GitHub

Fork the `mlperf-inference-results-scc24` branch of the repository URL at [https://github.com/mlcommons/cm4mlperf-inference](https://github.com/mlcommons/cm4mlperf-inference). 

Run the following command after **replacing `--repo_url` with your GitHub fork URL**.

```bash
cm run script --tags=push,github,mlperf,inference,submission \
   --repo_url=https://github.com/<myfork>/cm4mlperf-inference \
   --repo_branch=mlperf-inference-results-scc24 \
   --commit_message="Results on system <HW Name>" \
   --quiet
```

Once uploaded give a Pull Request to the origin repository. Github action will be running there and once 
finished you can see your submitted results at [https://docs.mlcommons.org/cm4mlperf-inference](https://docs.mlcommons.org/cm4mlperf-inference).
