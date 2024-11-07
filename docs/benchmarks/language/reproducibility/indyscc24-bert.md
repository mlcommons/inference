---
hide:
  - toc
---

# Question and Answering using Bert Large for IndySCC 2024

## Introduction

This guide is designed for the [IndySCC 2024](https://sc24.supercomputing.org/students/indyscc/) to walk participants through running and optimizing the [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549) using [Bert Large](https://github.com/mlcommons/inference/tree/master/language/bert#supported-models) across various software and hardware configurations. The goal is to maximize system throughput (measured in samples per second) without compromising accuracy.

For a valid MLPerf inference submission, two types of runs are required: a performance run and an accuracy run. In this competition, we focus on the `Offline` scenario, where throughput is the key metric—higher values are better. The official MLPerf inference benchmark for Bert Large requires processing a minimum of 10833 samples in both performance and accuracy modes using the Squad v1.1 dataset.

## Scoring

In the IndySCC 2024, your objective will be to run a reference (unoptimized) Python implementation of the MLPerf inference benchmark to complete a successful submission passing the submission checker. Only one of the available framework needs to be submitted.


!!! info
    Both MLPerf and CM automation are evolving projects.
    If you encounter issues or have questions, please submit them [here](https://github.com/mlcommons/cm4mlops/issues)

## Artifacts to submit to the SCC committee
All the needed files are automatically pushed to the GitHub repository if you manage to complete the given commands. No additional files need to be submitted.


=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "bert-99", "reference", extra_variation_tags="", fixed_scenarios=["Offline"],categories=["Edge"], setup_tips=False) }}


## Submission Commands

### Generate actual submission tree

```bash
cm run script --tags=generate,inference,submission \
   --clean \
   --run-checker \
   --tar=yes \
   --env.CM_TAR_OUTFILE=submission.tar.gz \
   --division=open \
   --category=edge \
   --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
   --run_style=test \
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
