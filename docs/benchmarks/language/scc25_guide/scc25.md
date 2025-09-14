---
hide:
  - toc
---

# Text Summarization with Llama2-70b for Student Cluster Competition 2025

## Introduction

This guide is designed for the [Student Cluster Competition 2025](https://sc25.supercomputing.org/students/student-cluster-competition/) to walk participants through running and optimizing the [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549) using [Llama2 70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) across various software and hardware configurations. The goal is to maximize system throughput (measured in Tokens per second) without compromising accuracy. Since the model performs poorly on CPUs, it is essential to run it on GPUs.

For a valid MLPerf Inference submission in this competition, you must run both a performance test and an accuracy test—**no compliance runs are required**. We use the **Offline** scenario, where throughput is the key metric (higher is better). For Llama 2-70B with the OpenOrca dataset (24,576 samples), the **performance run** must process an integer multiple of the full dataset (24,576 × *N* samples), while the **accuracy run** must process **exactly** the full dataset (24,576 samples). These requirements are taken care of by the MLPerf inference implementations. Setup for NVIDIA GPUs typically takes 2–3 hours and can be done offline. The final output is a tarball (`mlperf_submission.tar.gz`) containing MLPerf-compatible results which can be submitted to the organizers via a CLI command.

## Scoring

In the SCC, your first objective will be to get a valid MLPerf benchmark run. Traditionally running the reference MLPerf inference implementation (in Python) is easier compared to running Nvidia MLPerf inference implementation. Since for SCC25 we are having the Llama2-70b model, running the reference implementation needs around 600GB of VRAM and is tested only on 8xH100 Nvidia GPUs. If you have lower VRAM, trying the vendor implementation like of Nvidia or AMD is the best option.  

MLCommons provides [automation](https://github.com/mlcommons/mlperf-automations/) to run the MLPerf inference benchmarks which you can make use of. Currently the automation supports the reference implementation as well as Nvidia implementation and this is useful for you to get a quick valid result as the automation produces the required final output. You can also use the manual steps by following the [reference](https://github.com/mlcommons/inference/tree/master/language/llama2-70b), [Nvidia](https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/NVIDIA) or [AMD](https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/AMD) implementation readmes.

Once the initial run is successful, you'll have the opportunity to optimize the benchmark further by maximizing system utilization, applying quantization techniques, adjusting ML frameworks, experimenting with batch sizes, and more, all of which can earn you additional points.

Since vendor implementations of the MLPerf inference benchmark vary, teams will compete within their respective hardware categories (e.g., Nvidia GPUs, AMD GPUs). Points will be awarded based on the throughput achieved on your system.

Additionally, significant bonus points will be awarded if your team enhances an existing implementation, enables multi-node execution, or adds/extends scripts to [mlperf-automations repository](https://github.com/mlcommons/mlperf-automations/tree/dev/script) supporting new devices, frameworks, implementations etc. All improvements must be made publicly available under the Apache 2.0 license and submitted as pull requests by November 10, 2025 and only the code which is *merge ready* will be considered for evaluation. As a guideline, below are some examples which can fetch you bonus points. 

* Adds multi-node execution support for Nvidia, AMD or reference implementations
* Support automation for AMD implementation
* Supports fp8/fp4 quantization for Reference implementation
* Automate the [network reference implementation](https://github.com/mlcommons/inference/blob/master/language/llama2-70b/SUT_API.py) (this uses OpenAI compatible endpoints)
* The MLPerf automation supports docker run of Nvidia implementation. Supporting apptainer is a valuable contribution

PS: For any query regarding the contribution, feel free to raise an issue in the [Inference](https://github.com/mlcommons/inference) or [MLPerf automations](https://github.com/mlcommons/mlperf-automations) repositories.

!!! info
    Both MLPerf and MLC automation are evolving projects.
    If you encounter issues related to SCC, please submit them [here](https://github.com/mlcommons/inference/issues) with **scc-25** label
    with proper information about the command used, error logs and any additional usefull information to debug the issue.

## Artifacts to submit to the SCC committee

You will need to submit the following files:

* `mlperf_submission.run` - MLC commands to run MLPerf inference benchmark saved to this file.
* `mlperf_submission.md` - description of your platform and some highlights of the MLPerf benchmark execution.
* `<Team Name>` under which results are pushed to the github repository. 


## SCC interview

You are encouraged to highlight and explain the obtained MLPerf inference throughput on your system
and describe any improvements and extensions to this benchmark (such as adding new hardware backend
or supporting multi-node execution) useful for the community and [MLCommons](https://mlcommons.org).

## Run Commands

=== "MLCommons-Python"
    ## MLPerf Reference Implementation in Python
    
{{ mlperf_inference_implementation_readme (4, "llama2-70b-99", "reference", fixed_scenarios=["Offline"], categories=["Datacenter"], setup_tips=False, implementation_tips=False, skip_test_query_count=True) }}

{{ mlperf_inference_implementation_readme (4, "llama2-70b-99.9", "reference", fixed_scenarios=["Offline"], categories=["Datacenter"], setup_tips=False, implementation_tips=False, skip_test_query_count=True) }}

=== "Nvidia"
    ## Nvidia MLPerf Implementation

{{ mlperf_inference_implementation_readme (4, "llama2-70b-99", "nvidia", fixed_scenarios=["Offline"], categories=["Datacenter"], setup_tips=False, implementation_tips=False, skip_test_query_count=True) }}

{{ mlperf_inference_implementation_readme (4, "llama2-70b-99.9", "nvidia", fixed_scenarios=["Offline"], categories=["Datacenter"], setup_tips=False, implementation_tips=False, skip_test_query_count=True) }}

## Submission Commands

### Generate actual submission tree


```bash
mlcr generate,inference,submission,_wg-inference \
   --clean \
   --run-checker \
   --tar=yes \
   --env.MLC_TAR_OUTFILE=submission.tar.gz \
   --division=open \
   --category=datacenter \
   --env.CM_DETERMINE_MEMORY_CONFIGURATION=yes \
   --quiet \
   --submitter=<Team Name>
```

* Use `--hw_name="My system name"` to give a meaningful system name.
* At the end, a **.tar** file would be generated inside the current working directory.

### Submit Results

> **Note:**
Further instructions on the final submission will be published as the deadline approaches.

<!-- Fork the `mlperf-inference-results-scc25` branch of the repository URL at [mlperf-automations](https://github.com/mlcommons/mlperf-automations). 

Run the following command after **replacing `--repo_url` with your GitHub fork URL**.

```bash
mlcr push,github,mlperf,inference,submission \
   --repo_url=https://github.com/<myfork>/mlperf-automations \
   --repo_branch=mlperf-inference-results-scc25 \
   --commit_message="Results on system <HW Name>" \
   --quiet
```

Once uploaded give a Pull Request to the origin repository. Github action will be running there and once 
finished you can see your submitted results at [https://docs.mlcommons.org/mlperf-automations](https://docs.mlcommons.org/mlperf-automations). -->
