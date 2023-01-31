
# Test 05 - Vary RNG seeds
## Introduction
The purpose of this test is to ensure that the SUT does not favor a particular set of Loadgen RNG seed values. The pass condition is that performance with non-default RNG seed values should be similar to the submitted performance.

The seeds that are changed are listed below:
 - qsl_rng_seed - determines order of samples in QSL
 - sample_index_rng_seed - determines subset of samples in each loadable set
  - schedule_rng_seed - determines scheduling of samples in server mode

## Prerequisites
This script works best with Python 3.3 or later.
This script also assumes that the submission runs have already been run and that results comply with the submission directory structure as described in [https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference)
## Pass Criteria
Performance must be within 5% of the submission performance. In single stream mode, latencies can be very short for high performance systems and run-to-run variation due to external disturbances (OS) can be significant. In such cases and when submission latencies are less or equal to 0.2ms, the pass threshold is relaxed to 20%.

## Instructions

### Part I
Run the benchmark with the provided audit.config in the corresponding benchmark subdirectory. 

The audit.config file must be copied to the directory where the benchmark is being run from. Verification that audit.config was properly read can be done by checking that loadgen has found audit.config in mlperf_log_detail.txt 

Alternatively, you can alter the mlperf.conf (or the configuration file's copy your benchmark is using) by setting the value `*.*.test05 = 1`. For this option make sure you have the right values for `*.*.test05_qsl_rng_seed`, `*.*.test05_sample_index_rng_seed` and `*.*.test05_schedule_rng_seed` included in your configuration file.

### Part II
Run the verification script:
  `python3 run_verification.py -r RESULTS_DIR -c COMPLIANCE_DIR -o OUTPUT_DIR [--dtype {byte,float32,int32,int64}]`
  
- RESULTS_DIR: Specifies the path to the corresponding results directory that contains the accuracy and performance subdirectories containing the submission logs, i.e. `inference_results_v0.7/closed/NVIDIA/results/GPU/resnet/Offline`
- COMPLIANCE_DIR: Specifies the path to the directory containing the logs from the compliance test run.
- OUTPUT_DIR: Specifies the path to the output directory where compliance logs will be uploaded from, i.e. `inference_results_v0.7/closed/NVIDIA/compliance/GPU/resnet/Offline`

Expected outcome:
              
    Performance check pass: True             
    TEST05 verification complete        

     


