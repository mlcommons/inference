
# Test 01 - Verify accuracy in performance mode
This repository provides the config files and scripts to run and verify TEST 01 - Verify accuracy in performance mode.

# Table of Contents
1. [Introduction](#introduction)
2. [Performance considerations](#Performance-considerations)
3. [Log size](#Log-size)
4. [Prerequisites](#Prerequisites)
5. [Non-determinism](#Non-determinism)
6. [Instructions](#Instructions)

## Introduction
The purpose of this test is to ensure that valid inferences are being performed in performance mode. By default, the inference result that is returned from SUT to Loadgen is not written to the accuracy JSON file and thus not checked for accuracy. In this test, the inference results of a subset of the total samples issued by loadgen are written to the accuracy JSON. In order to pass this test, two criteria must be satisfied:

 1. The inference results in the accuracy JSON file must match the inference results in the accuracy JSON generated in accuracy mode in the submission run.
 2. The performance while running this test must match the performance of the submission within 10%. 

## Performance considerations
The subset of samples results chosen to to be written to the accuracy JSON is determined randomly using a probability based on `accuracy_log_sampling_target` specified in the audit.config file divided by the total expected number of completed samples in the test run. This total expected number of completed samples is based on `min_duration_count`, `samples_per_query`, and `target_qps`. The goal is to ensure that a reasonable number of sample results gets written to the accuracy JSON regardless of the throughput of the system-under-test. Given that the number of actual completed samples may not match the expected number, the number of inference results written to the accuracy JSON may not exactly match `accuracy_log_sampling_target`.

There is an audit.config file for each individual benchmark, located in the benchmark subdirectories in this test directory. The `accuracy_log_sampling_target` value for each benchmark is chosen taking into consideration the performance sample count and size of the inference result. If performance with sampling enabled cannot meet the pass threshold set in verify_performance.py, `accuracy_log_sampling_target` may be reduced to check that performance approaches the submission score.


## Log size
In v0.7, the new workloads that have been added can generate significantly more output data than the workloads used in v0.5. Typically, the default mode of operation of the accuracy script is to check the accuracy JSON files using python JSON libraries. In the case that such scripts run out of memory, another fallback mode of operation can be enabled using UNIX-based commandline utilities which can be enabled using the `--unixmode` switch.

## Prerequisites
This script works best with Python 3.3 or later. For `--unixmode`,  the accuracy verification script also require the `wc`,`sed`,`awk`,`head`,`tail`,`grep`, and `md5sum` UNIX commandline utilities.
This script also assumes that the submission runs have already been run and that results comply with the submission directory structure as described in [https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference)
## Non-determinism
Under MLPerf inference rules, certain forms of non-determinism is acceptable, which can cause inference results to differ across runs. It is foreseeable that the results obtained during the accuracy run can be different from that obtained during the performance run, which will cause the accuracy checking script to report failure. Test failure will automatically result in an objection, but the objection can be overruled by providing proof of the quality of inference results. 
`create_accuracy_baseline.sh` is provided for this purpose. By running:

    bash ./create_accuracy_baseline.sh <path to mlperf_log_accuracy.json from the accuracy run> <path to mlperf_log_accuracy.json from the compliance test run>

 this script creates a baseline accuracy log called `mlperf_log_accuracy_baseline.json` using only a subset of the results from `mlperf_log_accuracy.json` from the accuracy run that corresponds to the QSL indices contained in `mlperf_log_accuracy.json` in the compliance test run. This provides an apples-to-apples accuracy log comparison between the accuracy run and compliance run.
The submitter can then run the reference accuracy script on `mlperf_log_accuracy_baseline.json` and the compliance test run's `mlperf_log_accuracy.json` and report the F1/mAP/DICE/WER/Top1%/AUC score. 

## Instructions

### Part I
Run test with the provided audit.config in the corresponding benchmark subdirectory. Note that audit.config must be copied to the directory where the benchmark is being run from. Verification that audit.config was properly read can be done by checking that loadgen has found audit.config in mlperf_log_detail.txt 

### Part II
Run the verification script:
  
    python3 run_verification.py -r RESULTS_DIR -c COMPLIANCE_DIR -o OUTPUT_DIR [--dtype {byte,float32,int32,int64}] [--unixmode]

  

 - RESULTS_DIR: Specifies the path to the corresponding results
   directory that contains the accuracy and performance subdirectories
   containing the submission logs, i.e.
   `inference_results_v0.7/closed/NVIDIA/results/GPU/resnet/Offline`. The script specifically requires mlperf_log_accuracy.json from the accuracy run and mlperf_log_summary.txt from the performance run.
  - COMPLIANCE_DIR: Specifies the path to the directory containing the logs from the compliance test run. 
   - OUTPUT_DIR: Specifies the path to the output directory where compliance logs will be uploaded from,   i.e. `inference_results_v0.7/closed/NVIDIA/compliance/GPU/resnet/Offline`

Expected outcome:

    Accuracy check pass: True                
    Performance check pass: True             
    TEST01 verification complete        

     
### Part III
**Note: This part is only necessary if the accuracy check in Part II fails.**

1. Create the baseline accuracy log for comparison to the compliance accuracy log, which will be named mlperf_log_accuracy_baseline.json:

 `bash ./create_accuracy_baseline.sh <path to mlperf_log_accuracy.json from the accuracy run> <path to mlperf_log_accuracy.json from the compliance test run>`

2. Run the reference accuracy script (i.e. the script that produces the F1/mAP/DICE/WER/Top1%/AUC score) on mlperf_log_accuracy_baseline.json, capture output and save to `<submitting_organization>/compliance/<system_desc_id>/<benchmark>/<scenario>/TEST01/accuracy/baseline_accuracy.txt` for upload.
3. Run accuracy script on mlperf_log_accuracy.json from the compliance run, capture output and save to `<submitting_organization>/compliance/<system_desc_id>/<benchmark>/<scenario>/TEST01/accuracy/compliance_accuracy.txt` for upload.

**For each target accuracy metric, the delta between the two accuracy results should be within (1-x)%. For example, for 99.9% accuracy metric, the delta of the accuracy results should be within 0.1%.**
