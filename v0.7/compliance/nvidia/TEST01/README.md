
# Test 01 - Verify accuracy in performance mode
## Introduction
The purpose of this test is to ensure that valid inferences are being performed in performance mode. By default, the inference result that is returned from SUT to Loadgen is not written to the accuracy JSON file and thus not checked for accuracy. In this test, the inference results of a subset of the total samples issued by loadgen are written to the accuracy JSON. In order to pass this test, two criteria must be satisfied:

 1. The inference results in the accuracy JSON file must match the inference results in the accuracy JSON generated in accuracy mode in the submission run.
 2. The performance while running this test must match the performance of the submission within 10%. 

## Performance considerations
The subset of samples results chosen to to be written to the accuracy JSON is determined randomly using a probability based on `accuracy_log_sampling_target` specified in the audit.config file divided by the total expected number of completed samples in the test run. This total expected number of completed samples is based on `min_duration_count`, `samples_per_query`, and `target_qps`. The goal is to ensure that a reasonable number of sample results gets written to the accuracy JSON regardless of the throughput of the system-under-test. Given that the number of actual completed samples may not match the expected number, the number of inference results written to the accuracy JSON may not exactly match `accuracy_log_sampling_target`.

There is an audit.config file for each individual benchmark, located in the benchmark subdirectories in this test directory. The `accuracy_log_sampling_target` value for each benchmark is chosen taking into consideration the performance sample count and size of the inference result. If performance with sampling enabled cannot meet the pass threshold set in verify_performance.py, `accuracy_log_sampling_target` may be reduced to check that performance approaches the submission score.

## Log size
3d-unet is unique in that its inference result output per-sample is drastically larger than that of other benchmarks. For all other benchmarks, the accuracy JSON results can be checked using python JSON libraries, which can be enabled by providing `--fastmode` to the run_verification.py script. For 3d-unet, using fastmode will result in verify_performance.py running out of memory, so the alternative way of using UNIX-based commandline utilities must be used by not supplying the `--fastmode` switch.

## Prerequisites
This script works best with Python 3.3 or later, although `--fastmode` will work with earlier versions. For 3d-unet, the accuracy verification script require the `wc`,`sed`,`awk`,`head`,`tail`,`grep`, and `md5sum` UNIX commandline utilities.

## Non-determinism
Note that under MLPerf inference rules, certain forms of non-determinism is acceptable, which can cause inference results to differ across runs. It is foreseeable that the results obtained during the accuracy run can be different from that obtained during the performance run, which will cause the accuracy checking script to report failure. Test failure will automatically result in an objection, but the objection can be overturned by comparing the quality of the results generated in performance mode to that obtained in accuracy mode. This can be done by using the accuracy measurement scripts provided as part of the repo to ensure that the accuracy score meets the target. An example is provided for GNMT in the gnmt folder.

## Instructions

### Part I
Run test with the provided audit.config in the corresponding benchmark subdirectory. Note that audit.config must be copied to the directory where the benchmark is being run from. Verification that audit.config was properly read can be done by checking that loadgen has found audit.config in mlperf_log_detail.txt 

### Part II
Run the verification script:
  `python3 run_verification.py -r RESULTS_DIR -c COMPLIANCE_DIR -o OUTPUT_DIR [--dtype {byte,float32,int32,int64}] [--fastmode]`
  
RESULTS_DIR: Specifies the path to the corresponding results directory that contains the accuracy and performance subdirectories containing the submission logs, i.e. `inference_results_v0.7/closed/NVIDIA/results/GPU/resnet/Offline`
COMPLIANCE_DIR: Specifies the path to the directory containing the logs from the compliance test run.
OUTPUT_DIR: Specifies the path to the output directory where compliance logs will be uploaded from, i.e. `inference_results_v0.7/closed/NVIDIA/compliance/GPU/resnet/Offline`

Expected outcome:

    Accuracy check pass: True                
    Performance check pass: True             
    TEST01 verification complete        

     


