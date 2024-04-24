# Test 06 - Verify consistency of the output of the Llama-v2-70b
This repository provides the config files and scripts to run and verify TEST 06 - Verify consistency of the Llama-v2-70b output.

# Table of contents
1. [Introduction](#introduction)
2. [Requisites](#Requisites)
2. [Instructions](#instructions)

## Introduction

The purpose of this test is to ensure the consistency of the output of the Llama2 model and avoid a potential EOS exploit. This test will make a performance run, with a limit of 100 samples and logging them into `mlperf_log_accuracy.json`. To achieve a passing result in this test, three criteria must be met:
- In the case the first token is reported independently (not applicable for Offline scenario), it should match for every query with the first token of the model output.
- For each query, the model output should only end with zero or one EOS token
- The number of reported tokens should match with the length of it's

## Requisites

For this test, you need to be able to run the `Llama2-70b` benchmark. Therefore all it's requirements are also required for this test. Additionally, you need to have `numpy` installed.
```
pip install numpy
```

## Instructions
### Part I
Run the Llama-v2-70b benchmark with the provided audit.config in the corresponding subdirectory. Note that audit.config must be copied to the directory where the benchmark is being run from. Verification that audit.config was properly read can be done by checking that loadgen has found audit.config in mlperf_log_detail.txt

### Part II
Run the verification script
```
python3 run_verification.py -c COMPLIANCE_DIR -o OUTPUT_DIR -s SCENARIO
```
- COMPLIANCE_DIR: Specifies the path to the directory containing the logs from the compliance test run. 
- OUTPUT_DIR: Specifies the path to the output directory where compliance logs will be uploaded from,   i.e. `inference_results_v0.7/closed/NVIDIA/compliance/TEST06/llama2-70b/Offline`
- SCENARIO: Specifies the scenario the benchmark was run. One of ["Offline", "Server", "SingleStream", "MultiStream"]

Expected output
```
First token check pass: True                
EOS check pass: True             
Sample length check pass: True  
TEST06 verification complete   
```

Or:

```
First token check pass: Skipped                
EOS check pass: True             
Sample length check pass: True  
TEST06 verification complete     
```