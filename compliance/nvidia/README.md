# Compliance Testing
This repository provides the compliance tests that need to be run by the submitter in order to demonstrate a valid submission.

# Table of Contents
1. [Introduction](#introduction)
2. [Test Infrastructure](#Test-Infrastructure)
3. [Test Methodology](#Test-Methodology)
4. [Tests Required for each Benchmark](#tests-required-for-each-benchmark)

## Introduction
The purpose of compliance testing is to ensure a basic level of compliance with a subset of the MLPerf rules. The tests are designed to be complementary to third-party auditing which will be introduced in future rounds of MLPerf. The tests are not meant to root-cause issues with the submission, but can help detect anomalies in the submission that need to be investigated further by the submitter.

Each compliance test must be run once for each submission run and the logs from the compliance test run must be uploaded along with the rest of the submission collateral. In MLPerf Inference v0.7, effort has been made to reduce the burden on submitters to perform compliance testing through improvements to documentation, scripting, and LoadGen's compliance functionality. More documentation is provided on the purpose of each test in the corresponding test directory, along with more detailed instructions.

## Test Infrastructure
The compliance tests exercise functionality in LoadGen, triggered through the use of a config file that overrides LoadGen functionality. This enables LoadGen to run in a variety of compliance testing modes. When LoadGen::StartTest() is invoked, LoadGen checks if a `audit.config` file exists in the current working directory. If the file is found, LoadGen will log this event in `mlperf_log_detail.txt`.  The LoadGen settings that are used will be logged in `mlperf_log_summary.txt`. The configuration parameters in `audit.config` override any settings set by `mlperf.conf` or `user.conf`.
## Test Methodology
Running a compliance test entails typically three steps:
#### 1. Setup
Copy the provided `audit.config` file from the test repository into the current working directory from where the benchmark typically starts execution.
#### 2. Execution
Run the benchmark as one normally would for a submission run. LoadGen will read `audit.config` and execute the compliance test.
Note: remove `audit.config` file from the working directory afterwards to prevent unintentionally running in compliance testing mode in future runs.
#### 3. Verification
Run the provided python-based verification script to ensure that the compliance test has successfully completed and meets expectations in terms of performance and/or accuracy. The script will also copy the output compliance logs to a path specified by the user in the correct  directory structure in preparation for upload to the MLPerf submission repository.
## Test Submission
The `run_verification.py` found in each test directory will copy the test files to be submitted to the directory specified. These files will follow the directory structure specified in [https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference](https://github.com/mlperf/policies/blob/master/submission_rules.adoc#562-inference) under `audit/` and are must be uploaded in order for the submission to be considered valid.

## Tests Required for each Benchmark

| model | Required Compliance Tests
| ---- | ---- |
| resnet50-v1.5 | [TEST01](./TEST01/), [TEST04](./TEST04/), [TEST05](./TEST05/) |
| retinanet 800x800 | [TEST01](./TEST01/), [TEST05](./TEST05/) |
| bert | [TEST01](./TEST01/), [TEST05](./TEST05/) |
| dlrm-v2 | [TEST01](./TEST01/), [TEST05](./TEST05/) |
| 3d-unet | [TEST01](./TEST01/), [TEST05](./TEST05/) |
| rnnt | [TEST01](./TEST01/), [TEST05](./TEST05/) |
| gpt-j | - |
| stable-diffusion-xl | [TEST01](./TEST01/), [TEST04](./TEST04/) |
| Llama2-70b | [TEST06](./TEST06/) |
| mixtral-8x7b | [TEST06](./TEST06/) |
