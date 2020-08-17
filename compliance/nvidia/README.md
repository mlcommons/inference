# Compliance Testing
This repository provides the compliance tests that need to be run in order to demonstrate a valid submission.

# Table of Contents
1. [Introduction](#introduction)
2. [Test Infrastructure](#Test-Infrastructure)
3. [Test Methodology](#Test-Methodology)

## Introduction
A handful of compliance tests have been created to help ensure that submissions comply with a subset of the MLPerf rules. Each compliance test must be run once for each submission run and the logs from the compliance test run must be uploaded along with the rest of submission collateral. Scripts are provided in each of the test subdirectories to help with copying the compliance test logs into the correct directory structure for upload. 

## Test Infrastructure
The compliance tests exercise functionality in LoadGen, enabled through the use of a config file that overrides LoadGen functionality, enabling it to run in a variety of compliance testing modes. Upon invocation, LoadGen checks if a `audit.config` file exists in the current working directory. The configuration parameters in `audit.config` override any settings set by `mlperf.conf` or `user.conf`.
## Test Methodology
Running a compliance test entails typically three steps:
#### 1. Setup
Copy the provided `audit.config` file from the test repository into the current working directory from where the benchmark typically starts execution.
#### 2. Execution
Run the benchmark as one normally would for a submission run. LoadGen will read `audit.config` and execute the compliance test.
Note: remove `audit.config` file from the working directory afterwards to prevent unintentionally running in compliance testing mode in future runs.
#### 3. Verification
Run the provided python-based verification script to ensure that the compliance test has successfully completed and meets expectations in terms of performance and/or accuracy. The script will also copy the output compliance logs to a path specified by the user in the correct  directory structure in preparation for upload to the MLPerf submission repository.



