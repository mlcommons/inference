#! /usr/bin/env python3
# Copyright 2018-2022 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import sys
import shutil
import subprocess
import argparse
import json

import numpy as np

sys.path.append(os.getcwd())

dtype_map = {
    "byte": np.byte,
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64
}

def main():


    py3 = sys.version_info >= (3,0)
    # Parse arguments to identify the path to the accuracy logs from
    #   the accuracy and performance runs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", "-r",
        help="Specifies the path to the corresponding results directory that contains the accuracy and performance subdirectories containing the submission logs, i.e. inference_results_v0.7/closed/NVIDIA/results/T4x8/resnet/Offline.",
        required=True
    )
    parser.add_argument(
        "--compliance_dir", "-c",
        help="Specifies the path to the directory containing the logs from the compliance test run.",
        required=True
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Specifies the path to the output directory where compliance logs will be uploaded from, i.e. inference_results_v0.7/closed/NVIDIA/compliance/T4x8/resnet/Offline.",
        required=True
    )
    parser.add_argument(
        "--dtype", default="byte", choices=["byte", "float32", "int32", "int64"], help="data type of the label (not needed in unixmode")
    parser.add_argument(
        "--unixmode", action="store_true",
        help="Use UNIX commandline utilities to verify accuracy (uses less memory but much slower.")

    args = parser.parse_args()

    print("Parsing arguments.")
    results_dir = args.results_dir
    compliance_dir = args.compliance_dir
    output_dir = os.path.join(args.output_dir, "TEST01")
    unixmode = ""
    if args.unixmode:
        unixmode = " --unixmode"
        for binary in ["wc", "md5sum", "grep", "awk", "sed", "head", "tail"]:
            missing_binary = False
            if shutil.which(binary) == None:
                print("Error: This script requires the {:} commandline utility".format(binary))
                missing_binary = True
        if missing_binary:
            exit()

    dtype = args.dtype

    verify_accuracy_binary = os.path.join(os.path.dirname(__file__),"verify_accuracy.py")
    # run verify accuracy
    verify_accuracy_command = "python3 " + verify_accuracy_binary + " --dtype " + args.dtype + unixmode + " -r " + results_dir + "/accuracy/mlperf_log_accuracy.json" + " -t " + compliance_dir + "/mlperf_log_accuracy.json | tee verify_accuracy.txt"
    try:
        os.system(verify_accuracy_command)
    except Exception:
        print("Exception occurred trying to execute:\n  " + verify_accuracy_command)
    # check if verify accuracy script passes

    accuracy_pass_command = "grep PASS verify_accuracy.txt"
    try:
        accuracy_pass = "TEST PASS" in subprocess.check_output(accuracy_pass_command, shell=True).decode("utf-8")
    except Exception:
        accuracy_pass = False

    # run verify performance
    verify_performance_binary = os.path.join(os.path.dirname(__file__),"verify_performance.py")
    verify_performance_command = "python3 " + verify_performance_binary + " -r " + results_dir + "/performance/run_1/mlperf_log_summary.txt" + " -t " + compliance_dir + "/mlperf_log_summary.txt | tee verify_performance.txt"
    try:
        os.system(verify_performance_command)
    except Exception:
        print("Exception occurred trying to execute:\n  " + verify_performance_command)

    # check if verify performance script passes
    performance_pass_command = "grep PASS verify_performance.txt"
    try:
        performance_pass = "TEST PASS" in subprocess.check_output(performance_pass_command, shell=True).decode("utf-8")
    except Exception:
        performance_pass = False
    
    # setup output compliance directory structure
    output_accuracy_dir = os.path.join(output_dir, "accuracy")
    output_performance_dir = os.path.join(output_dir, "performance", "run_1")
    try:
        if not os.path.isdir(output_accuracy_dir):
            os.makedirs(output_accuracy_dir)
    except Exception:
        print("Exception occurred trying to create " + output_accuracy_dir)
    try:
        if not os.path.isdir(output_performance_dir):
            os.makedirs(output_performance_dir)
    except Exception:
        print("Exception occurred trying to create " + output_performance_dir)

    # copy compliance logs to output compliance directory
    shutil.copy2("verify_accuracy.txt",output_dir)
    shutil.copy2("verify_performance.txt",output_dir)
    accuracy_file = os.path.join(compliance_dir,"mlperf_log_accuracy.json")
    summary_file = os.path.join(compliance_dir,"mlperf_log_summary.txt")
    detail_file = os.path.join(compliance_dir,"mlperf_log_detail.txt")

    try:
        shutil.copy2(accuracy_file,output_accuracy_dir)
    except Exception:
        print("Exception occured trying to copy " + accuracy_file + " to " + output_accuracy_dir)
    try:
        shutil.copy2(summary_file,output_performance_dir)
    except Exception:
        print("Exception occured trying to copy " + summary_file + " to " + output_performance_dir)
    try:
        shutil.copy2(detail_file,output_performance_dir)
    except Exception:
        print("Exception occured trying to copy " + detail_file + " to " + output_performance_dir)

    print("Accuracy check pass: {:}".format(accuracy_pass))
    print("Performance check pass: {:}".format(performance_pass))
    print("TEST01 verification complete")

if __name__ == '__main__':
	main()
