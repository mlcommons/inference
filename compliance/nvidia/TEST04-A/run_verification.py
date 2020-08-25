#! /usr/bin/env python3
# Copyright 2018 The MLPerf Authors. All Rights Reserved.
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
    # Parse arguments to identify the path to the logs from the performance runs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test4A_dir", "-a",
        help="Specifies the path to the directory containing the logs from the TEST04-A audit config run.",
        required=True
    )
    parser.add_argument(
        "--test4B_dir", "-b",
        help="Specifies the path to the directory containing the logs from the TEST04-B audit config test run.",
        required=True
    )
    parser.add_argument(
        "--output_dir", "-o",
        help="Specifies the path to the output directory where compliance logs will be uploaded to, i.e. inference_results_v0.7/closed/NVIDIA/compliance/T4x8/resnet/Offline.",
        required=True
    )
    parser.add_argument(
        "--dtype", default="byte", choices=["byte", "float32", "int32", "int64"], help="data type of the label (only needed in fastmode")

    args = parser.parse_args()

    print("Parsing arguments.")
    test4A_dir = args.test4A_dir
    test4B_dir = args.test4B_dir
    output_dir_A = os.path.join(args.output_dir, "TEST04-A")
    output_dir_B = os.path.join(args.output_dir, "TEST04-B")


    dtype = args.dtype

    # run verify performance
    verify_performance_binary = os.path.join(os.path.dirname(__file__),"verify_test4_performance.py")
    
    verify_performance_command = "python3 " + verify_performance_binary + " -u " + test4A_dir + "/mlperf_log_summary.txt" + " -s " + test4B_dir + "/mlperf_log_summary.txt | tee verify_performance.txt"
    try:
        os.system(verify_performance_command)
    except:
        print("Exception occurred trying to execute:\n  " + verify_performance_command)

    # check if verify performance script passes
    performance_pass_command = "grep PASS verify_performance.txt"
    performance_pass = "TEST PASS" in subprocess.check_output(performance_pass_command, shell=True).decode("utf-8")
    
    # setup output compliance directory structure
    output_performance_dir_A = os.path.join(output_dir_A, "performance", "run_1")
    output_performance_dir_B = os.path.join(output_dir_B, "performance", "run_1")

    try:
        if not os.path.isdir(output_performance_dir_A):
            os.makedirs(output_performance_dir_A)
    except:
        print("Exception occurred trying to create " + output_performance_dir_A)

    try:
        if not os.path.isdir(output_performance_dir_B):
            os.makedirs(output_performance_dir_B)
    except:
        print("Exception occurred trying to create " + output_performance_dir_B)

    # copy compliance logs to output compliance directory
    shutil.copy2("verify_performance.txt",output_dir_A)

    summary_file_A = os.path.join(test4A_dir,"mlperf_log_summary.txt")
    detail_file_A = os.path.join(test4A_dir,"mlperf_log_detail.txt")

    summary_file_B = os.path.join(test4B_dir,"mlperf_log_summary.txt")
    detail_file_B = os.path.join(test4B_dir,"mlperf_log_detail.txt")

    try:
        shutil.copy2(summary_file_A,output_performance_dir_A)
    except:
        print("Exception occured trying to copy " + summary_file_A + " to " + output_performance_dir_A)
    try:
        shutil.copy2(detail_file_A,output_performance_dir_A)
    except:
        print("Exception occured trying to copy " + detail_file_A + " to " + output_performance_dir_A)


    try:
        shutil.copy2(summary_file_B,output_performance_dir_B)
    except:
        print("Exception occured trying to copy " + summary_file_B + " to " + output_performance_dir_B)
    try:
        shutil.copy2(detail_file_B,output_performance_dir_B)
    except:
        print("Exception occured trying to copy " + detail_file_B + " to " + output_performance_dir_B)

    print("Performance check pass: {:}".format(performance_pass))
    print("TEST04 verification complete")

if __name__ == '__main__':
	main()
