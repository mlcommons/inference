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
import subprocess
import sys
import shutil
sys.path.append(os.getcwd())

import argparse
import json

import numpy as np

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
        "--reference_accuracy", "-r",
        help="Specifies the path to the accuracy log from a submission/accuracy run.",
        default=""
    )
    parser.add_argument(
        "--test_accuracy", "-t",
        help="Specifies the path to the accuracy log from a performance run with accuracy log sampling enabled.",
        default=""
    )
    parser.add_argument(
        "--dtype", default="byte", choices=["byte", "float32", "int32", "int64"], help="data type of the label")

    parser.add_argument(
        "--unixmode", action="store_true",
        help="Use unix commandline utilities instead of python JSON library (uses less memory but much slower.")

    parser.add_argument(
        "--fastmode", action="store_true",
        help="This flag has been deprecated. This script runs in fastmode by default. Use --unixmode to run in low memory consumption mode.")
    args = parser.parse_args()

    print("Verifying accuracy. This might take a while...")
    acc_log  = args.reference_accuracy
    perf_log = args.test_accuracy

    if not args.unixmode:
        with open(acc_log, "r") as acc_json:
            acc_data = json.load(acc_json)

        with open(perf_log, "r") as perf_json:
            perf_data = json.load(perf_json)

        # read accuracy log json and create a dictionary of qsl_idx/data pairs
        results_dict = {}
        num_acc_log_duplicate_keys = 0
        num_acc_log_data_mismatch = 0
        num_perf_log_qsl_idx_match = 0
        num_perf_log_data_mismatch = 0
        num_missing_qsl_idxs = 0

        print("Reading accuracy mode results...")
        for sample in acc_data:
            #print sample["qsl_idx"]
            qsl_idx = sample["qsl_idx"]
            data = sample["data"]
            if data == '':
                data = ""
            if qsl_idx in results_dict.keys():
                num_acc_log_duplicate_keys += 1
                if results_dict[qsl_idx] != data:
                    num_acc_log_data_mismatch += 1
            else:
                results_dict[qsl_idx] = data

        print("Reading performance mode results...")
        for sample in perf_data:
            qsl_idx = sample["qsl_idx"]
            data = np.frombuffer(bytes.fromhex(sample['data']), dtype_map[args.dtype]) if py3 == True \
                else np.frombuffer(bytearray.fromhex(sample['data']), dtype_map[args.dtype])

            if qsl_idx in results_dict.keys():
                num_perf_log_qsl_idx_match += 1
                data_perf = np.frombuffer(bytes.fromhex(results_dict[qsl_idx]), dtype_map[args.dtype]) \
                    if py3 == True else np.frombuffer(bytearray.fromhex(results_dict[qsl_idx]), dtype_map[args.dtype])
                if data_perf.size == 0 or data.size == 0:
                    if data_perf.size != data.size:
                        num_perf_log_data_mismatch += 1
                elif data[0] != data_perf[0]:
                    num_perf_log_data_mismatch += 1
            else:
                num_missing_qsl_idxs += 1

            results_dict[sample["qsl_idx"]] = sample["data"]


        print("num_acc_log_entries = {:}".format(len(acc_data)))
        print("num_acc_log_duplicate_keys = {:}".format(num_acc_log_duplicate_keys))
        print("num_acc_log_data_mismatch = {:}".format(num_acc_log_data_mismatch))
        print("num_perf_log_entries = {:}".format(len(perf_data)))
        print("num_perf_log_qsl_idx_match = {:}".format(num_perf_log_qsl_idx_match))
        print("num_perf_log_data_mismatch = {:}".format(num_perf_log_data_mismatch))
        print("num_missing_qsl_idxs = {:}".format(num_missing_qsl_idxs))
        if num_perf_log_data_mismatch == 0 and num_perf_log_qsl_idx_match > 0:
            print("TEST PASS\n")
        else:
            print("TEST FAIL\n")
        exit()

    py33 = sys.version_info >= (3,3)

    if not py33:
        print("Error: This script requires Python v3.3 or later")
        exit()


    get_perf_lines_cmd = "wc -l " + perf_log + "| awk '{print $1}'"
    num_perf_lines = int(subprocess.check_output(get_perf_lines_cmd, shell=True).decode("utf-8"))

    get_acc_lines_cmd = "wc -l " + acc_log + "| awk '{print $1}'"
    num_acc_lines = int(subprocess.check_output(get_acc_lines_cmd, shell=True).decode("utf-8"))

    num_acc_log_entries = num_acc_lines - 2
    num_perf_log_entries = num_perf_lines - 2
    #print(perf_qsl_idx)
    #print(get_perf_lines_cmd)
    #print(num_perf_lines)
    
    num_perf_log_data_mismatch = 0
    num_perf_log_data_match = 0
    print("Each dot represents 1% completion:")
    for perf_line in range(0, num_perf_lines):
        if perf_line % int(num_perf_lines/100) == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        # first and last line are brackets
        if perf_line == 0 or perf_line == int(num_perf_lines)-1:
            continue

        # calculate md5sum of line in perf mode accuracy_log
        perf_md5sum_cmd = "head -n " + str(perf_line + 1) + " " + perf_log + "| tail -n 1| sed -r 's/,//g' | sed -r 's/\"seq_id\" : \S+//g' | md5sum"
        #print(perf_md5sum_cmd)
        perf_md5sum = subprocess.check_output(perf_md5sum_cmd, shell=True).decode("utf-8")

        # get qsl idx
        get_qsl_idx_cmd = "head -n " + str(perf_line + 1) + " " + perf_log + "| tail -n 1| awk -F\": |,\" '{print $4}'"
        qsl_idx = subprocess.check_output(get_qsl_idx_cmd, shell=True).decode("utf-8").rstrip()

        # calculate md5sum of line in acc mode accuracy_log
        acc_md5sum_cmd = "grep \"qsl_idx\\\" : " + qsl_idx + ",\" " + acc_log + "| sed -r 's/,//g' | sed -r 's/\"seq_id\" : \S+//g' | md5sum"
        acc_md5sum = subprocess.check_output(acc_md5sum_cmd, shell=True).decode("utf-8")

        if perf_md5sum != acc_md5sum:
            num_perf_log_data_mismatch += 1
        else:
            num_perf_log_data_match += 1

    print("")
    print("num_acc_log_entries = {:}".format(num_acc_log_entries))
    print("num_perf_log_data_mismatch = {:}".format(num_perf_log_data_mismatch))
    print("num_perf_log_entries = {:}".format(num_perf_log_entries))
    if num_perf_log_data_mismatch == 0 and num_perf_log_data_match > 0:
        print("TEST PASS\n")
    else:
        print("TEST FAIL\n")

if __name__ == '__main__':
	main()
