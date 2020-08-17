#! /usr/bin/env python3
import os
import sys
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
        "--accuracy_log", "-a",
        help="Specifies the path to the accuracy log from a submission/accuracy run.",
        default=""
    )
    parser.add_argument(
        "--performance_log", "-p",
        help="Specifies the path to the accuracy log from a performance run with accuracy log sampling enabled.",
        default=""
    )
    parser.add_argument(
        "--dtype", default="byte", choices=["byte", "float32", "int32", "int64"], help="data type of the label")
    args = parser.parse_args()

    print("Verifying accuracy. This might take a while...")
    acc_log  = args.accuracy_log
    perf_log = args.performance_log
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
    if num_perf_log_data_mismatch > 0 :
        print("TEST FAIL\n");
    else :
        print("TEST PASS\n");

if __name__ == '__main__':
	main()