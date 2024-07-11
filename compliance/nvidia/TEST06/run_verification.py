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
import argparse
import json

import numpy as np

DTYPE_MAP = {
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "float32": np.float32
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compliance_dir", "-c", 
                        help="Specifies the path to the directory containing the logs from the compliance test run.",
                        required=True)
    parser.add_argument("--output_dir", "-o",
                        help="Specifies the path to the output directory where compliance logs will be uploaded from, i.e. inference_results_v0.7/closed/NVIDIA/compliance/T4x8/resnet/Offline.",
                        required=True)
    parser.add_argument("--eos_token_id", '-e', default=2, help="EOS token id of the tokenizer")
    parser.add_argument("--dtype", "-d", default="int64", choices=["int64", "int32", "int16", "float32"])
    parser.add_argument("--scenario", "-s", required=True, choices=["Offline", "Server", "SingleStream", "MultiStream"])
    args = parser.parse_args()
    return args

def eos_check(acc_data, dtype, eos_token_id=2):
    for sample in acc_data:
        data = np.frombuffer(bytes.fromhex(sample["data"]), dtype=dtype)
        i = data.shape[0] - 1
        n_eos_tokens = 0
        while (i > 0):
            if data[i] == eos_token_id:
                n_eos_tokens += 1
            if n_eos_tokens >= 2:
                # Allow output to be [eos_token_id, eos_token_id]
                return len(data) == 2
            if data[i] != eos_token_id:
                break
            i-=1
    return True

def first_token_check(acc_data, dtype):
    for sample in acc_data:
        data = np.frombuffer(bytes.fromhex(sample["data"]), dtype=dtype)
        token_data = np.frombuffer(bytes.fromhex(sample["token_data"]), dtype=dtype)
        for t1, t2 in zip(data, token_data):
            if t1 != t2:
                return False
        
    return True

def sample_len_check(acc_data, dtype):
    for sample in acc_data:
        data = np.frombuffer(bytes.fromhex(sample["data"]), dtype=dtype)
        token_count = int(sample["token_count"])
        if len(data) != token_count:
            return False
    return True


def main():
    args = get_args()
    accuracy_file = os.path.join(args.compliance_dir, "mlperf_log_accuracy.json")
    
    with open(accuracy_file, "r") as acc_json:
        acc_data = json.load(acc_json)
    
    try:
        eos_pass = eos_check(acc_data, DTYPE_MAP[args.dtype], args.eos_token_id)
    except Exception:
        print("Unexpected error occured while doing the EOS check")
        eos_pass = False

    need_first_token_check = (args.scenario != "Offline")
    first_token_pass = True
    if need_first_token_check:
        try:
            first_token_pass = first_token_check(acc_data, DTYPE_MAP[args.dtype])
        except Exception:
            print("Unexpected error occured while doing the first token check")
            first_token_pass = False

    sample_len_pass = sample_len_check(acc_data, DTYPE_MAP[args.dtype])

    # Construct output based on the results of checks
    output = ""
    # Add first token check
    if need_first_token_check:
        output += f"First token check pass: {first_token_pass}\n"
    else:
        output += f"First token check pass: Skipped\n"
    
    # Add EOS check
    output += f"EOS check pass: {eos_pass}\n"

    # Add sample length check
    output += f"Sample length check pass: {sample_len_pass}\n"

    if eos_pass and first_token_pass and sample_len_pass:
        output += "TEST06 verification complete\n"
    else:
        output += "TEST06 verification failed\n"

    # Output test output to console and folder
    output_dir = os.path.join(args.output_dir, "TEST06")
    output_accuracy_dir = os.path.join(output_dir, "accuracy")
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_accuracy_dir):
        os.makedirs(output_accuracy_dir)

    with open(os.path.join(output_dir, "verify_accuracy.txt"), "w") as f:
        f.write(output)

    try:
        shutil.copy2(accuracy_file,output_accuracy_dir)
    except Exception:
        print("Exception occured trying to copy " + accuracy_file + " to " + output_accuracy_dir)
    print(output)

if __name__ == "__main__":
    main()
