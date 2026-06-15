#! /usr/bin/env python3
# Copyright 2018-2025 The MLPerf Authors. All Rights Reserved.
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

import json
import argparse
import os
import sys
import re

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "tools",
            "submission")))
from log_parser import MLPerfLog  # noqa

RESULT_FIELD = {
    "Offline": "result_samples_per_second",
    "SingleStream": "early_stopping_latency_ss",
    "MultiStream": "early_stopping_latency_ms",
    "Server": "result_completed_samples_per_sec",
}


def parse_result_log(file_path):
    score, target_latency = 0, None

    mlperf_log = MLPerfLog(file_path)
    scenario = mlperf_log["effective_scenario"]
    score = float(mlperf_log[RESULT_FIELD[scenario]])

    if not (
        "result_validity" in mlperf_log.get_keys()
        and mlperf_log["result_validity"] == "VALID"
    ):
        sys.exit("TEST FAIL: Invalid results in {}".format(file_path))

    if mlperf_log.has_error():
        print(
            "WARNING: {} ERROR reported in {}".format(
                line.split()[0],
                file_path))

    res = float(mlperf_log[RESULT_FIELD[scenario]])
    if scenario == "Server":
        target_latency = mlperf_log["effective_target_latency_ns"]

    return scenario, score, target_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reference_log_details",
        help="Path to reference performance log_details file.",
        required=True)
    parser.add_argument(
        "-t",
        "--test_log_details",
        help="Path to test performance log_details file.",
        required=True)
    args = parser.parse_args()

    print("Verifying performance.")
    ref_scenario, ref_score, ref_target_latency = parse_result_log(
        args.reference_log_details)
    test_scenario, test_score, test_target_latency = parse_result_log(
        args.test_log_details)

    if test_scenario != ref_scenario:
        sys.exit("TEST FAIL: Test and reference scenarios do not match!")

    if ref_scenario == "Server" and test_target_latency != ref_target_latency:
        sys.exit("TEST FAIL: Server target latency mismatch")

    print(f"Reference score = {ref_score}")
    print(f"Test score = {test_score}")

    threshold = 0.10
    if (ref_scenario == "SingleStream" and ref_score <= 200000) or (
            ref_scenario == "MultiStream" and ref_score <= 1600000):
        threshold = 0.20

    if ref_score * (1 - threshold) <= test_score <= ref_score * \
            (1 + threshold):
        print("TEST PASS")
    else:
        print("TEST FAIL: Test score invalid")


if __name__ == "__main__":
    main()
