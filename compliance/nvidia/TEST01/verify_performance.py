#!/usr/bin/env python3
# Copyright 2018-2022 The MLPerf Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# Full license text: http://www.apache.org/licenses/LICENSE-2.0

import argparse
import os
import sys
import re

sys.path.append(os.getcwd())

# Mapping scenarios to their respective metrics
METRIC_MAP = {
    "SingleStream": ".*Early stopping (90th|99.9th) percentile estimate",
    "MultiStream": ".*Early stopping 99th percentile estimate",
    "Server": "Completed samples per second",
    "Offline": "Samples per second"
}

def parse_summary(file_path):
    """Parses the summary file and extracts mode, score, and target latency."""
    mode, score, target_latency = None, None, None

    with open(file_path, "r") as file:
        for line in file:
            if re.match("Scenario", line):
                mode = line.split(": ", 1)[1].strip()
                continue

            if mode in METRIC_MAP and re.match(METRIC_MAP[mode], line):
                score = float(line.split(": ", 1)[1].strip())
                continue

            if mode == "Server" and re.match("target_latency \(ns\)", line):
                target_latency = float(line.split(": ", 1)[1].strip())
                continue

            if re.match("Result is", line):
                if "INVALID" in line:
                    sys.exit(f"TEST FAIL: {file_path} results are invalid")

            if re.match(r"\d+ ERROR", line):
                error_count = line.split(" ", 1)[0].strip()
                print(f"WARNING: {error_count} ERROR reported in {file_path}")

    return mode, score, target_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference_summary", required=True, help="Path to reference performance summary log.")
    parser.add_argument("-t", "--test_summary", required=True, help="Path to test performance summary log.")
    args = parser.parse_args()

    print("Verifying performance...")

    ref_mode, ref_score, ref_target_latency = parse_summary(args.reference_summary)
    test_mode, test_score, test_target_latency = parse_summary(args.test_summary)

    if ref_mode != test_mode:
        sys.exit("TEST FAIL: Test and reference scenarios do not match!")

    if ref_mode == "Server" and ref_target_latency and test_target_latency and ref_target_latency != test_target_latency:
        sys.exit("TEST FAIL: Server target latency mismatch")

    print(f"Reference score = {ref_score}")
    print(f"Test score = {test_score}")

    # Define thresholds
    threshold = 0.10
    if (ref_mode == "SingleStream" and ref_score <= 200000) or (ref_mode == "MultiStream" and ref_score <= 1600000):
        threshold = 0.20  # Increased tolerance for short latencies

    # Performance validation
    lower_bound, upper_bound = ref_score * (1 - threshold), ref_score * (1 + threshold)
    if lower_bound <= test_score <= upper_bound:
        print("TEST PASS")
    else:
        print("TEST FAIL: Test score invalid")

if __name__ == "__main__":
    main()
