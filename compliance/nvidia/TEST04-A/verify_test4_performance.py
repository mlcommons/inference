#! /usr/bin/env python3
import os
import sys
import re
sys.path.append(os.getcwd())

import argparse
import json

def main():
    # Parse arguments to identify the path to the accuracy logs from
    #   the accuracy and performance runs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unique_sample", "-u",
        help="Specifies the path to the summary log for TEST04-A.",
        default=""
    )
    parser.add_argument(
        "--same_sample", "-s",
        help="Specifies the path to the summary log for TEST04-B.",
        default=""
    )
    args = parser.parse_args()

    print("Verifying performance.")
    ref_file = open(args.unique_sample, "r")
    test_file = open(args.same_sample, "r")
    ref_score = 0
    test_score = 0
    ref_mode = ''
    test_mode = ''
    performance_issue_unqiue = ''
    performance_issue_same = ''

    for line in ref_file:
        if re.match("Scenario", line):
            ref_mode = line.split(": ",1)[1].strip()
            continue

        if ref_mode == "SingleStream":
            if re.match("90th percentile latency", line):
                ref_score = line.split(": ",1)[1].strip()
                continue

        if ref_mode == "MultiStream":
            if re.match("99th percentile latency", line):
                ref_score = line.split(": ",1)[1].strip()
                continue

        if ref_mode == "Server":
            if re.match("Scheduled samples per second", line):
                ref_score = line.split(": ",1)[1].strip()
                continue

        if ref_mode == "Offline":
            if re.match("Samples per second", line):
                ref_score = line.split(": ",1)[1].strip()
                continue


        if re.match("\d+ ERROR", line):
            error = line.split(" ",1)[0].strip()
            print("WARNING: " + error + " ERROR reported in TEST04-A results")

        if re.match("performance_issue_unique",  line):
            performance_issue_unique = line.split(": ",1)[1].strip()
            if performance_issue_unique == 'false':
                sys.exit("TEST FAIL: Invalid test settings in TEST04-A summary.")
            break

    for line in test_file:
        if re.match("Scenario", line):
            test_mode = line.split(": ",1)[1].strip()
            continue

        if test_mode == "SingleStream":
            if re.match("90th percentile latency", line):
                test_score = line.split(": ",1)[1].strip()
                continue

        if test_mode == "MultiStream":
            if re.match("99th percentile latency", line):
                test_score = line.split(": ",1)[1].strip()
                continue

        if test_mode == "Server":
            if re.match("Scheduled samples per second", line):
                test_score = line.split(": ",1)[1].strip()
                continue

        if test_mode == "Offline":
            if re.match("Samples per second", line):
                test_score = line.split(": ",1)[1].strip()
                continue

        if re.match("\d+ ERROR", line):
            error = line.split(" ",1)[0].strip()
            print("WARNING: " + error + " ERROR reported in TEST04-B results")

        if re.match("performance_issue_same",  line):
            performance_issue_same = line.split(": ",1)[1].strip()
            if performance_issue_same == 'false':
                sys.exit("TEST FAIL: Invalid test settings in TEST04-B summary.")
            break

    if test_mode != ref_mode:
        sys.exit("Test and reference scenarios do not match!")

    print("TEST04-A score = {}".format(ref_score))
    print("TEST04-B score = {}".format(test_score))

    threshold = 0.10

    # In single stream mode, latencies can be very short for high performance systems
    # and run-to-run variation due to external disturbances (OS) can be significant.
    # In this case we relax pass threshold to 20%

    if ref_mode == "SingleStream" and float(ref_score) <= 200000:
        threshold = 0.20

    if float(test_score) < float(ref_score) * (1 + threshold) and float(test_score) > float(ref_score) * (1 - threshold):
        print("TEST PASS")
    elif (float(test_score) > float(ref_score) and test_mode == "SingleStream"):
        print("TEST PASS")
        print("Note: TEST04-B is significantly slower than TEST04-A")
    elif (float(test_score) < float(ref_score) and test_mode != "SingleStream"):
        print("TEST PASS")
        print("Note: TEST04-B is significantly slower than TEST04-A")
    else:
        print("TEST FAIL: Test score invalid")

if __name__ == '__main__':
	main()

