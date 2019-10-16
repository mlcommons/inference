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
        "--reference_summary", "-r",
        help="Specifies the path to the summary log for TEST00.",
        default=""
    )
    parser.add_argument(
        "--test_summary", "-t",
        help="Specifies the path to the summary log for this test.",
        default=""
    )
    args = parser.parse_args()

    print("Verifying performance.")
    ref_file = open(args.reference_summary, "r")
    test_file = open(args.test_summary, "r")
    ref_score = 0
    test_score = 0
    ref_mode = ''
    test_mode = ''

    for line in ref_file:
        if re.match("Scenario", line):
            ref_mode = line.split(": ",1)[1].strip()
            continue

        if ref_mode == "Single Stream":
            if re.match("90th percentile latency", line):
                ref_score = line.split(": ",1)[1].strip()
                continue

        if ref_mode == "Multi Stream":
            if re.match("Samples per query", line):
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

        if re.match("Result is", line):
            valid = line.split(": ",1)[1].strip()
            if valid == 'INVALID':
                sys.exit("TEST FAIL: Reference results are invalid")

        if re.match("\d+ ERROR", line):
            error = line.split(" ",1)[0].strip()
            print("WARNING: " + error + " ERROR reported in reference results")


    for line in test_file:
        if re.match("Scenario", line):
            test_mode = line.split(": ",1)[1].strip()
            continue

        if test_mode == "Single Stream":
            if re.match("90th percentile latency", line):
                test_score = line.split(": ",1)[1].strip()
                continue

        if test_mode == "Multi Stream":
            if re.match("Samples per query", line):
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

        if re.match("Result is", line):
            valid = line.split(": ",1)[1].strip()
            if valid == 'INVALID':
                sys.exit("TEST FAIL: Test results are invalid")
            
        if re.match("\d+ ERROR", line):
            error = line.split(" ",1)[0].strip()
            print("WARNING: " + error + " ERROR reported in test results")

    if test_mode != ref_mode:
        sys.exit("Test and reference scenarios do not match!")

    print("reference score = {}".format(ref_score))
    print("test score = {}".format(test_score))

 
    threshold = 0.10

    # In single stream mode, latencies can be very short for high performance systems
    # and run-to-run variation due to external disturbances (OS) can be significant.
    # In this case we relax pass threshold to 20%

    if ref_mode == "Single Stream" and float(ref_score) <= 200000:
        threshold = 0.20
        
    if float(test_score) < float(ref_score) * (1 + threshold) and float(test_score) > float(ref_score) * (1 - threshold):
        print("TEST PASS")
    else:
        print("TEST FAIL: Test score invalid")

if __name__ == '__main__':
	main()

