"""
Tool to create a csv file from a mlperf inference submission directory
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import json
import logging
import os
import re
import sys
import time

# pylint: disable=missing-docstring


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

VALID_MODELS = ["ssd-small", "ssd-large", "mobilenet", "resnet", "gnmt"]
VALID_DIVISIONS = ["open", "closed"]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--output", help="output")
    parser.add_argument("--submitter", help="filter to submitter")
    args = parser.parse_args()
    return args


def list_dir(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def list_files(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def split_path(m):
    return m.replace("\\", "/").split("/")


def model_map(model):
    if model.startswith("mobilenet"):
        model = "mobilenet"
    elif model.startswith("rcnn"):
        model = "ssd-small"
    elif model.startswith("resnet50"):
        model = "resnet"
    elif model.startswith("ssdlite") or model.startswith("ssd-inception") or model.startswith("yolo") or \
            model.startswith("ssd-mobilenet") or model.startswith("ssd-resnet50"):
        model = "ssd-small"
    if model not in VALID_MODELS:
        model = None
    return model


def get_accuracy(model, dir):
    is_valid = False
    acc = 0
    # look for: accuracy=... or mAP=...
    with open(os.path.join(dir, "accuracy.txt"), "r") as f:
        for line in f:
            m = re.match("^accuracy=([\d\.]+).*", line)
            if m:
                acc = m.group(1)
                break
            m = re.match("^mAP=([\d\.]+).*", line)
            if m:
                acc = m.group(1)
                break
            m = re.match("^BLEU\:\s*([\d\.]+).*", line)
            if m:
                acc = m.group(1)
                break
    return float(acc)


RESULT_VALUE = {
    "Offline": "Samples per second",
    "SingleStream": "90th percentile latency (ns)",
    "MultiStream": "Samples per query",
    "Server": "Scheduled samples per second"
}

TOMS = 1000 * 1000


def get_performance(model, scenario, dir, kv):
    rt = {}
    # look for: Result is: VALID
    fname = os.path.join(dir, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match("^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
            if m:
                rt[m.group(1).strip()] = m.group(2).strip()

    if scenario == "singlestream":
        scenario = "SingleStream"
    if scenario == "server":
        scenario = "Server"
    if scenario == "offline":
        scenario = "Offline"
    if scenario == "multistream":
        scenario = "MultiStream"
    kv["scenario"] = scenario
    res = float(rt[RESULT_VALUE[scenario]])
    if scenario in ["SingleStream"]:
        res /= TOMS
    kv["result"] = res
    kv["p50"] = float(rt["50.00 percentile latency (ns)"]) / TOMS
    kv["p90"] = float(rt["90.00 percentile latency (ns)"]) / TOMS
    kv["p99"] = float(rt["99.00 percentile latency (ns)"]) / TOMS


def walk_results_dir(dir, filter_submitter, results):
    for division in list_dir("."):
        if division not in ["closed", "open"]:
            continue
        for submitter in list_dir(division):
            if "example" in submitter:
                continue
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
                log.warning("no submission in {}/{}".format(division, submitter))
                continue
            for system_desc in list_dir(results_path):
                # check if system_id is good. Report failure for each model/scenario.
                for model in list_dir(results_path, system_desc):
                    try:
                        model_norm = model_map(model)
                        for scenario in list_dir(results_path, system_desc, model):
                            name = os.path.join(results_path, system_desc, model, scenario).replace("\\", "/")
                            nn = os.path.join(submitter, division, system_desc, model)
                            kv = {"name": nn, "model": model_norm, "system": system_desc,
                                  "division": division, "submitter": submitter}
                            acc_path = os.path.join(name, "accuracy")
                            if not os.path.exists(os.path.join(acc_path, "accuracy.txt")):
                                log.error("{} has no accuracy.txt".format(acc_path))
                            kv["acc"] = get_accuracy(model, acc_path)
                            n = ["1"]
                            for i in n:
                                perf_path = os.path.join(name, "performance", "run_" + str(i))
                                get_performance(model_norm, scenario, perf_path, kv)
                            results.append(kv)
                    except Exception as ex:
                        log.error("{}, {}".format(name, ex))


def main():
    args = get_args()

    os.chdir(args.input)

    results = []
    walk_results_dir(args.input, args.submitter, results)
    columns = ['name', 'model', 'system', 'division', 'submitter', 'acc', 'scenario', 'result',
               'p50', 'p90', 'p99']
    if args.output:
        with open(args.output, "w") as f:
            f.write(",".join(columns) + "\n")
            for r in results:
                col = [str(r[c]) for c in columns]
                f.write(",".join(col) + "\n")


if __name__ == "__main__":
    sys.exit(main())
