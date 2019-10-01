"""
A checker for mlperf inference submissions
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
REQUIRED_DIRECTORIES = ["code", "results", "measurements", "systems"]
REQUIRED_PERF_FILES = ["mlperf_log_accuracy.json", "mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_ACC_FILES = REQUIRED_PERF_FILES + ["accuracy.txt"]
REQUIRED_MESAURE_FILES = ["mlperf.conf", "README.md"]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    args = parser.parse_args()
    return args


def path_to_dict(path):
    d, f = {}, {}
    for dirName, subdirList, fileList in os.walk(path):
        d[dirName] = sorted(subdirList)
        f[dirName] = sorted(fileList)
    return d, f


def check_accuracy(model, dir):
    with open(os.path.join(dir, "accuracy.txt"), "r") as f:
        for line in f:
            # look for: accuracy = 71.676%, good = 35838, total = 50000
            m = re.match("^accuracy=(\d+\.\d+).*", line)
            if m:
                return True
            # look for: mAP = 25.460%
            m = re.match("^mAP=(\d+\.\d+).*", line)
            if m:
                return True
                # TODO: we might as well check the total count
    return False


def check_performance(model, dir):
    with open(os.path.join(dir, "mlperf_log_summary.txt"), "r") as f:
        for line in f:
            # look for: Result is: VALID
            m = re.match("^Result\s+is\s*\:\s+VALID", line)
            if m:
                return True
    return False


def files_exists(list1, list2):
    if list1 and list2:
        for i in ["mlperf_log_trace.json", "results.json"]:
            try:
                list1.remove(i)
            except:
                pass
        return sorted(list1) == sorted(list2)
    return False


def check_results_dir(dir):
    good_submissions = []
    bad_submissions = {}
    dirs, files = path_to_dict("results")

    for device in dirs["results"]:
        # check if system_id is good. Report failure for each model/scenario.
        system_id = os.path.join("systems", device + ".json")
        device_bad = not os.path.exists(system_id)
        for model in dirs[os.path.join("results", device)]:
            for scenario in dirs[os.path.join("results", device, model)]:
                name = os.path.join(device, model, scenario)
                acc_path = os.path.join("results", device, model, scenario, "accuracy")
                if not files_exists(files.get(acc_path), REQUIRED_ACC_FILES):
                    bad_submissions[name] = "{} has file missmatch".format(acc_path)
                    continue
                if not check_accuracy(model, acc_path):
                    bad_submissions[name] = "{} has issues".format(acc_path)
                    continue
                n = ["1"]
                if scenario in ["Server"]:
                    n = ["1", "2", "3", "4", "5"]
                for i in n:
                    perf_path = os.path.join("results", device, model, scenario, "performance", i)
                    if not files_exists(files.get(perf_path), REQUIRED_PERF_FILES):
                        bad_submissions[name] = "{} has file missmatch".format(perf_path)
                        continue
                    if not check_performance(model, perf_path):
                        bad_submissions[name] = "{} has issues".format(perf_path)
                        continue
                if device_bad:
                    bad_submissions[name] = "results/{}: no such system id {}".format(name, system_id)
                else:
                    good_submissions.append(name)
        for k, v in bad_submissions.items():
            log.error(v)
        for name in good_submissions:
            log.info("results/{} OK".format(name))

    return good_submissions, bad_submissions


def check_meta(dir, good_submissions, meta):
    system_ids = set([i.replace("\\", "//").split("/")[0] for i in good_submissions])
    for system_id in system_ids:
        errors = []
        fname = os.path.join("systems", system_id + ".json")
        try:
            with open(fname, "r") as f:
                j = json.load(f)
                # make sure all required sections/fields are in the meta data
                for k, v in meta.items():
                    z = j.get(k)
                    if z is None:
                        errors.append("{} has no section".format(fname, k))
                    else:
                        for field, fieldval in v.items():
                            sz = j[k].get(field)
                            if sz is None and fieldval == "required":
                                errors.append("{} section {} field {} missing".format(fname, k, field))
                # make sure no undefined sections/fields are in the meta data
                for k, v in j.items():
                    z = meta.get(k)
                    if z is None:
                        errors.append("{} has unknwon section {}".format(fname, k))
                    else:
                        for field, fieldval in v.items():
                            sz = meta[k].get(field)
                            if sz is None:
                                errors.append("{} section {} field {} unknown".format(fname, k, field))

        except Exception as ex:
            errors.append("{} unexpected error {}".format(fname, ex))
        if errors:
            for i in errors:
                log.error(i)
        else:
            log.info("{} OK".format(fname))
        return errors


def check_measurement_dir(good_submissions):
    errors = []
    for submission in good_submissions:
        fname = os.path.join("measurements", submission)
        if not os.path.exists(fname):
            errors.append("{} directory missing".format(fname))
        cols = submission.replace("\\", "/").split("/")
        system_id = cols[0]
        system_file = None
        dirs, files = path_to_dict(fname)
        for i in REQUIRED_MESAURE_FILES:
            if i not in files[fname]:
                errors.append("{} is missing {}".format(fname, i))
        for i in files[fname]:
            if i.startswith(system_id) and i.endswith(".json"):
                system_file = i
                break
        if not system_file:
            errors.append("{} is missing {}*.json".format(fname, system_id))
        impl = system_file[len(system_id) + 1:-5]
        code_dir = os.path.join("code", cols[1], impl)
        if not os.path.exists(code_dir):
            errors.append("{} is missing code dir {}".format(fname, code_dir))

    if errors:
        for i in errors:
            log.error(i)
    else:
        log.info("{} OK".format(fname))
    return errors


def main():
    args = get_args()

    script_path = os.path.dirname(sys.argv[0])
    with open(os.path.join(script_path, "meta.json"), "r") as f:
        meta = json.load(f)

    os.chdir(args.input)
    # 1. check results directory
    good_submissions, bad_submissions = check_results_dir(args.input)

    # 2. check the meta data under systems
    meta_errors = check_meta(args.input, good_submissions, meta)

    # 3. check measurement and code dir
    measurement_errros = check_measurement_dir(good_submissions)
    if bad_submissions or meta_errors or measurement_errros:
        log.error("SUMMARY: there are errros in the submission")
    else:
        log.info("SUMMARY: submission looks OK")


if __name__ == "__main__":
    main()
