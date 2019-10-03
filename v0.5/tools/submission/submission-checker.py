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
REQUIRED_PERF_FILES = ["mlperf_log_accuracy.json", "mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_ACC_FILES = REQUIRED_PERF_FILES + ["accuracy.txt"]
REQUIRED_MEASURE_FILES = ["mlperf.conf", "user.conf", "README.md"]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    args = parser.parse_args()
    return args


def split_path(m):
    return m.replace("\\", "/").split("/")


def path_to_dict(path):
    d, f = {}, {}
    for dirName, subdirList, fileList in os.walk(path):
        d[dirName] = sorted(subdirList)
        f[dirName] = sorted(fileList)
    return d, f


def check_accuracy_dir(model, dir):
    is_valid = False
    # look for: accuracy=... or mAP=...
    with open(os.path.join(dir, "accuracy.txt"), "r") as f:
        for line in f:
            m = re.match("^accuracy=([\d\.]+).*", line)
            if m:
                is_valid = True
                break
            m = re.match("^mAP=([\d\.]+).*", line)
            if m:
                is_valid = True
                break
            m = re.match("^BLEU\:\s*([\d\.]+).*", line)
            if m:
                is_valid = True
                break
    # check if there are any errors in the detailed log
    fname = os.path.join(dir, "mlperf_log_detail.txt")
    with open(fname, "r") as f:
        for line in f:
            # look for: ERROR
            if "ERROR" in line:
                # TODO: should this be a failed run?
                log.warning("{} contains errors".format(fname))
    return is_valid


def check_performance_dir(model, dir):
    is_valid = False
    # look for: Result is: VALID
    with open(os.path.join(dir, "mlperf_log_summary.txt"), "r") as f:
        for line in f:
            m = re.match("^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
                break
    # check if there are any errors in the detailed log
    fname = os.path.join(dir, "mlperf_log_detail.txt")
    with open(fname, "r") as f:
        for line in f:
            # look for: ERROR
            if "ERROR" in line:
                # TODO: does this make the run fail?
                log.warning("{} contains errors".format(fname))
    return is_valid


def files_diff(list1, list2):
    """returns a list of files that are missing or added."""
    if list1 and list2:
        for i in ["mlperf_log_trace.json", "results.json"]:
            try:
                list1.remove(i)
            except:
                pass
        if len(list1) > len(list2):
            return list(set(list1) - set(list2))
        else:
            return list(set(list2) - set(list1))
    return []


def check_results_dir(dir):
    good_submissions = []
    bad_submissions = {}
    dirs, files = path_to_dict("results")

    for device in dirs["results"]:
        # check if system_id is good. Report failure for each model/scenario.
        system_id = os.path.join("systems", device + ".json")
        device_bad = not os.path.exists(system_id)
        for model in dirs[os.path.join("results", device)]:
            if model not in VALID_MODELS:
                bad_submissions[os.path.join(device, model)] = \
                    "{} has an invalid model name {}".format(os.path.join("results", device), model)
                log.error("{} has an invalid model name {}".format(os.path.join("results", device), model))
                continue
            for scenario in dirs[os.path.join("results", device, model)]:
                name = os.path.join(device, model, scenario)
                acc_path = os.path.join("results", device, model, scenario, "accuracy")
                if not os.path.exists(os.path.join(acc_path, "accuracy.txt")):
                    log.error("{} has no accuracy.txt. Generate it with accuracy-imagenet.py or accuracy-coco.py or "
                              "process_accuracy.py".format(acc_path))
                diff = files_diff(files.get(acc_path), REQUIRED_ACC_FILES)
                if diff:
                    bad_submissions[name] = "{} has file list mismatch ({})".format(acc_path, diff)
                    continue
                if not check_accuracy_dir(model, acc_path):
                    bad_submissions[name] = "{} has issues".format(acc_path)
                    continue
                n = ["1"]
                if scenario in ["Server"]:
                    n = ["1", "2", "3", "4", "5"]
                for i in n:
                    perf_path = os.path.join("results", device, model, scenario, "performance", "run_" + str(i))
                    diff = files_diff(files.get(perf_path), REQUIRED_PERF_FILES)
                    if diff:
                        bad_submissions[name] = "{} has file list mismatch ({})".format(perf_path, diff)
                        continue
                    if not check_performance_dir(model, perf_path):
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


def compare_json(fname, template, errors):
    try:
        with open(fname, "r") as f:
            j = json.load(f)
        # make sure all required sections/fields are there
        for k, v in template.items():
            sz = j.get(k)
            if sz is None and v == "required":
                errors.append("{} field {} missing".format(fname, k))

        # make sure no undefined sections/fields are in the meta data
        for k, v in j.items():
            z = template.get(k)
            if z is None:
                errors.append("{} has unknwon field {}".format(fname, k))
    except Exception as ex:
        errors.append("{} unexpected error {}".format(fname, ex))


def check_meta(dir, good_submissions, system_desc_id, system_desc_id_imp):
    system_ids = set([split_path(i)[0] for i in good_submissions])
    for system_id in system_ids:
        errors = []
        fname = os.path.join("systems", system_id + ".json")
        compare_json(fname, system_desc_id, errors)
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
        cols = split_path(submission)
        system_id = cols[0]
        system_file = None
        dirs, files = path_to_dict(fname)
        for i in REQUIRED_MEASURE_FILES:
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
        log.info("{} OK".format(fname))

    if errors:
        for i in errors:
            log.error(i)
    return errors


def main():
    args = get_args()

    script_path = os.path.dirname(sys.argv[0])
    with open(os.path.join(script_path, "system_desc_id.json"), "r") as f:
        system_desc_id = json.load(f)
    with open(os.path.join(script_path, "system_desc_id_imp.json"), "r") as f:
        system_desc_id_imp = json.load(f)

    os.chdir(args.input)
    # 1. check results directory
    good_submissions, bad_submissions = check_results_dir(args.input)

    # 2. check the meta data under systems
    meta_errors = check_meta(args.input, good_submissions, system_desc_id, system_desc_id_imp)

    # 3. check measurement and code dir
    measurement_errros = check_measurement_dir(good_submissions)
    if bad_submissions or meta_errors or measurement_errros:
        log.error("SUMMARY: there are errros in the submission")
    else:
        log.info("SUMMARY: submission looks OK")


if __name__ == "__main__":
    main()
