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
VALID_DIVISIONS = ["open", "closed"]
REQUIRED_PERF_FILES = ["mlperf_log_accuracy.json", "mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_ACC_FILES = REQUIRED_PERF_FILES + ["accuracy.txt"]
REQUIRED_MEASURE_FILES = ["mlperf.conf", "user.conf", "README.md"]
PERFORMANCE_SAMPLE_COUNT = {
    "mobilenet": 1024,
    "resnet50": 1024,
    "resnet": 1024,
    "ssd-mobilenet": 256,
    "ssd-small": 256,
    "ssd-resnet34": 64,
    "ssd-large": 64,
    "gnmt": 3903900,
}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
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
    rt = {}
    # look for: Result is: VALID
    fname = os.path.join(dir, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match("^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
            m = re.match("^\s*([\w\s.\(\)\/]+)\s*\:\s*([\d\.]+).*", line)
            if m:
                rt[m.group(1).strip()] = m.group(2).strip()

    if int(rt['performance_sample_count']) < PERFORMANCE_SAMPLE_COUNT[model]:
        log.error("{} performance_sample_count should be {}".format(fname, PERFORMANCE_SAMPLE_COUNT[model]))
        is_valid = False

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


def check_results_dir(dir, filter_submitter):
    good_submissions = []
    bad_submissions = {}

    for division in list_dir("."):
        for submitter in list_dir(division):
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            for system_desc in list_dir(results_path):
                # check if system_id is good. Report failure for each model/scenario.
                system_id_json = os.path.join(division, submitter, "systems", system_desc + ".json")
                device_bad = not os.path.exists(system_id_json)
                for model in list_dir(results_path, system_desc):
                    if model not in VALID_MODELS:
                        bad_submissions[os.path.join(system_desc, model)] = \
                            "{} has an invalid model name {}".format(os.path.join(results_path, system_desc), model)
                        log.error("{} has an invalid model name {}".format(os.path.join(results_path, system_desc), model))
                        continue
                    for scenario in list_dir(results_path, system_desc, model):
                        name = os.path.join(results_path, system_desc, model, scenario)
                        acc_path = os.path.join(name, "accuracy")
                        if not os.path.exists(os.path.join(acc_path, "accuracy.txt")):
                            log.error("{} has no accuracy.txt. Generate it with accuracy-imagenet.py or accuracy-coco.py or "
                                      "process_accuracy.py".format(acc_path))
                        diff = files_diff(list_files(acc_path), REQUIRED_ACC_FILES)
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
                            perf_path = os.path.join(name, "performance", "run_" + str(i))
                            diff = files_diff(list_files(perf_path), REQUIRED_PERF_FILES)
                            if diff:
                                bad_submissions[name] = "{} has file list mismatch ({})".format(perf_path, diff)
                                continue
                            if not check_performance_dir(model, perf_path):
                                bad_submissions[name] = "{} has issues".format(perf_path)
                                continue
                        if device_bad:
                            bad_submissions[name] = "{}: no such system id {}".format(name, system_desc)
                        else:
                            good_submissions.append(name)
        for k, v in bad_submissions.items():
            log.error(v)
        for name in good_submissions:
            log.info("{} OK".format(name))

    return good_submissions, bad_submissions


def compare_json(fname, template, errors):
    error_count = len(errors)
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
    return error_count == len(errors)


def check_system_desc_id(good_submissions, systems_json):
    errors = []
    checked = set()
    for submission in good_submissions:
        parts = split_path(submission)
        system_desc = parts[3]
        submitter = parts[1]
        division = parts[0]
        if division not in VALID_DIVISIONS:
            errors.append(("{} has invalid division {}".format(submission, j["submitter"], division)))
            continue

        fname = os.path.join(parts[0], parts[1], "systems", system_desc + ".json")
        if fname not in checked:
            checked.add(fname)
            if not compare_json(fname, systems_json, errors):
                continue
            with open(fname, "r") as f:
                j = json.load(f)
                if j["submitter"] != submitter:
                    errors.append(("{} has submitter {}, directory has {}".format(fname, j["submitter"], submitter)))
                    continue
                if j["division"] != division:
                    errors.append(("{} has division {}, division has {}".format(fname, j["division"], division)))
                    continue
            log.info("{} OK".format(fname))
    if errors:
        for i in errors:
            log.error(i)
    return errors


def check_measurement_dir(good_submissions, systems_imp_json):
    errors = []
    for submission in good_submissions:
        parts = split_path(submission)
        system_desc = parts[3]
        measurement_dir = os.path.join(parts[0], parts[1], "measurements",  system_desc)
        if not os.path.exists(measurement_dir):
            errors.append("{} directory missing".format(measurement_dir))
            continue
        model = parts[4]
        scenario = parts[5]
        fname = os.path.join(measurement_dir, model, scenario)
        files = list_files(fname)
        system_file = None
        for i in REQUIRED_MEASURE_FILES:
            if i not in files:
                errors.append("{} is missing {}".format(fname, i))
        for i in files:
            if i.startswith(system_desc) and i.endswith("_" + scenario + ".json"):
                system_file = i
                end = len("_" + scenario + ".json")
                break
            elif i.startswith(system_desc) and i.endswith(".json"):
                system_file = i
                end = len(".json")
                break
        if system_file:
            compare_json(os.path.join(fname, system_file), systems_imp_json, errors)
            impl = system_file[len(system_desc) + 1:-end]
            code_dir = os.path.join(parts[0], parts[1], "code", model, impl)
            if not os.path.exists(code_dir):
                errors.append("{} is missing".format(code_dir))
            else:
                log.info("{} OK".format(fname))
        else:
            errors.append("{} is missing {}*.json".format(fname, system_desc))

    if errors:
        for i in errors:
            log.error(i)
    return errors


def main():
    args = get_args()

    script_path = os.path.dirname(sys.argv[0])
    with open(os.path.join(script_path, "system_desc_id.json"), "r") as f:
        systems_json = json.load(f)
    with open(os.path.join(script_path, "system_desc_id_imp.json"), "r") as f:
        systems_imp_json = json.load(f)

    os.chdir(args.input)

    # 1. check results directory
    good_submissions, bad_submissions = check_results_dir(args.input, args.submitter)

    # 2. check the meta data under systems
    meta_errors = check_system_desc_id(good_submissions, systems_json)

    # 3. check measurement and code dir
    measurement_errors = check_measurement_dir(good_submissions, systems_imp_json)
    if bad_submissions or meta_errors or measurement_errors:
        log.error("SUMMARY: there are errros in the submission")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
