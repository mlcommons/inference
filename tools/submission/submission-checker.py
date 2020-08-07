"""
A checker for mlperf inference submissions
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import re
import sys

# pylint: disable=missing-docstring


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


MODEL_CONFIG = {
    "v0.5": {
        "models": ["ssd-small", "ssd-large", "mobilenet", "resnet", "gnmt"],
        "required-scenarios-datacenter": {
            # anything goes
        },
        "required-scenarios-edge": {
            # anything goes
        },
        "accuracy-target": {
            "mobilenet": ("acc", 71.68 * 0.98),
            "resnet": ("acc", 76.46 * 0.99),
            "ssd-small": ("mAP", 22 * 0.99),
            "ssd-large": ("mAP", 20 * 0.99),
            "gnmt": ("bleu", 23.9 * 0.99),
        },
        "performance-sample-count": {
            "mobilenet": 1024,
            "resnet": 1024,
            "ssd-small": 256,
            "ssd-large": 64,
            "gnmt": 3903900,
        },
        "seeds": {
            "qsl_rng_seed": 3133965575612453542,
            "sample_index_rng_seed": 665484352860916858,
            "schedule_rng_seed": 3622009729038561421,
        },
    },
    "v0.7": {
        "models": ["ssd-large", "resnet", "rnnt", "3d-unet", "dlrm", "bert"],
        "required-scenarios-datacenter": {
            "resnet": ["Server", "Offline"],
            "ssd-large": ["Server", "Offline"],
            "rnnt": ["Server", "Offline"],
            "bert": ["Server", "Offline"],
            "dlrm": ["Server", "Offline"],
            "3d-unet": ["Offline"],
        },
        "required-scenarios-edge": {
            "resnet": ["Server", "Offline"],
            "ssd-large": ["Server", "Offline"],
            "rnnt": ["Server", "Offline"],
            "bert": ["Server", "Offline"],
            "dlrm": ["Server", "Offline"],
            "3d-unet": ["Offline"],
        },
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "ssd-large": ("mAP", 20 * 0.99),
            "rnnt": ("WER", 7.452 * 0.99),
            "bert": ("F1", [90.874 * 0.99, 90.874 * 0.999]),
            "dlrm": ("AUC", [76.46 * 0.99, 76.46 * 0.999]),
            "3d-unet": ("mean", [0.853 * 0.99, 0.853 * 0.999]),
        },
        "performance-sample-count": {
            "ssd-large": 64,
            "resnet": 1024,
            "rnnt": 2513,
            "bert": 3903900,
            "dlrm": 204800,
            "3d-unet": 16,
        },
        "seeds": {
            "qsl_rng_seed": 3133965575612453542,
            "sample_index_rng_seed": 665484352860916858,
            "schedule_rng_seed": 3622009729038561421,
        },
    },
}

VALID_DIVISIONS = ["open", "closed"]
REQUIRED_PERF_FILES = ["mlperf_log_accuracy.json", "mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_ACC_FILES = REQUIRED_PERF_FILES + ["accuracy.txt"]
REQUIRED_MEASURE_FILES = ["mlperf.conf", "user.conf", "README.md"]
TO_MS = 1000 * 1000

MODEL_MAPPING = {
    "ssd-mobilenet": "ssd-small",
    "ssd-resnet34": "ssd-large",
    "resnet50": "resnet"
}

RESULT_FIELD = {
    "Offline": "Samples per second",
    "Single": "90th percentile latency (ns)",
    "Multi": "Samples per query",
    "Server": "Scheduled samples per second"
}

ACC_PATTERN = {
    "acc": r"^accuracy=([\d\.]+).*",
    "AUC": r"^AUC=([\d\.]+).*",
    "mAP": r"^mAP=([\d\.]+).*",
    "bleu": r"^BLEU\:\s*([\d\.]+).*",
    "F1": r"^{\"exact_match\"\:\s*[\d\.]+,\s*\"f1\"\:\s*([\d\.]+)}",
    "WER": r"Word Error Rate\:\s*([\d\.]+).*",
    "mean": r"Accuracy\:\s*mean\s*=\s*([\d\.]+).*",
}

SYSTEM_DESC_REQUIRED_FIELDS = [
    "division", "submitter", "status", "system_name", "number_of_nodes", "host_processor_model_name",
    "host_processors_per_node", "host_processor_core_count", "host_memory_capacity", "host_storage_capacity",
    "host_storage_type", "accelerators_per_node", "accelerator_model_name", "accelerator_memory_capacity",
    "framework", "operating_system"
]

SYSTEM_DESC_OPTIONAL_FIELDS = [
    "system_type", "other_software_stack", "host_processor_frequency", "host_processor_caches",
    "host_memory_configuration", "host_processor_interconnect", "host_networking", "host_networking_topology",
    "accelerator_frequency", "accelerator_host_interconnect", "accelerator_interconnect",
    "accelerator_interconnect_topology", "accelerator_memory_configuration",
    "accelerator_on-chip_memories", "cooling", "hw_notes", "sw_notes"
]

SYSTEM_IMP_REQUIRED_FILES = [
    "input_data_types", "retraining", "starting_weights_filename", "weight_data_types",
    "weight_transformations",
]


class Config():
    """Select config value by mlperf version and submission type."""
    def __init__(self, version):
        self.base = MODEL_CONFIG.get(version)
        self.version = version
        self.models = self.base["models"]
        self.seeds = self.base["seeds"]
        self.accuracy_target = self.base["accuracy-target"]
        self.performance_sample_count = self.base["performance-sample-count"]

    def set_type(self, submission_type):
        if submission_type is None and self.version in ["v0.5"]:
            return
        elif submission_type == "datacenter":
            self.required = self.base["required-scenarios-datacenter"]
        elif submission_type == "edge":
            self.required = self.base["required-scenarios-edge"]
        else:
            raise ValueError("innvalid system type")

    def get_required(self, model):
        if self.version in ["v0.5"]:
            return set()
        if model not in self.required:
            raise ValueError("model not known: " + model)
        return set(self.required[model])

    def get_accuracy_target(self, model):
        if model not in self.accuracy_target:
            raise ValueError("model not known: " + model)
        return self.accuracy_target[model]

    def get_performance_sample_count(self, model):
        if model not in self.performance_sample_count:
            raise ValueError("model not known: " + model)
        return self.performance_sample_count[model]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--version", default="v0.7", choices=list(MODEL_CONFIG.keys()), help="mlperf version")
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument("--csv", default="summary.csv", help="csv file with results")
    args = parser.parse_args()
    return args


def model_map(config, model):
    """Map models names to the official mlperf name."""
    if model in config.models:
        return model
    if model in MODEL_MAPPING:
        return  MODEL_MAPPING[model]
    if model.startswith("mobilenet"):
        model = "mobilenet"
    elif model.startswith("rcnn"):
        model = "ssd-small"
    elif model.startswith("ssdlite") or model.startswith("ssd-inception") or model.startswith("yolo") or \
            model.startswith("ssd-mobilenet") or model.startswith("ssd-resnet50"):
        model = "ssd-small"
    return model


def list_dir(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def list_files(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def split_path(m):
    return m.replace("\\", "/").split("/")


def ignore_errors(line):
    if "check for ERROR in detailed" in line:
        return True
    if "Loadgen built with uncommitted changes" in line:
        return True
    if "Ran out of generated queries to issue before the minimum query count and test duration were reached" in line:
        return True
    if "CAS failed":
        return True
    return False


def check_accuracy_dir(config, model, dir):
    is_valid = False
    acc = None
    model_norm = model_map(config, model)
    acc_type, acc_target = config.get_accuracy_target(model_norm)
    if not isinstance(acc_target, list):
        acc_target = [acc_target]
    acc_target = list(sorted(acc_target, reverse=True))
    pattern = ACC_PATTERN[acc_type]
    with open(os.path.join(dir, "accuracy.txt"), "r") as f:
        for line in f:
            m = re.match(pattern, line)
            if m:
                is_valid = True
                acc = m.group(1)
                break

    if acc:
        for a in acc_target:
            if float(acc) >= a:
                is_valid = True
                break
        if not is_valid:
            log.error("%s accuracy not met: expected=%f, found=%f", dir, acc_target, acc)

    # check if there are any errors in the detailed log
    fname = os.path.join(dir, "mlperf_log_detail.txt")
    if not os.path.exists(fname):
        log.error("%s is missing", fname)
        is_valid = False
    else:
        with open(fname, "r") as f:
            for line in f:
                # look for: ERROR
                if "ERROR" in line:
                    if ignore_errors(line):
                        continue
                    # TODO: should this be a failed run?
                    log.error("%s contains error: %s", fname, line)
                    is_valid = False
    return is_valid, acc


def check_performance_dir(config, model, dir):
    is_valid = False
    rt = {}
    # look for: Result is: VALID
    fname = os.path.join(dir, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
            m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
            if m:
                rt[m.group(1).strip()] = m.group(2).strip()

    model = model_map(config, model)
    performance_sample_count = config.get_performance_sample_count(model)
    if int(rt['performance_sample_count']) < performance_sample_count:
        log.error("%s performance_sample_count should be %d", fname, performance_sample_count)
        is_valid = False

    # check if there are any errors in the detailed log
    fname = os.path.join(dir, "mlperf_log_detail.txt")
    with open(fname, "r") as f:
        for line in f:
            # look for: ERROR
            if "ERROR" in line:
                if ignore_errors(line):
                    continue
                log.error("%s contains error: %s", fname, line)
                is_valid = False

        for seed in ["qsl_rng_seed", "sample_index_rng_seed", "schedule_rng_seed"]:
            if int(rt[seed]) != config.seeds[seed]:
                log.error("%s %s is wrong, expected=%s, found=%s", fname, seed, config.seeds[seed], rt[seed])

    scenario = rt["Scenario"]
    res = float(rt[RESULT_FIELD[scenario]])
    if scenario in ["Single Stream"]:
        res /= TO_MS

    return is_valid, res


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


def check_results_dir(config, dir, filter_submitter, csv):
    head = ["Organization", "Availability", "Division", "Platform", "Model", "Scenario", "Result", "Accuracy", "Location"]
    fmt = ",".join(["{}"] * len(head)) + "\n"
    csv.write(",".join(head) + "\n")
    results = {}

    for division in list_dir("."):
        if division not in VALID_DIVISIONS:
            log.error("invalid division in input dir %s", division)
            continue
        is_closed = division == "closed"
        for submitter in list_dir(division):
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
                log.error("no submission in %s", results_path)
                results[results_path] = None
                continue

            for system_desc in list_dir(results_path):

                # 
                # check if system_id is good.
                #
                system_id_json = os.path.join(division, submitter, "systems", system_desc + ".json")
                if not os.path.exists(system_id_json):
                    log.error("no system_desc for %s/%s/%s", division, submitter, system_desc)
                    results[os.path.join(results_path, system_desc)] = None
                    continue

                name = os.path.join(results_path, system_desc)
                with open(system_id_json) as f:
                    system_json = json.load(f)
                    system_type = system_json.get("system_type")
                    available = system_json.get("status")
                    if config.version == "v0.7" and system_type not in ["datacenter", "edge"]:
                        log.error("%s has invalid system type (%s)", system_id_json, system_type)
                        results[name] = None
                        continue
                    config.set_type(system_type)
                    if not check_system_desc_id(name, system_json, submitter, division):
                        results[name] = None

                # 
                # Look at each model
                #
                for model in list_dir(results_path, system_desc):
                    if is_closed and model not in config.models:
                        log.error("%s has a invalid model (%s) for closed division", name, model)
                        results[name] = None
                        continue

                    # 
                    # Look at each scenario
                    #
                    required_scenarios = config.get_required(MODEL_MAPPING.get(model, model))
                    for scenario in list_dir(results_path, system_desc, model):
                        name = os.path.join(results_path, system_desc, model, scenario)
                        results[name] = None

                        # check if measurement_dir is good.
                        measurement_dir = os.path.join(division, submitter, "measurements",
                                                       system_desc, model, scenario)
                        if not os.path.exists(measurement_dir):
                            log.error("no measurement_dir for %s", name)
                            results[measurement_dir] = None
                        else:
                            if not check_measurement_dir(measurement_dir, name, system_desc,
                                                         os.path.join(division, submitter), model, scenario):
                                log.error("measurement_dir %s has issues", measurement_dir)
                                results[measurement_dir] = None

                        # check accuracy
                        accuracy_is_valid = False
                        acc_path = os.path.join(name, "accuracy")
                        if not os.path.exists(os.path.join(acc_path, "accuracy.txt")):
                            log.error(
                                "%s has no accuracy.txt. Generate it with accuracy-imagenet.py or accuracy-coco.py or "
                                "process_accuracy.py", acc_path)
                        else:
                            diff = files_diff(list_files(acc_path), REQUIRED_ACC_FILES)
                            if diff:
                                log.error("%s has file list mismatch (%s)", acc_path, diff)
                            accuracy_is_valid, acc = check_accuracy_dir(config, model, acc_path)
                            if accuracy_is_valid:
                                log.info("%s, accuracy is %s", acc_path, acc)
                            else:
                                log.error("%s, accuracy not valid", acc_path)

                        if scenario in ["Server"]:
                            n = ["run_1", "run_2", "run_3", "run_4", "run_5"]
                        else:
                            n = ["run_1"]

                        for i in n:
                            perf_path = os.path.join(name, "performance", i)
                            if not os.path.exists(perf_path):
                                log.error("%s is missing", perf_path)
                                continue
                            diff = files_diff(list_files(perf_path), REQUIRED_PERF_FILES)
                            if diff:
                                log.error("%s has file list mismatch (%s)", perf_path, diff)
                            try:
                                is_valid, r = check_performance_dir(config, model, perf_path)
                            except:
                                is_valid, r = False, None
                            if is_valid:
                                results[name] = r
                                required_scenarios.discard(scenario)
                            else:
                                log.error("%s has issues", perf_path)

                        if results.get(name):
                            if accuracy_is_valid:
                                log.info("%s is OK", name)
                                csv.write(fmt.format(submitter, available, division, system_desc, model, scenario,
                                          r, acc, name))
                            else:
                                results[name] = None
                                log.error("%s is OK but accuracy has issues", name)

                    if required_scenarios:
                        name = os.path.join(results_path, system_desc, model)
                        results[name] = None
                        log.error("%s does not have all required scenarios, missing %s", name, required_scenarios)


    return results


def check_system_desc_id(fname, systems_json, submitter, division):
    is_valid = True
    # check all required fields
    for k in SYSTEM_DESC_REQUIRED_FIELDS:
        if k not in systems_json:
            is_valid = False
            log.error("%s, field %s is missing", fname, k)

    all_fields = SYSTEM_DESC_REQUIRED_FIELDS + SYSTEM_DESC_OPTIONAL_FIELDS
    for k in systems_json.keys():
        if k not in all_fields:
            log.warning("%s, field %s is unknwon", fname, k)

    if systems_json.get("submitter") != submitter:
        log.error("%s has submitter %s, directory has %s", fname, systems_json.get("submitter"), submitter)
        is_valid = False
    if systems_json.get("division") != division:
        log.error("%s has division %s, division has %s", fname, systems_json.get("division"), division)
        is_valid = False
    return is_valid


def check_measurement_dir(measurement_dir, fname, system_desc, root, model, scenario):
    files = list_files(measurement_dir)
    system_file = None
    is_valid = True
    for i in REQUIRED_MEASURE_FILES:
        if i not in files:
            log.error("%s is missing %s", measurement_dir, i)
            is_valid = False
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
        with open(os.path.join(measurement_dir, system_file), "r") as f:
            j = json.load(f)
            for k in SYSTEM_IMP_REQUIRED_FILES:
                if k not in j:
                    is_valid = False
                    log.error("%s, field %s is missing", fname, k)

        impl = system_file[len(system_desc) + 1:-end]
        code_dir = os.path.join(root, "code", model, impl)
        if not os.path.exists(code_dir):
            log.error("%s is missing %s*.json", fname, system_desc)
    else:
        log.error("%s is missing %s*.json", fname, system_desc)

    return is_valid


def main():
    args = get_args()

    config = Config(args.version)

    with open(args.csv, "w") as csv:
        os.chdir(args.input)
        # check results directory
        results = check_results_dir(config, args.input, args.submitter, csv)

    # log results
    with_results = 0
    for k, v in results.items():
        if v is None:
            log.error("NoResults %s", k)
        else:
            log.info("Results %s %s", k, v)
            with_results += 1

    # print summary
    log.info("Results=%d, NoResults=%d", with_results, len(results) - with_results)
    if len(results) != with_results: # bad_submissions or meta_errors or measurement_errors:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
