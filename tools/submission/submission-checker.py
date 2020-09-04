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
        "optional-scenarios-datacenter": {
            # anything goes
        },
        "required-scenarios-edge": {
            # anything goes
        },
        "optional-scenarios-edge": {
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
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-mobilenet": "ssd-small",
            "ssd-resnet34": "ssd-large",
            "resnet50": "resnet",
        },
        "seeds": {
            "qsl_rng_seed": 3133965575612453542,
            "sample_index_rng_seed": 665484352860916858,
            "schedule_rng_seed": 3622009729038561421,
        },
        "ignore_errors": [
            "check for ERROR in detailed",
            "Loadgen built with uncommitted changes",
            "Ran out of generated queries to issue before the minimum query count and test duration were reached",
            "CAS failed",
        ],
    },
    "v0.7": {
        "models": [
            "ssd-small", "ssd-large", "resnet", "rnnt",
            "bert-99", "bert-99.9",
            "dlrm-99", "dlrm-99.9",
            "3d-unet-99", "3d-unet-99.9",
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Server", "Offline"],
            "ssd-large": ["Server", "Offline"],
            "rnnt": ["Server", "Offline"],
            "bert-99": ["Server", "Offline"],
            "bert-99.9": ["Server", "Offline"],
            "dlrm-99": ["Server", "Offline"],
            "dlrm-99.9": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
        },
        "optional-scenarios-datacenter": {
        },
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "Offline"],
            "ssd-small": ["SingleStream", "Offline"],
            "ssd-large": ["SingleStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-edge": {
            "resnet": ["MultiStream"],
            "ssd-small": ["MultiStream"],
            "ssd-large": ["MultiStream"],
        },
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "ssd-small": ("mAP", 22 * 0.99),
            "ssd-large": ("mAP", 20 * 0.99),
            "rnnt": ("WER", (100 - 7.452) * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-99": ("AUC", 80.25 * 0.99),
            "dlrm-99.9": ("AUC", 80.25 * 0.999),
            "3d-unet-99": ("DICE", 0.853 * 0.99),
            "3d-unet-99.9": ("DICE", 0.853 * 0.999),
        },
        "performance-sample-count": {
            "ssd-small": 256,
            "ssd-large": 64,
            "resnet": 1024,
            "rnnt": 2513,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-99": 204800,
            "dlrm-99.9": 204800,
            "3d-unet-99": 16,
            "3d-unet-99.9": 16,
        },
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-mobilenet": "ssd-small",
            "ssd-resnet34": "ssd-large",
            "mobilenet": "resnet",
            "resnet50": "resnet",
        },
        "seeds": {
            "qsl_rng_seed": 12786827339337101903,
            "sample_index_rng_seed": 12640797754436136668,
            "schedule_rng_seed": 3135815929913719677,
        },
        "ignore_errors": [
            "CAS failed",
        ],
        "latency-constraint": {
            "resnet": {"Server": 15000000, "MultiStream": 50000000},
            "ssd-small": {"MultiStream": 50000000},
            "ssd-large": {"Server": 100000000, "MultiStream": 66000000},
            "rnnt": {"Server": 1000000000},
            "bert-99": {"Server": 130000000},
            "bert-99.9": {"Server": 130000000},
            "dlrm-99": {"Server": 30000000},
            "dlrm-99.9": {"Server": 30000000},
        },
        "min-queries": {
            "resnet": {"SingleStream":1024, "Server": 270336, "MultiStream": 270336, "Offline": 1},
            "ssd-small": {"SingleStream":1024, "MultiStream": 270336, "Offline": 1},
            "ssd-large": {"SingleStream":1024, "Server": 270336, "MultiStream": 270336, "Offline": 1},
            "rnnt": {"SingleStream": 1024, "Server": 90112, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 90112, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 90112, "Offline": 1},
            "dlrm-99": {"Server": 90112, "Offline": 1},
            "dlrm-99.9": {"Server": 90112, "Offline": 1},
            "3d-unet-99": {"SingleStream":1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream":1024, "Offline": 1},
        },
    },
}

VALID_DIVISIONS = ["open", "closed"]
REQUIRED_PERF_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
OPTIONAL_PERF_FILES = ["mlperf_log_accuracy.json"]
REQUIRED_ACC_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt", "accuracy.txt", "mlperf_log_accuracy.json"]
REQUIRED_MEASURE_FILES = ["mlperf.conf", "user.conf", "README.md"]
TO_MS = 1000 * 1000
MAX_ACCURACY_LOG_SIZE = 10 * 1024
OFFLINE_MIN_SPQ = 24576
TEST_DURATION_MS = 60000

SCENARIO_MAPPING = {
    "singlestream": "SingleStream",
    "multistream": "MultiStream",
    "server": "Server",
    "offline": "Offline",
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
    "WER": r"Word Error Rate\:.*, accuracy=([0-9\.]+)%",
    "DICE": r"Accuracy\:\s*mean\s*=\s*([\d\.]+).*",
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
        self.latency_constraint = self.base.get("latency-constraint", {})
        self.min_queries = self.base.get("min-queries", {})
        self.required = None
        self.optional = None

    def set_type(self, submission_type):
        if submission_type is None and self.version in ["v0.5"]:
            return
        elif submission_type == "datacenter":
            self.required = self.base["required-scenarios-datacenter"]
            self.optional = self.base["optional-scenarios-datacenter"]
        elif submission_type == "edge":
            self.required = self.base["required-scenarios-edge"]
            self.optional = self.base["optional-scenarios-edge"]
        else:
            raise ValueError("invalid system type")

    def get_mlperf_model(self, model):
        # prefered - user is already using the official name
        if model in self.models:
            return model
        
        # simple mapping, ie resnet50->resnet ?
        mlperf_model = self.base["model_mapping"].get(model)
        if mlperf_model:
            return mlperf_model

        # try to guess
        if model.startswith("mobilenet"):
            model = "mobilenet"
        if model.startswith("efficientnet"):
            model = "resnet"
        elif model.startswith("rcnn"):
            model = "ssd-small"
        elif model.startswith("ssdlite") or model.startswith("ssd-inception") or model.startswith("yolo") or \
            model.startswith("ssd-mobilenet") or model.startswith("ssd-resnet50"):
            model = "ssd-small"
        # map again, for example v0.7 does not have mobilenet so it needs to be mapped to resnet
        mlperf_model = self.base["model_mapping"].get(model, model)
        return mlperf_model

    def get_required(self, model):
        if self.version in ["v0.5"]:
            return set()
        model = self.get_mlperf_model(model)
        if model not in self.required:
            return None
        return set(self.required[model])

    def get_optional(self, model):
        if self.version in ["v0.5"]:
            return set(["SingleStream", "MultiStream", "Server", "Offline"])
        model = self.get_mlperf_model(model)
        if model not in self.optional:
            return set()
        return set(self.optional[model])

    def get_accuracy_target(self, model):
        if model not in self.accuracy_target:
            raise ValueError("model not known: " + model)
        return self.accuracy_target[model]

    def get_performance_sample_count(self, model):
        model = self.get_mlperf_model(model)
        if model not in self.performance_sample_count:
            raise ValueError("model not known: " + model)
        return self.performance_sample_count[model]

    def ignore_errors(self, line):
        for error in self.base["ignore_errors"]:
            if error in line:
                return True
        return False

    def get_min_query_count(self, model, scenario):
        model = self.get_mlperf_model(model)
        if model not in self.min_queries:
            raise ValueError("model not known: " + model)
        return self.min_queries[model].get(scenario)


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--version", default="v0.7", choices=list(MODEL_CONFIG.keys()), help="mlperf version")
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument("--csv", default="summary.csv", help="csv file with results")
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


def check_accuracy_dir(config, model, path):
    is_valid = False
    acc = None
    hash_val = None
    acc_type, acc_target = config.get_accuracy_target(model)
    pattern = ACC_PATTERN[acc_type]
    with open(os.path.join(path, "accuracy.txt"), "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(pattern, line)
            if m:
                acc = m.group(1)
            m = re.match(r"^hash=([\w\d]+)$", line)
            if m:
                hash_val = m.group(1)
            if hash_val and acc:
                break

    if acc and float(acc) >= acc_target:
        is_valid = True
    else:
        log.warning("%s accuracy not met: expected=%f, found=%s", path, acc_target, acc)

    if not hash_val:
        log.error("%s not hash value for mlperf_log_accuracy.json", path)
        is_valid = False

    # check mlperf_log_accuracy.json
    fname = os.path.join(path, "mlperf_log_accuracy.json")
    if not os.path.exists(fname):
        log.error("%s is missing", fname)
        is_valid = False
    else:
        if os.stat(fname).st_size > MAX_ACCURACY_LOG_SIZE:
            log.error("%s is not truncated", fname)
            is_valid = False

    # check if there are any errors in the detailed log
    fname = os.path.join(path, "mlperf_log_detail.txt")
    if not os.path.exists(fname):
        log.error("%s is missing", fname)
        is_valid = False
    else:
        with open(fname, "r") as f:
            for line in f:
                # look for: ERROR
                if "ERROR" in line:
                    if config.ignore_errors(line):
                        continue
                    log.error("%s contains error: %s", fname, line)
                    is_valid = False

    return is_valid, acc


def check_performance_dir(config, model, path):
    is_valid = False
    rt = {}
    # look for: Result is: VALID
    fname = os.path.join(path, "mlperf_log_summary.txt")
    with open(fname, "r") as f:
        for line in f:
            m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
            if m:
                is_valid = True
            m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.]+).*", line)
            if m:
                rt[m.group(1).strip()] = m.group(2).strip()

    performance_sample_count = config.get_performance_sample_count(model)
    if int(rt['performance_sample_count']) < performance_sample_count:
        log.error("%s performance_sample_count, found %d, needs to be > %s",
                  fname, performance_sample_count, rt['performance_sample_count'])
        is_valid = False

    # check if there are any errors in the detailed log
    fname = os.path.join(path, "mlperf_log_detail.txt")
    with open(fname, "r") as f:
        for line in f:
            # look for: ERROR
            if "ERROR" in line:
                if config.ignore_errors(line):
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

    if config.version != "v0.5":
        # not supported for v0.5

        # check if the benchmark meets latency constraint
        target_latency = config.latency_constraint.get(model, dict()).get(scenario)
        if target_latency:
            if int(rt['99.00 percentile latency (ns)']) > target_latency:
                log.error("%s Latency constraint not met, expected=%s, found=%s",
                            fname, target_latency, rt['99.00 percentile latency (ns)'])

        # Check Minimum queries were issued to meet test duration
        min_query_count = config.get_min_query_count(model, scenario)
        if int(rt['min_query_count']) < min_query_count:
            log.error("%s Required minimum Query Count not met by user config, Expected=%s, Found=%s",
                        fname, min_query_count, rt['min_query_count'])
        if scenario == "Offline" and (int(rt['samples_per_query']) < OFFLINE_MIN_SPQ):
            log.error("%s Required minimum samples per query not met by user config, Expected=%s, Found=%s",
                        fname, OFFLINE_MIN_SPQ, rt['samples_per_query'])

        # Test duration of 60s is met
        if int(rt["min_duration (ms)"]) < TEST_DURATION_MS:
            log.error("%s Test duration lesser than 60s in user config. expected=%s, found=%s",
                        fname, TEST_DURATION_MS, rt["min_duration (ms)"])

    return is_valid, res


def files_diff(list1, list2, optional=None):
    """returns a list of files that are missing or added."""
    if not optional:
        optional = []
    if list1 and list2:
        for i in ["mlperf_log_trace.json", "results.json"] + optional:
            try:
                list1.remove(i)
            except:
                pass
        if len(list1) > len(list2):
            return list(set(list1) - set(list2))
        else:
            return list(set(list2) - set(list1))
    return []


def check_results_dir(config, filter_submitter, csv):
    """
    Walk the results directory and do the checking.

    We are called with the cdw at the root of the submission directory.
    level1 division - closed|open
    level2 submitter - for example mlperf_org
    level3 - results, systems, measurements, code

    For results the structure from here is:
    results/$system_desc/$benchmark_model/$scenario/performance/run_n
    and
    results/$system_desc/$benchmark_model/$scenario/accuracy

    We first walk into results/$system_desc
        make sure there is a system_desc.json and its good
    Next we walk into the model
        make sure the model is good, make sure all required scenarios are there.
    Next we walk into each scenario
        check the performance directory
        check the accuracy directory
        if all was good, add the result to the results directory
        if there are errors write a None as result so we can report later what failed
    """
    head = [
        "Organization", "Availability", "Division", "SystemType", "Platform", "Model",
        "MlperfModel", "Scenario", "Result", "Accuracy", "Location",
    ]
    fmt = ",".join(["{}"] * len(head)) + "\n"
    csv.write(",".join(head) + "\n")
    results = {}

    # we are at the top of the submission directory
    for division in list_dir("."):
        # we are looking at ./$division, ie ./closed        
        if division not in VALID_DIVISIONS:
            if division != ".git":
                log.error("invalid division in input dir %s", division)
            continue
        is_closed = division == "closed"

        for submitter in list_dir(division):
            # we are looking at ./$division/$submitter, ie ./closed/mlperf_org
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
                log.error("no submission in %s", results_path)
                results[results_path] = None
                continue

            for system_desc in list_dir(results_path):
                # we are looking at ./$division/$submitter/results/$system_desc, ie ./closed/mlperf_org/results/t4-ort

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
                for model_name in list_dir(results_path, system_desc):
                    # we are looking at ./$division/$submitter/results/$system_desc/$model,
                    #   ie ./closed/mlperf_org/results/t4-ort/bert
                    name = os.path.join(results_path, system_desc, model_name)
                    mlperf_model = config.get_mlperf_model(model_name)

                    if is_closed and mlperf_model not in config.models:
                        # for closed division we want the model name to match.
                        # for open division the model_name might be different than the task
                        log.error("%s has a invalid model %s for closed division", name, model_name)
                        results[name] = None
                        continue


                    #
                    # Look at each scenario
                    #
                    required_scenarios = config.get_required(mlperf_model)
                    if required_scenarios is None:
                        log.error("%s has a invalid model %s, system_type=%s", name, mlperf_model, system_type)
                        results[name] = None
                        continue

                    all_scenarios = set(list(required_scenarios) + list(config.get_optional(mlperf_model)))
                    for scenario in list_dir(results_path, system_desc, model_name):
                        # some submissions in v0.5 use lower case scenarios - map them for now
                        scenario_fixed = SCENARIO_MAPPING.get(scenario, scenario)

                        # we are looking at ./$division/$submitter/results/$system_desc/$model/$scenario,
                        #   ie ./closed/mlperf_org/results/t4-ort/bert/Offline
                        name = os.path.join(results_path, system_desc, model_name, scenario)
                        results[name] = None
                        if scenario_fixed not in all_scenarios:
                            log.warning("%s ignoring scenario %s (neither required nor optional)", name, scenario)
                            continue

                        # check if measurement_dir is good.
                        measurement_dir = os.path.join(division, submitter, "measurements",
                                                       system_desc, model_name, scenario)
                        if not os.path.exists(measurement_dir):
                            log.error("no measurement_dir for %s", name)
                            results[measurement_dir] = None
                        else:
                            if not check_measurement_dir(measurement_dir, name, system_desc,
                                                         os.path.join(division, submitter), model_name, scenario):
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
                            accuracy_is_valid, acc = check_accuracy_dir(config, mlperf_model, acc_path)
                            if not accuracy_is_valid and not is_closed:
                                log.warning("%s, accuracy not valid but taken for open", acc_path)
                                # TODO: is this correct?
                                accuracy_is_valid = True

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
                            diff = files_diff(list_files(perf_path), REQUIRED_PERF_FILES, OPTIONAL_PERF_FILES)
                            if diff:
                                log.error("%s has file list mismatch (%s)", perf_path, diff)
                            try:
                                is_valid, r = check_performance_dir(config, mlperf_model, perf_path)
                            except:
                                is_valid, r = False, None
                            if is_valid:
                                results[name] = r
                                required_scenarios.discard(scenario_fixed)
                            else:
                                log.error("%s has issues", perf_path)

                        if results.get(name):
                            if accuracy_is_valid:
                                log.info("%s is OK", name)
                                csv.write(fmt.format(submitter, available, division, system_type, system_desc, model_name,
                                                     mlperf_model, scenario_fixed, r, acc, name))
                            else:
                                results[name] = None
                                log.error("%s is OK but accuracy has issues", name)

                    if required_scenarios:
                        name = os.path.join(results_path, system_desc, model_name)
                        if is_closed:
                            results[name] = None
                            log.error("%s does not have all required scenarios, missing %s", name, required_scenarios)
                        else:
                            log.warning("%s ignorning missing scenarios in open division (%s)", name, required_scenarios)

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
            log.error("%s is missing code_dir %s", fname, code_dir)
    else:
        log.error("%s is missing %s*.json", fname, system_desc)

    return is_valid


def main():
    args = get_args()

    config = Config(args.version)

    with open(args.csv, "w") as csv:
        os.chdir(args.input)
        # check results directory
        results = check_results_dir(config, args.submitter, csv)

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
