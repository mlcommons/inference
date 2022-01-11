"""
A checker for mlperf inference submissions
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import datetime
import json
import logging
import os
import re
import sys

from log_parser import MLPerfLog

# pylint: disable=missing-docstring


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

submission_checker_dir = os.path.dirname(os.path.realpath(__file__))

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
            "resnet": ["Offline"],
            "ssd-large": ["Offline"],
            "rnnt": ["Offline"],
            "bert-99": ["Offline"],
            "bert-99.9": ["Offline"],
            "dlrm-99": ["Offline"],
            "dlrm-99.9": ["Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
        },
        "optional-scenarios-datacenter": {
            "resnet": ["Server"],
            "ssd-large": ["Server"],
            "rnnt": ["Server"],
            "bert-99": ["Server"],
            "bert-99.9": ["Server"],
            "dlrm-99": ["Server"],
            "dlrm-99.9": ["Server"],
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
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream":1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream":1024, "Offline": 1},
        },
    },
    "v1.0": {
        "models": [
            "ssd-small", "ssd-large", "resnet", "rnnt",
            "bert-99", "bert-99.9",
            "dlrm-99", "dlrm-99.9",
            "3d-unet-99", "3d-unet-99.9",
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Offline"],
            "ssd-large": ["Offline"],
            "rnnt": ["Offline"],
            "bert-99": ["Offline"],
            "bert-99.9": ["Offline"],
            "dlrm-99": ["Offline"],
            "dlrm-99.9": ["Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
        },
        "optional-scenarios-datacenter": {
            "resnet": ["Server"],
            "ssd-large": ["Server"],
            "rnnt": ["Server"],
            "bert-99": ["Server"],
            "bert-99.9": ["Server"],
            "dlrm-99": ["Server"],
            "dlrm-99.9": ["Server"],
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
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline"],
            "ssd-small": ["SingleStream", "Offline"],
            "ssd-large": ["SingleStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["Offline"],
            "dlrm-99": ["Offline"],
            "dlrm-99.9": ["Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {
            "resnet": ["MultiStream", "Server"],
            "ssd-small": ["MultiStream"],
            "ssd-large": ["MultiStream", "Server"],
            "rnnt": ["Server"],
            "bert-99": ["Server"],
            "bert-99.9": ["Server"],
            "dlrm-99": ["Server"],
            "dlrm-99.9": ["Server"],
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
            "qsl_rng_seed": 7322528924094909334,
            "sample_index_rng_seed": 1570999273408051088,
            "schedule_rng_seed": 3507442325620259414,
        },
        "ignore_errors": [
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
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream":1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream":1024, "Offline": 1},
        },
    },
    "v1.1": {
        "models": [
            "ssd-small", "ssd-large", "resnet", "rnnt",
            "bert-99", "bert-99.9",
            "dlrm-99", "dlrm-99.9",
            "3d-unet-99", "3d-unet-99.9",
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Offline"],
            "ssd-large": ["Offline"],
            "rnnt": ["Offline"],
            "bert-99": ["Offline"],
            "bert-99.9": ["Offline"],
            "dlrm-99": ["Offline"],
            "dlrm-99.9": ["Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
        },
        "optional-scenarios-datacenter": {
            "resnet": ["Server"],
            "ssd-large": ["Server"],
            "rnnt": ["Server"],
            "bert-99": ["Server"],
            "bert-99.9": ["Server"],
            "dlrm-99": ["Server"],
            "dlrm-99.9": ["Server"],
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
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline"],
            "ssd-small": ["SingleStream", "Offline"],
            "ssd-large": ["SingleStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["Offline"],
            "dlrm-99": ["Offline"],
            "dlrm-99.9": ["Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {
            "resnet": ["Server"],
            "ssd-large": ["Server"],
            "rnnt": ["Server"],
            "bert-99": ["Server"],
            "bert-99.9": ["Server"],
            "dlrm-99": ["Server"],
            "dlrm-99.9": ["Server"],
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
            "qsl_rng_seed": 1624344308455410291,
            "sample_index_rng_seed": 517984244576520566,
            "schedule_rng_seed": 10051496985653635065,
        },
        "ignore_errors": [
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
            "resnet": {"SingleStream":1024, "Server": 270336, "Offline": 1},
            "ssd-small": {"SingleStream":1024, "Offline": 1},
            "ssd-large": {"SingleStream":1024, "Server": 270336, "Offline": 1},
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream":1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream":1024, "Offline": 1},
        },
    },
    "v2.0": {
        "models": [
            "ssd-small", "ssd-large", "resnet", "rnnt",
            "bert-99", "bert-99.9",
            "dlrm-99", "dlrm-99.9",
            "3d-unet-99", "3d-unet-99.9",
        ],
        # FIXME: required/optional scenarios for v2.0 needs to be filled up correctly; below lists are temporary
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
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "ssd-small": ["SingleStream", "MultiStream", "Offline"],
            "ssd-large": ["SingleStream", "MultiStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-edge": {
        },
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline"],
            "ssd-small": ["SingleStream", "Offline"],
            "ssd-large": ["SingleStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "bert-99.9": ["Offline"],
            "dlrm-99": ["Offline"],
            "dlrm-99.9": ["Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {
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
            "3d-unet-99": ("DICE", 0.86331 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86331 * 0.999),
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
            "3d-unet-99": 42,
            "3d-unet-99.9": 42,
        },
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-mobilenet": "ssd-small",
            "ssd-resnet34": "ssd-large",
            "mobilenet": "resnet",
            "resnet50": "resnet",
        },
        "seeds": {
            "qsl_rng_seed": 1624344308455410291,
            "sample_index_rng_seed": 517984244576520566,
            "schedule_rng_seed": 10051496985653635065,
        },
        "ignore_errors": [
        ],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "ssd-large": {"Server": 100000000},
            "rnnt": {"Server": 1000000000},
            "bert-99": {"Server": 130000000},
            "bert-99.9": {"Server": 130000000},
            "dlrm-99": {"Server": 30000000},
            "dlrm-99.9": {"Server": 30000000},
        },
        "min-queries": {
            "resnet": {"SingleStream":1024, "MultiStream": 270336, "Server": 270336, "Offline": 1},
            "ssd-small": {"SingleStream":1024, "MultiStream": 270336, "Offline": 1},
            "ssd-large": {"SingleStream":1024, "MultiStream": 270336, "Server": 270336, "Offline": 1},
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream":1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream":1024, "Offline": 1},
        },
    },
}

VALID_DIVISIONS = ["open", "closed"]
VALID_AVAILABILITIES = ["available", "preview", "rdi"]
REQUIRED_PERF_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
OPTIONAL_PERF_FILES = ["mlperf_log_accuracy.json"]
REQUIRED_PERF_POWER_FILES = ["spl.txt"]
REQUIRED_POWER_FILES = ["client.json", "client.log", "ptd_logs.txt", "server.json", "server.log"]
REQUIRED_ACC_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt", "accuracy.txt", "mlperf_log_accuracy.json"]
REQUIRED_MEASURE_FILES = ["mlperf.conf", "user.conf", "README.md"]
MS_TO_NS = 1000 * 1000
S_TO_MS = 1000
MAX_ACCURACY_LOG_SIZE = 10 * 1024
OFFLINE_MIN_SPQ = 24576
TEST_DURATION_MS_PRE_1_0 = 60000
TEST_DURATION_MS = 600000
REQUIRED_COMP_PER_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_TEST01_ACC_FILES_1 = ["mlperf_log_accuracy.json", "accuracy.txt"]
REQUIRED_TEST01_ACC_FILES = REQUIRED_TEST01_ACC_FILES_1 + ["baseline_accuracy.txt", "compliance_accuracy.txt"]

SCENARIO_MAPPING = {
    "singlestream": "SingleStream",
    "multistream": "MultiStream",
    "server": "Server",
    "offline": "Offline",
}

RESULT_FIELD = {
    "Offline": "Samples per second",
    "SingleStream": "90th percentile latency (ns)",
    "MultiStream": "Samples per query",
    "Server": "Scheduled samples per second"
}

RESULT_FIELD_NEW = {
    "Offline": "result_samples_per_second",
    "SingleStream": "result_90.00_percentile_latency_ns",
    "MultiStreamLegacy": "effective_samples_per_query",
    "MultiStream": "result_99.00_percentile_per_query_latency_ns",
    "Server": "result_scheduled_samples_per_sec"
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

SYSTEM_DESC_REQUIED_FIELDS_SINCE_V1 = [
    "system_type", "other_software_stack", "host_processor_frequency", "host_processor_caches",
    "host_memory_configuration", "host_processor_interconnect", "host_networking", "host_networking_topology",
    "accelerator_frequency", "accelerator_host_interconnect", "accelerator_interconnect",
    "accelerator_interconnect_topology", "accelerator_memory_configuration",
    "accelerator_on-chip_memories", "cooling", "hw_notes", "sw_notes"
]

SYSTEM_DESC_REQUIED_FIELDS_POWER = [
    "power_management", "filesystem", "boot_firmware_version", "management_firmware_version", "other_hardware",
    "number_of_type_nics_installed", "nics_enabled_firmware", "nics_enabled_os", "nics_enabled_connected",
    "network_speed_mbit", "power_supply_quantity_and_rating_watts", "power_supply_details", "disk_drives",
    "disk_controllers"
]


SYSTEM_IMP_REQUIRED_FILES = [
    "input_data_types", "retraining", "starting_weights_filename", "weight_data_types",
    "weight_transformations",
]


class Config():
    """Select config value by mlperf version and submission type."""
    def __init__(self, version, extra_model_benchmark_map, ignore_uncommited=False, more_power_check=False):
        self.base = MODEL_CONFIG.get(version)
        self.set_extra_model_benchmark_map(extra_model_benchmark_map)
        self.version = version
        self.models = self.base["models"]
        self.seeds = self.base["seeds"]
        self.accuracy_target = self.base["accuracy-target"]
        self.performance_sample_count = self.base["performance-sample-count"]
        self.latency_constraint = self.base.get("latency-constraint", {})
        self.min_queries = self.base.get("min-queries", {})
        self.required = None
        self.optional = None
        self.ignore_uncommited = ignore_uncommited
        self.more_power_check = more_power_check

    def set_extra_model_benchmark_map(self, extra_model_benchmark_map):
        if extra_model_benchmark_map:
            for mapping in extra_model_benchmark_map.split(';'):
                model_name, mlperf_model = mapping.split(':')
                self.base['model_mapping'][model_name] = mlperf_model

    def set_type(self, submission_type):
        if submission_type is None and self.version in ["v0.5"]:
            return
        elif submission_type == "datacenter":
            self.required = self.base["required-scenarios-datacenter"]
            self.optional = self.base["optional-scenarios-datacenter"]
        elif submission_type == "edge":
            self.required = self.base["required-scenarios-edge"]
            self.optional = self.base["optional-scenarios-edge"]
        elif submission_type == "datacenter,edge" or submission_type == "edge,datacenter":
            self.required = self.base["required-scenarios-datacenter-edge"]
            self.optional = self.base["optional-scenarios-datacenter-edge"]
        else:
            raise ValueError("invalid system type")

    def get_mlperf_model(self, model):
        # preferred - user is already using the official name
        if model in self.models:
            return model

        # simple mapping, ie resnet50->resnet ?
        mlperf_model = self.base["model_mapping"].get(model)
        if mlperf_model:
            return mlperf_model

        # try to guess
        if "ssdlite" in model or "ssd-inception" in model or "yolo" in model or \
            "ssd-mobilenet" in model or "ssd-resnet50" in model:
            model = "ssd-small"
        elif "mobilenet" in model:
            model = "mobilenet"
        elif "efficientnet" in model:
            model = "resnet"
        elif "rcnn" in model:
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
        if self.ignore_uncommited and "ERROR : Loadgen built with uncommitted changes!" in line:
            return True
        return False

    def get_min_query_count(self, model, scenario):
        model = self.get_mlperf_model(model)
        if model not in self.min_queries:
            raise ValueError("model not known: " + model)
        return self.min_queries[model].get(scenario)

    def has_new_logging_format(self):
        return self.version not in ["v0.5", "v0.7"]

    def uses_legacy_multistream(self):
        return self.version in ["v0.5", "v0.7", "v1.0", "v1.1"]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--version", default="v1.1", choices=list(MODEL_CONFIG.keys()), help="mlperf version")
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument("--csv", default="summary.csv", help="csv file with results")
    parser.add_argument("--skip_compliance", action="store_true", help="Pass this cmdline option to skip checking compliance/ dir")
    parser.add_argument("--extra-model-benchmark-map", help="extra model name to benchmark mapping")
    parser.add_argument("--debug", action="store_true", help="extra debug output")
    parser.add_argument("--submission-exceptions", action="store_true", help="ignore certain errors for submission")
    parser.add_argument("--more-power-check", action="store_true", help="apply Power WG's check.py script on each power submission. Requires Python 3.7+")
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


def find_error_in_detail_log(config, fname):
    is_valid = True
    if not os.path.exists(fname):
        log.error("%s is missing", fname)
        is_valid = False
    else:
        if config.has_new_logging_format():
            mlperf_log = MLPerfLog(fname)
            if mlperf_log.has_error():
                if config.ignore_uncommited:
                    has_other_errors = False
                    for error in mlperf_log.get_errors():
                        if "Loadgen built with uncommitted changes!" not in error["value"]:
                            has_other_errors = True

                log.error("%s contains errors:", fname)
                for error in mlperf_log.get_errors():
                    log.error("%s", error["value"])

                if not config.ignore_uncommited or has_other_errors:
                    is_valid = False
        else:
            with open(fname, "r") as f:
                for line in f:
                    # look for: ERROR
                    if "ERROR" in line:
                        if config.ignore_errors(line):
                            if "ERROR : Loadgen built with uncommitted changes!" in line:
                                log.warning("%s contains error: %s", fname, line)
                            continue
                        log.error("%s contains error: %s", fname, line)
                        is_valid = False
    return is_valid


def check_accuracy_dir(config, model, path, verbose):
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
    elif verbose:
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
    if not find_error_in_detail_log(config, fname):
        is_valid = False

    return is_valid, acc


def check_performance_dir(config, model, path, scenario_fixed):
    is_valid = False
    rt = {}

    # look for: Result is: VALID
    if config.has_new_logging_format():
        fname = os.path.join(path, "mlperf_log_detail.txt")
        mlperf_log = MLPerfLog(fname)
        if "result_validity" in mlperf_log.get_keys() and mlperf_log["result_validity"] == "VALID":
            is_valid = True
        performance_sample_count = mlperf_log["effective_performance_sample_count"]
        qsl_rng_seed = mlperf_log["effective_qsl_rng_seed"]
        sample_index_rng_seed = mlperf_log["effective_sample_index_rng_seed"]
        schedule_rng_seed = mlperf_log["effective_schedule_rng_seed"]
        scenario = mlperf_log["effective_scenario"]
        scenario_for_res = "MultiStreamLegacy" if scenario == "MultiStream" and config.uses_legacy_multistream() else\
                           scenario
        res = float(mlperf_log[RESULT_FIELD_NEW[scenario_for_res]])
        latency_99_percentile = mlperf_log["result_99.00_percentile_latency_ns"]
        latency_mean = mlperf_log["result_mean_latency_ns"]
        if scenario in ["MultiStream"]:
            latency_99_percentile = mlperf_log["result_99.00_percentile_per_query_latency_ns"]
            latency_mean = mlperf_log["result_mean_query_latency_ns"]
        min_query_count = mlperf_log["effective_min_query_count"]
        samples_per_query = mlperf_log["effective_samples_per_query"]
        min_duration = mlperf_log["effective_min_duration_ms"]
        if scenario == "SingleStream":
            # qps_wo_loadgen_overhead is only used for inferring Offline from SingleStream; only for old submissions
            qps_wo_loadgen_overhead = mlperf_log["result_qps_without_loadgen_overhead"]
    else:
        fname = os.path.join(path, "mlperf_log_summary.txt")
        with open(fname, "r") as f:
            for line in f:
                m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
                if m:
                    is_valid = True
                m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.][\w\+\.\s]*)", line)
                if m:
                    rt[m.group(1).strip()] = m.group(2).strip()
        performance_sample_count = int(rt['performance_sample_count'])
        qsl_rng_seed = int(rt["qsl_rng_seed"])
        sample_index_rng_seed = int(rt["sample_index_rng_seed"])
        schedule_rng_seed = int(rt["schedule_rng_seed"])
        scenario = rt["Scenario"].replace(" ","")
        res = float(rt[RESULT_FIELD[scenario]])
        latency_99_percentile = int(rt['99.00 percentile latency (ns)'])
        latency_mean = int(rt['Mean latency (ns)'])
        min_query_count = int(rt['min_query_count'])
        samples_per_query = int(rt['samples_per_query'])
        min_duration = int(rt["min_duration (ms)"])
        if scenario == "SingleStream":
            qps_wo_loadgen_overhead = float(rt["QPS w/o loadgen overhead"])

    # check if there are any errors in the detailed log
    fname = os.path.join(path, "mlperf_log_detail.txt")
    if not find_error_in_detail_log(config, fname):
        is_valid = False

    required_performance_sample_count = config.get_performance_sample_count(model)
    if performance_sample_count < required_performance_sample_count:
        log.error("%s performance_sample_count, found %d, needs to be >= %d",
                  fname, performance_sample_count, required_performance_sample_count)
        is_valid = False

    if qsl_rng_seed != config.seeds["qsl_rng_seed"]:
        log.error("%s qsl_rng_seed is wrong, expected=%s, found=%s", fname, config.seeds["qsl_rng_seed"], qsl_rng_seed)
    if sample_index_rng_seed != config.seeds["sample_index_rng_seed"]:
        log.error("%s sample_index_rng_seed is wrong, expected=%s, found=%s", fname, config.seeds["sample_index_rng_seed"], sample_index_rng_seed)
    if schedule_rng_seed != config.seeds["schedule_rng_seed"]:
        log.error("%s schedule_rng_seed is wrong, expected=%s, found=%s", fname, config.seeds["schedule_rng_seed"], schedule_rng_seed)

    if scenario == "SingleStream" or (scenario == "MultiStream" and not config.uses_legacy_multistream()):
        res /= MS_TO_NS

    if config.version != "v0.5":
        # FIXME: for open we script this because open can submit in all scenarios
        # not supported for v0.5
        # check if the benchmark meets latency constraint
        target_latency = config.latency_constraint.get(model, dict()).get(scenario)
        if target_latency:
            if latency_99_percentile > target_latency:
                log.error("%s Latency constraint not met, expected=%s, found=%s",
                            fname, target_latency, latency_99_percentile)

        # Check Minimum queries were issued to meet test duration
        required_min_query_count = config.get_min_query_count(model, scenario)
        if required_min_query_count and min_query_count < required_min_query_count:
            log.error("%s Required minimum Query Count not met by user config, Expected=%s, Found=%s",
                        fname, required_min_query_count, min_query_count)
        if scenario == "Offline" and (samples_per_query < OFFLINE_MIN_SPQ):
            log.error("%s Required minimum samples per query not met by user config, Expected=%s, Found=%s",
                        fname, OFFLINE_MIN_SPQ, samples_per_query)

        # Test duration of 600s is met
        required_min_duration = TEST_DURATION_MS_PRE_1_0 if config.version in ["v0.5", "v0.7"] else TEST_DURATION_MS
        if min_duration < required_min_duration:
            log.error("%s Test duration lesser than 600s in user config. expected=%s, found=%s",
                        fname, required_min_duration, min_duration)

    inferred = False
    # special case for results inferred from different scenario
    if scenario_fixed in ["Offline"] and scenario in ["SingleStream"]:
        inferred = True
        res = qps_wo_loadgen_overhead
    
    if (scenario_fixed in ["Offline"] and not config.uses_legacy_multistream()) and scenario in ["MultiStream"]:
        inferred = True
        res = samples_per_query * S_TO_MS / (latency_mean / MS_TO_NS)

    if (scenario_fixed in ["MultiStream"] and not config.uses_legacy_multistream()) and scenario in ["SingleStream"]:
        inferred = True
        res = (latency_99_percentile * samples_per_query) / MS_TO_NS

    return is_valid, res, inferred


def check_power_dir(power_path, ranging_path, testing_path, scenario_fixed, more_power_check=False):

    is_valid = True
    power_metric = 0

    # check if all the required files are present
    required_files = REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
    diff = files_diff(list_files(testing_path), required_files, OPTIONAL_PERF_FILES)
    if diff:
        log.error("%s has file list mismatch (%s)", testing_path, diff)
        is_valid = False
    diff = files_diff(list_files(ranging_path), required_files, OPTIONAL_PERF_FILES)
    if diff:
        log.error("%s has file list mismatch (%s)", ranging_path, diff)
        is_valid = False
    diff = files_diff(list_files(power_path), REQUIRED_POWER_FILES)
    if diff:
        log.error("%s has file list mismatch (%s)", power_path, diff)
        is_valid = False

    # parse the power logs
    server_json_fname = os.path.join(power_path, "server.json")
    with open(server_json_fname) as f:
        server_timezone = datetime.timedelta(seconds=json.load(f)["timezone"])
    client_json_fname = os.path.join(power_path, "client.json")
    with open(client_json_fname) as f:
        client_timezone = datetime.timedelta(seconds=json.load(f)["timezone"])
    detail_log_fname = os.path.join(testing_path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(detail_log_fname)
    datetime_format = '%m-%d-%Y %H:%M:%S.%f'
    power_begin = datetime.datetime.strptime(mlperf_log["power_begin"], datetime_format) + client_timezone
    power_end = datetime.datetime.strptime(mlperf_log["power_end"], datetime_format) + client_timezone
    spl_fname = os.path.join(testing_path, "spl.txt")
    power_list = []
    with open(spl_fname) as f:
        for line in f:
            timestamp = datetime.datetime.strptime(line.split(",")[1], datetime_format) + server_timezone
            if timestamp > power_begin and timestamp < power_end:
                power_list.append(float(line.split(",")[3]))
    if len(power_list) == 0:
        log.error("%s has no power samples falling in power range: %s - %s", spl_fname, power_begin, power_end)
        is_valid = False
    else:
        avg_power = sum(power_list) / len(power_list)
        if scenario_fixed in ["Offline", "Server"]:
            # In Offline and Server scenarios, the power metric is in W.
            power_metric = avg_power
        elif scenario_fixed in ["MultiStream"]:
            # In SingleStream and MultiStream scenarios, the power metric is in J/sample.
            power_duration = (power_end - power_begin).total_seconds()
            num_samples = mlperf_log["generated_query_count"] * mlperf_log["generated_samples_per_query"]
            power_metric = avg_power * power_duration / num_samples
        elif scenario_fixed in ["SingleStream"]:
            # TODO: Currently, LoadGen does NOT print out the actual number of queries in detail logs. There is a
            # "generated_query_count", but LoadGen exits early when the min_duration has been met, so it is not equal to
            # the actual number of queries. For now, we will make use of "result_qps_with_loadgen_overhead", which is
            # defined as: (sample_count - 1) / pr.final_query_issued_time, where final_query_issued_time can be
            # approximated by power_duration (off by one query worth of latency, which is in general negligible compared
            # to 600-sec total runtime and can be offsetted by removing the "+1" when reconstructing the sample_count).
            # Not an ideal approach, but probably the best we can do in v1.0.
            # As for MultiStream, it always runs for 270336 queries, so using "generated_query_count" as above is fine.
            # In v1.1, we should make LoadGen print out the actual number of queries and use it to compute the J/sample 
            # for SingleStream and MultiStream power metrics.
            power_duration = (power_end - power_begin).total_seconds()
            num_samples = mlperf_log["result_qps_with_loadgen_overhead"] * power_duration
            power_metric = avg_power * power_duration / num_samples

    if more_power_check:
        python_version_major = int(sys.version.split(" ")[0].split(".")[0])
        python_version_minor = int(sys.version.split(" ")[0].split(".")[1])
        assert python_version_major == 3 and python_version_minor >= 7, "The --more-power-check only supports Python 3.7+"
        assert os.path.exists(os.path.join(submission_checker_dir, "power-dev", "compliance", "check.py")), \
            "Please run 'git submodule update --init tools/submission/power-dev' to get Power WG's check.py."
        sys.path.insert(0, os.path.join(submission_checker_dir, "power-dev"))
        from compliance.check import check as check_power_more
        perf_path = os.path.dirname(power_path)
        check_power_result = check_power_more(perf_path)
        sys.stdout.flush()
        sys.stderr.flush()
        if check_power_result != 0:
            log.error("Power WG check.py did not pass for: %s", perf_path)
            is_valid = False

    return is_valid, power_metric



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


def check_results_dir(config, filter_submitter,  skip_compliance, csv, debug=False):
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
        "Organization", "Availability", "Division", "SystemType", "SystemName", "Platform", "Model",
        "MlperfModel", "Scenario", "Result", "Accuracy", "number_of_nodes", "host_processor_model_name",
        "host_processors_per_node", "host_processor_core_count", "accelerator_model_name", "accelerators_per_node",
        "Location", "framework", "operating_system", "notes", "compliance", "errors", "version", "inferred", "power"
    ]
    fmt = ",".join(["{}"] * len(head)) + "\n"
    csv.write(",".join(head) + "\n")
    results = {}

    def log_result(submitter, available, division, system_type, system_name, system_desc, model_name, mlperf_model,
                   scenario_fixed, r, acc, system_json, name, compliance, errors, config, inferred=0, power_metric=0):

        notes = system_json.get("hw_notes", "")
        if system_json.get("sw_notes"):
            notes = notes + ". " + system_json.get("sw_notes")

        csv.write(fmt.format(
            submitter, available, division, system_type, '\"' + system_name + '\"', system_desc, model_name,
            mlperf_model, scenario_fixed, r, acc,
            system_json.get("number_of_nodes"),
            '"'+system_json.get("host_processor_model_name")+ '"',
            system_json.get("host_processors_per_node"),
            system_json.get("host_processor_core_count"),
            '"'+system_json.get("accelerator_model_name")+ '"',
            system_json.get("accelerators_per_node"),
            name.replace("\\", "/"),
            '"'+system_json.get("framework", "")+'"',
            '"'+system_json.get("operating_system", "")+'"',
            '"'+notes +'"',
            compliance,
            errors,
            config.version,
            inferred,
            power_metric))

    # we are at the top of the submission directory
    for division in list_dir("."):
        # we are looking at ./$division, ie ./closed
        if division not in VALID_DIVISIONS:
            if division not in [".git", ".github"]:
                log.error("invalid division in input dir %s", division)
            continue
        is_closed = division == "closed"

        for submitter in list_dir(division):
            # we are looking at ./$division/$submitter, ie ./closed/mlperf_org
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
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
                    available = system_json.get("status").lower()
                    if available not in VALID_AVAILABILITIES:
                        log.error("%s has invalid status (%s)", system_id_json, available)
                        results[name] = None
                        continue
                    system_type = system_json.get("system_type")
                    if config.version not in ["v0.5"]:
                        valid_system_types = ["datacenter", "edge"]
                        if config.version not in ["v0.7"]:
                            valid_system_types += ["datacenter,edge", "edge,datacenter"]
                        if system_type not in valid_system_types:
                            log.error("%s has invalid system type (%s)", system_id_json, system_type)
                            results[name] = None
                            continue
                    config.set_type(system_type)
                    if not check_system_desc_id(name, system_json, submitter, division, config.version):
                        results[name] = None
                        continue

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

                    errors = 0
                    all_scenarios = set(list(required_scenarios) + list(config.get_optional(mlperf_model)))
                    for scenario in list_dir(results_path, system_desc, model_name):
                        # some submissions in v0.5 use lower case scenarios - map them for now
                        scenario_fixed = SCENARIO_MAPPING.get(scenario, scenario)

                        # we are looking at ./$division/$submitter/results/$system_desc/$model/$scenario,
                        #   ie ./closed/mlperf_org/results/t4-ort/bert/Offline
                        name = os.path.join(results_path, system_desc, model_name, scenario)
                        results[name] = None
                        if is_closed and scenario_fixed not in all_scenarios:
                            log.warning("%s ignoring scenario %s (neither required nor optional)", name, scenario)
                            continue

                        # check if measurement_dir is good.
                        measurement_dir = os.path.join(division, submitter, "measurements",
                                                       system_desc, model_name, scenario)
                        if not os.path.exists(measurement_dir):
                            log.error("no measurement_dir for %s", measurement_dir)
                            results[measurement_dir] = None
                            errors += 1
                        else:
                            if not check_measurement_dir(measurement_dir, name, system_desc,
                                                         os.path.join(division, submitter), model_name, scenario):
                                log.error("%s measurement_dir has issues", measurement_dir)
                                # results[measurement_dir] = None
                                errors += 1
                                # FIXME: we should not accept this submission
                                # continue

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
                            accuracy_is_valid, acc = check_accuracy_dir(config, mlperf_model, acc_path, debug or is_closed)
                            if not accuracy_is_valid and not is_closed:
                                if debug:
                                    log.warning("%s, accuracy not valid but taken for open", acc_path)
                                accuracy_is_valid = True
                            if not accuracy_is_valid:
                                # a little below we'll not copy this into the results csv
                                errors += 1
                                log.error("%s, accuracy not valid", acc_path)

                        inferred = 0
                        if scenario in ["Server"] and config.version in ["v0.5", "v0.7"]:
                            n = ["run_1", "run_2", "run_3", "run_4", "run_5"]
                        else:
                            n = ["run_1"]

                        # check if this submission has power logs
                        power_path = os.path.join(name, "performance", "power")
                        has_power = os.path.exists(power_path)
                        if has_power:
                            log.info("Detected power logs for %s", name)

                        for i in n:
                            perf_path = os.path.join(name, "performance", i)
                            if not os.path.exists(perf_path):
                                log.error("%s is missing", perf_path)
                                continue
                            if has_power:
                                required_perf_files = REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
                            else:
                                required_perf_files = REQUIRED_PERF_FILES
                            diff = files_diff(list_files(perf_path), required_perf_files, OPTIONAL_PERF_FILES)
                            if diff:
                                log.error("%s has file list mismatch (%s)", perf_path, diff)

                            try:
                                is_valid, r, is_inferred = check_performance_dir(config, mlperf_model, perf_path, scenario_fixed)
                                if is_inferred:
                                    inferred = 1
                                    log.info("%s has inferred results, qps=%s", perf_path, r)
                            except Exception as e:
                                log.error("%s caused exception in check_performance_dir: %s", perf_path, e)
                                is_valid, r = False, None

                            power_metric = 0
                            if has_power:
                                try:
                                    ranging_path = os.path.join(name, "performance", "ranging")
                                    power_is_valid, power_metric = check_power_dir(power_path, ranging_path, perf_path, scenario_fixed,
                                        more_power_check=config.more_power_check)
                                    if not power_is_valid:
                                        is_valid = False
                                        power_metric = 0
                                except Exception as e:
                                    log.error("%s caused exception in check_power_dir: %s", perf_path, e)
                                    is_valid, r, power_metric = False, None, 0

                            if is_valid:
                                results[name] = r if r is None or power_metric == 0 else "{:f} with power_metric = {:f}".format(r, power_metric)
                                required_scenarios.discard(scenario_fixed)
                            else:
                                log.error("%s has issues", perf_path)
                                errors += 1

                        # check if compliance dir is good for CLOSED division
                        compliance = 0 if is_closed else 1
                        if is_closed and not skip_compliance:
                            compliance_dir = os.path.join(division, submitter, "compliance",
                                                          system_desc, model_name, scenario)
                            if not os.path.exists(compliance_dir):
                                log.error("no compliance dir for %s", name)
                                # results[name] = None
                            else:
                                if not check_compliance_dir(compliance_dir, mlperf_model, scenario):
                                    log.error("compliance dir %s has issues", compliance_dir)
                                    # results[name] = None
                                else:
                                    compliance = 1

                        if results.get(name):
                            if accuracy_is_valid:
                                log_result(submitter, available, division, system_type, system_json.get("system_name"), system_desc, model_name, mlperf_model,
                                           scenario_fixed, r, acc, system_json, name, compliance, errors, config, inferred=inferred, power_metric=power_metric)
                            else:
                                results[name] = None
                                log.error("%s is OK but accuracy has issues", name)

                    if required_scenarios:
                        name = os.path.join(results_path, system_desc, model_name)
                        if is_closed:
                            results[name] = None
                            log.error("%s does not have all required scenarios, missing %s", name, required_scenarios)
                        elif debug:
                            log.warning("%s ignoring missing scenarios in open division (%s)", name, required_scenarios)

    return results


def check_system_desc_id(fname, systems_json, submitter, division, version):
    is_valid = True
    # check all required fields
    if version in ["v0.5", "v0.7"]:
        required_fields = SYSTEM_DESC_REQUIRED_FIELDS
    else:
        required_fields = SYSTEM_DESC_REQUIRED_FIELDS + SYSTEM_DESC_REQUIED_FIELDS_SINCE_V1
    for k in required_fields:
        if k not in systems_json:
            is_valid = False
            log.error("%s, field %s is missing", fname, k)

    if version in ["v0.5", "v0.7"]:
        all_fields = required_fields + SYSTEM_DESC_REQUIED_FIELDS_SINCE_V1
    else:
        # TODO: SYSTEM_DESC_REQUIED_FIELDS_POWER should be mandatory when a submission has power logs, but since we
        # check power submission in check_results_dir, the information is not available yet at this stage.
        all_fields = required_fields + SYSTEM_DESC_REQUIED_FIELDS_POWER
    for k in systems_json.keys():
        if k not in all_fields:
            log.warning("%s, field %s is unknown", fname, k)

    if systems_json.get("submitter").lower() != submitter.lower():
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
        code_dir = os.path.join(root, "code", model)
        if os.path.isfile(code_dir):
            with open(code_dir, "r") as f:
                line = f.read()
                code_dir = os.path.join(root, "code", line.strip(), impl)
        else:
            code_dir = os.path.join(root, "code", model, impl)

        if not os.path.exists(code_dir):
            # see if the code dir is per model
            if not os.path.exists(os.path.dirname(code_dir)):
                log.error("%s is missing code_dir %s", fname, code_dir)
                is_valid = False
    else:
        log.error("%s is missing %s*.json", fname, system_desc)
        is_valid = False

    return is_valid

def check_compliance_perf_dir(test_dir, require_verify_perf=True):
    is_valid = False

    fname = os.path.join(test_dir, "verify_performance.txt")
    if require_verify_perf and not os.path.exists(fname):
        log.error("%s is missing in %s", fname, test_dir)
        is_valid = False
    else:
        if require_verify_perf:
            with open(fname, "r") as f:
                for line in f:
                    # look for: TEST PASS
                    if "TEST PASS" in line:
                        is_valid = True
                        break
        # Check performance dir
        test_perf_path = os.path.join(test_dir, "performance", "run_1")
        if not os.path.exists(test_perf_path):
            log.error("%s has no performance/run_1 directory", test_dir)
            is_valid = False
        else:
            diff = files_diff(list_files(test_perf_path), REQUIRED_COMP_PER_FILES)
            if diff:
                log.info("%s has file list mismatch (%s)", test_perf_path, diff)

    return is_valid

def check_compliance_acc_dir(test_dir):
    is_valid = False
    acc_passed = False

    fname = os.path.join(test_dir, "verify_accuracy.txt")
    if not os.path.exists(fname):
        log.error("%s is missing in %s", fname, test_dir)
        is_valid = False
    else:
        # Accuracy can fail for TEST01
        is_valid = True
        with open(fname, "r") as f:
            for line in f:
                # look for: TEST PASS
                if "TEST PASS" in line:
                    acc_passed = True
                    break
        # Check Accuracy dir
        test_acc_path = os.path.join(test_dir, "accuracy")
        if not os.path.exists(test_acc_path):
            log.error("%s has no accuracy directory", test_dir)
            is_valid = False
        else:
            diff = files_diff(list_files(test_acc_path), REQUIRED_TEST01_ACC_FILES_1 if acc_passed else REQUIRED_TEST01_ACC_FILES)
            if diff:
                log.info("%s has file list mismatch (%s)", test_acc_path, diff)

    return is_valid

def check_compliance_dir(compliance_dir, model, scenario):
    compliance_perf_pass = True
    compliance_acc_pass = True
    test_list = ["TEST01", "TEST04-A", "TEST04-B", "TEST05"]

    if model in ["rnnt", "bert-99", "bert-99.9", "dlrm-99", "dlrm-99.9"] or scenario in ["MultiStream"]:
       test_list.remove("TEST04-A")
       test_list.remove("TEST04-B")

    #Check performance of all Tests
    for test in test_list:
        test_dir = os.path.join(compliance_dir, test)
        if not os.path.exists(test_dir):
            log.error("Missing %s in compliance dir %s", test, compliance_dir)
            compliance_perf_pass = False
        else:
            compliance_perf_pass = check_compliance_perf_dir(test_dir, test != "TEST04-B")



    #Check accuracy for TEST01
    compliance_acc_pass = check_compliance_acc_dir(os.path.join(compliance_dir, "TEST01"))

    return compliance_perf_pass and compliance_acc_pass

def main():
    args = get_args()

    config = Config(args.version, args.extra_model_benchmark_map, ignore_uncommited=args.submission_exceptions,
        more_power_check=args.more_power_check)

    with open(args.csv, "w") as csv:
        os.chdir(args.input)
        # check results directory
        results = check_results_dir(config, args.submitter, args.skip_compliance, csv, args.debug)

    # log results
    log.info("---")
    with_results = 0
    for k, v in sorted(results.items()):
        if v:
            log.info("Results %s %s", k, v)
            with_results += 1
    log.info("---")
    for k, v in sorted(results.items()):
        if v is None:
            log.error("NoResults %s", k)

    # print summary
    log.info("---")
    log.info("Results=%d, NoResults=%d", with_results, len(results) - with_results)
    if len(results) != with_results:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
