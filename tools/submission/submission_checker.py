"""A checker for mlperf inference submissions
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
from glob import glob

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
        "test05_seeds": {
            "qsl_rng_seed": 195,
            "sample_index_rng_seed": 235,
            "schedule_rng_seed": 634,
        },
        "ignore_errors": [
            "check for ERROR in detailed",
            "Loadgen built with uncommitted changes",
            "Ran out of generated queries to issue before the minimum query "
            "count and test duration were reached",
            "CAS failed",
        ],
    },
    "v0.7": {
        "models": [
            "ssd-small",
            "ssd-large",
            "resnet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-99",
            "dlrm-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
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
        "test05_seeds": {
            "qsl_rng_seed": 313588358309856706,
            "sample_index_rng_seed": 471397156132239067,
            "schedule_rng_seed": 413914573387865862,
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
            "resnet": {
                "SingleStream": 1024,
                "Server": 270336,
                "MultiStream": 270336,
                "Offline": 1,
            },
            "ssd-small": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
            "ssd-large": {
                "SingleStream": 1024,
                "Server": 270336,
                "MultiStream": 270336,
                "Offline": 1,
            },
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
        },
    },
    "v1.0": {
        "models": [
            "ssd-small",
            "ssd-large",
            "resnet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-99",
            "dlrm-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
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
        "test05_seeds": {
            "qsl_rng_seed": 313588358309856706,
            "sample_index_rng_seed": 471397156132239067,
            "schedule_rng_seed": 413914573387865862,
        },
        "ignore_errors": [],
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
            "resnet": {
                "SingleStream": 1024,
                "Server": 270336,
                "MultiStream": 270336,
                "Offline": 1,
            },
            "ssd-small": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
            "ssd-large": {
                "SingleStream": 1024,
                "Server": 270336,
                "MultiStream": 270336,
                "Offline": 1,
            },
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
        },
    },
    "v1.1": {
        "models": [
            "ssd-small",
            "ssd-large",
            "resnet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-99",
            "dlrm-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
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
        "test05_seeds": {
            "qsl_rng_seed": 313588358309856706,
            "sample_index_rng_seed": 471397156132239067,
            "schedule_rng_seed": 413914573387865862,
        },
        "ignore_errors": [],
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
            "resnet": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "ssd-small": {"SingleStream": 1024, "Offline": 1},
            "ssd-large": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
        },
    },
    "v2.0": {
        "models": [
            "ssd-small",
            "ssd-large",
            "resnet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-99",
            "dlrm-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
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
        "optional-scenarios-datacenter": {},
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "ssd-small": ["SingleStream", "MultiStream", "Offline"],
            "ssd-large": ["SingleStream", "MultiStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "ssd-small": ["SingleStream", "Offline", "MultiStream"],
            "ssd-large": ["SingleStream", "Offline", "MultiStream", "Server"],
            "rnnt": ["SingleStream", "Offline", "Server"],
            "bert-99": ["SingleStream", "Offline", "Server"],
            "bert-99.9": ["Offline", "Server"],
            "dlrm-99": ["Offline", "Server"],
            "dlrm-99.9": ["Offline", "Server"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {},
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
            "ssd_resnet101_v1_fpn_640x640": "ssd-small",
            "ssd_resnet101_v1_fpn_1024x1024": "ssd-large",
            "ssd_resnet152_v1_fpn_640x640": "ssd-small",
            "ssd_resnet152_v1_fpn_1024x1024": "ssd-large",
            "rcnn-resnet50-lowproposals-coco": "ssd-large",
            "rcnn-inception-resnet-v2-lowproposals-coco": "ssd-large",
            "rcnn-inception-v2-coco": "ssd-large",
            "rcnn-nas-lowproposals-coco": "ssd-large",
            "rcnn-resnet101-lowproposals-coco": "ssd-large",
            "ssd_mobilenet_v1_coco": "ssd-small",
            "ssd_mobilenet_v1_fpn_640x640": "ssd-small",
            "ssd_mobilenet_v1_quantized_coco": "ssd-small",
            "ssd_mobilenet_v2_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_1024x1024": "ssd-large",
        },
        "seeds": {
            "qsl_rng_seed": 6655344265603136530,
            "sample_index_rng_seed": 15863379492028895792,
            "schedule_rng_seed": 12662793979680847247,
        },
        "test05_seeds": {
            "qsl_rng_seed": 313588358309856706,
            "sample_index_rng_seed": 471397156132239067,
            "schedule_rng_seed": 413914573387865862,
        },
        "ignore_errors": [],
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
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "ssd-small": {"SingleStream": 1024, "MultiStream": 270336, "Offline": 1},
            "ssd-large": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
        },
    },
    "v2.1": {
        "models": [
            "resnet",
            "retinanet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-99",
            "dlrm-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Server", "Offline"],
            "retinanet": ["Server", "Offline"],
            "rnnt": ["Server", "Offline"],
            "bert-99": ["Server", "Offline"],
            "bert-99.9": ["Server", "Offline"],
            "dlrm-99": ["Server", "Offline"],
            "dlrm-99.9": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
        },
        "optional-scenarios-datacenter": {},
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "retinanet": ["SingleStream", "MultiStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "retinanet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "rnnt": ["SingleStream", "Offline", "Server"],
            "bert-99": ["SingleStream", "Offline", "Server"],
            "bert-99.9": ["Offline", "Server"],
            "dlrm-99": ["Offline", "Server"],
            "dlrm-99.9": ["Offline", "Server"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {},
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "retinanet": ("mAP", 37.55 * 0.99),
            "rnnt": ("WER", (100 - 7.452) * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-99": ("AUC", 80.25 * 0.99),
            "dlrm-99.9": ("AUC", 80.25 * 0.999),
            "3d-unet-99": ("DICE", 0.86170 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86170 * 0.999),
        },
        "performance-sample-count": {
            "resnet": 1024,
            # TODO: Update perf sample count for retinanet
            "retinanet": 64,
            "rnnt": 2513,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-99": 204800,
            "dlrm-99.9": 204800,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
        },
        # TODO: Update this list.
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-mobilenet": "ssd-small",
            "ssd-resnet34": "retinanet",
            "mobilenet": "resnet",
            "resnet50": "resnet",
            "ssd_resnet101_v1_fpn_640x640": "ssd-small",
            "ssd_resnet101_v1_fpn_1024x1024": "ssd-large",
            "ssd_resnet152_v1_fpn_640x640": "ssd-small",
            "ssd_resnet152_v1_fpn_1024x1024": "ssd-large",
            "rcnn-resnet50-lowproposals-coco": "ssd-large",
            "rcnn-inception-resnet-v2-lowproposals-coco": "ssd-large",
            "rcnn-inception-v2-coco": "ssd-large",
            "rcnn-nas-lowproposals-coco": "ssd-large",
            "rcnn-resnet101-lowproposals-coco": "ssd-large",
            "ssd_mobilenet_v1_coco": "ssd-small",
            "ssd_mobilenet_v1_fpn_640x640": "ssd-small",
            "ssd_mobilenet_v1_quantized_coco": "ssd-small",
            "ssd_mobilenet_v2_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_1024x1024": "ssd-large",
        },
        "seeds": {
            "qsl_rng_seed": 14284205019438841327,
            "sample_index_rng_seed": 4163916728725999944,
            "schedule_rng_seed": 299063814864929621,
        },
        "test05_seeds": {
            "qsl_rng_seed": 313588358309856706,
            "sample_index_rng_seed": 471397156132239067,
            "schedule_rng_seed": 413914573387865862,
        },
        "ignore_errors": [],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "retinanet": {"Server": 100000000},
            "rnnt": {"Server": 1000000000},
            "bert-99": {"Server": 130000000},
            "bert-99.9": {"Server": 130000000},
            "dlrm-99": {"Server": 30000000},
            "dlrm-99.9": {"Server": 30000000},
        },
        "min-queries": {
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "retinanet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
        },
    },
    "v3.0": {
        "models": [
            "resnet",
            "retinanet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-99",
            "dlrm-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Server", "Offline"],
            "retinanet": ["Server", "Offline"],
            "rnnt": ["Server", "Offline"],
            "bert-99": ["Server", "Offline"],
            "bert-99.9": ["Server", "Offline"],
            "dlrm-99": ["Server", "Offline"],
            "dlrm-99.9": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
        },
        "optional-scenarios-datacenter": {},
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "retinanet": ["SingleStream", "MultiStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "retinanet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "rnnt": ["SingleStream", "Offline", "Server"],
            "bert-99": ["SingleStream", "Offline", "Server"],
            "bert-99.9": ["Offline", "Server"],
            "dlrm-99": ["Offline", "Server"],
            "dlrm-99.9": ["Offline", "Server"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {},
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "retinanet": ("mAP", 37.55 * 0.99),
            "rnnt": ("WER", (100 - 7.452) * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-99": ("AUC", 80.25 * 0.99),
            "dlrm-99.9": ("AUC", 80.25 * 0.999),
            "3d-unet-99": ("DICE", 0.86170 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86170 * 0.999),
        },
        "performance-sample-count": {
            "resnet": 1024,
            # TODO: Update perf sample count for retinanet
            "retinanet": 64,
            "rnnt": 2513,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-99": 204800,
            "dlrm-99.9": 204800,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
        },
        # TODO: Update this list.
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-mobilenet": "ssd-small",
            "ssd-resnet34": "retinanet",
            "mobilenet": "resnet",
            "resnet50": "resnet",
            "ssd_resnet101_v1_fpn_640x640": "ssd-small",
            "ssd_resnet101_v1_fpn_1024x1024": "ssd-large",
            "ssd_resnet152_v1_fpn_640x640": "ssd-small",
            "ssd_resnet152_v1_fpn_1024x1024": "ssd-large",
            "rcnn-resnet50-lowproposals-coco": "ssd-large",
            "rcnn-inception-resnet-v2-lowproposals-coco": "ssd-large",
            "rcnn-inception-v2-coco": "ssd-large",
            "rcnn-nas-lowproposals-coco": "ssd-large",
            "rcnn-resnet101-lowproposals-coco": "ssd-large",
            "ssd_mobilenet_v1_coco": "ssd-small",
            "ssd_mobilenet_v1_fpn_640x640": "ssd-small",
            "ssd_mobilenet_v1_quantized_coco": "ssd-small",
            "ssd_mobilenet_v2_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_1024x1024": "ssd-large",
        },
        "seeds": {
            "qsl_rng_seed": 10003631887983097364,
            "sample_index_rng_seed": 17183018601990103738,
            "schedule_rng_seed": 12134888396634371638,
        },
        "test05_seeds": {
            "qsl_rng_seed": 14646058500348515648,
            "sample_index_rng_seed": 1207248993894122914,
            "schedule_rng_seed": 11879132697760422006,
        },
        "ignore_errors": [],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "retinanet": {"Server": 100000000},
            "rnnt": {"Server": 1000000000},
            "bert-99": {"Server": 130000000},
            "bert-99.9": {"Server": 130000000},
            "dlrm-99": {"Server": 30000000},
            "dlrm-99.9": {"Server": 30000000},
        },
        "min-queries": {
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "retinanet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-99": {"Server": 270336, "Offline": 1},
            "dlrm-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
        },
    },
    "v3.1": {
        "models": [
            "resnet",
            "retinanet",
            "rnnt",
            "bert-99",
            "bert-99.9",
            "dlrm-v2-99",
            "dlrm-v2-99.9",
            "3d-unet-99",
            "3d-unet-99.9",
            "gptj-99",
            "gptj-99.9",
            "gpt3-99",
            "gpt3-99.9"
        ],
        "required-scenarios-datacenter": {
            "resnet": ["Server", "Offline"],
            "retinanet": ["Server", "Offline"],
            "rnnt": ["Server", "Offline"],
            "bert-99": ["Server", "Offline"],
            "bert-99.9": ["Server", "Offline"],
            "dlrm-v2-99": ["Server", "Offline"],
            "dlrm-v2-99.9": ["Server", "Offline"],
            "3d-unet-99": ["Offline"],
            "3d-unet-99.9": ["Offline"],
            "gptj-99": ["Server", "Offline"],
            "gptj-99.9": ["Server", "Offline"],
            "gpt3-99": ["Server", "Offline"],
            "gpt3-99.9": ["Server", "Offline"],
        },
        "optional-scenarios-datacenter": {},
        "required-scenarios-edge": {
            "resnet": ["SingleStream", "MultiStream", "Offline"],
            "retinanet": ["SingleStream", "MultiStream", "Offline"],
            "rnnt": ["SingleStream", "Offline"],
            "bert-99": ["SingleStream", "Offline"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "gptj-99": ["SingleStream", "Offline"],
        },
        "optional-scenarios-edge": {},
        "required-scenarios-datacenter-edge": {
            "resnet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "retinanet": ["SingleStream", "Offline", "MultiStream", "Server"],
            "rnnt": ["SingleStream", "Offline", "Server"],
            "bert-99": ["SingleStream", "Offline", "Server"],
            "bert-99.9": ["Offline", "Server"],
            "dlrm-v2-99": ["Offline", "Server"],
            "dlrm-v2-99.9": ["Offline", "Server"],
            "3d-unet-99": ["SingleStream", "Offline"],
            "3d-unet-99.9": ["SingleStream", "Offline"],
            "gptj-99": ["SingleStream", "Offline", "Server"],
            "gptj-99.9": ["Server", "Offline"],
            "gpt3-99": ["Server", "Offline"],
            "gpt3-99.9": ["Server", "Offline"],
        },
        "optional-scenarios-datacenter-edge": {},
        "accuracy-target": {
            "resnet": ("acc", 76.46 * 0.99),
            "retinanet": ("mAP", 37.55 * 0.99),
            "rnnt": ("WER", (100 - 7.452) * 0.99),
            "bert-99": ("F1", 90.874 * 0.99),
            "bert-99.9": ("F1", 90.874 * 0.999),
            "dlrm-v2-99": ("AUC", 80.31 * 0.99),
            "dlrm-v2-99.9": ("AUC", 80.31 * 0.999),
            "3d-unet-99": ("DICE", 0.86170 * 0.99),
            "3d-unet-99.9": ("DICE", 0.86170 * 0.999),
            "gptj-99" : ("ROUGE1", 42.9865 * 0.99, "ROUGE2", 20.1235 * 0.99, "ROUGEL", 29.9881 * 0.99, "GEN_LEN", 4016878*0.9),
            "gptj-99.9" : ("ROUGE1", 42.9865 * 0.999, "ROUGE2", 20.1235 * 0.999, "ROUGEL", 29.9881 * 0.999, "GEN_LEN", 4016878*0.9),
            # TODO: Replace with metrics for GPT3
            "gpt3-99" : ("ROUGE1", 42.9865 * 0.99, "ROUGE2", 20.1235 * 0.99, "ROUGEL", 29.9881 * 0.99, "GEN_LEN", 4016878*0.9),
            "gpt3-99.9" : ("ROUGE1", 42.9865 * 0.999, "ROUGE2", 20.1235 * 0.999, "ROUGEL", 29.9881 * 0.999, "GEN_LEN", 4016878*0.9),
        },
        "performance-sample-count": {
            "resnet": 1024,
            "retinanet": 64,
            "rnnt": 2513,
            "bert-99": 10833,
            "bert-99.9": 10833,
            "dlrm-v2-99": 204800,
            "dlrm-v2-99.9": 204800,
            "3d-unet-99": 43,
            "3d-unet-99.9": 43,
            "gptj-99": 13368,
            "gptj-99.9": 13368,
            "gpt3-99": 13368,
            "gpt3-99.9": 13368,
        },
        # TODO: Update this list.
        "model_mapping": {
            # map model names to the official mlperf model class
            "ssd-mobilenet": "ssd-small",
            "ssd-resnet34": "retinanet",
            "mobilenet": "resnet",
            "resnet50": "resnet",
            "ssd_resnet101_v1_fpn_640x640": "ssd-small",
            "ssd_resnet101_v1_fpn_1024x1024": "ssd-large",
            "ssd_resnet152_v1_fpn_640x640": "ssd-small",
            "ssd_resnet152_v1_fpn_1024x1024": "ssd-large",
            "rcnn-resnet50-lowproposals-coco": "ssd-large",
            "rcnn-inception-resnet-v2-lowproposals-coco": "ssd-large",
            "rcnn-inception-v2-coco": "ssd-large",
            "rcnn-nas-lowproposals-coco": "ssd-large",
            "rcnn-resnet101-lowproposals-coco": "ssd-large",
            "ssd_mobilenet_v1_coco": "ssd-small",
            "ssd_mobilenet_v1_fpn_640x640": "ssd-small",
            "ssd_mobilenet_v1_quantized_coco": "ssd-small",
            "ssd_mobilenet_v2_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_320x320": "ssd-small",
            "ssd_mobilenet_v2_fpnlite_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_640x640": "ssd-small",
            "ssd_resnet50_v1_fpn_1024x1024": "ssd-large",
        },
        "seeds": {
            # TODO: Update random seeds
            "qsl_rng_seed": 148687905518835231,
            "sample_index_rng_seed": 520418551913322573,
            "schedule_rng_seed": 811580660758947900,
        },
        "test05_seeds": {
            # TODO: Update random seeds
            "qsl_rng_seed": 793197339507417767,
            "sample_index_rng_seed": 255610748586851044,
            "schedule_rng_seed": 352213341366340113,
        },
        "ignore_errors": [],
        "latency-constraint": {
            "resnet": {"Server": 15000000},
            "retinanet": {"Server": 100000000},
            "rnnt": {"Server": 1000000000},
            "bert-99": {"Server": 130000000},
            "bert-99.9": {"Server": 130000000},
            "dlrm-v2-99": {"Server": 60000000},
            "dlrm-v2-99.9": {"Server": 60000000},
            "gptj-99": {"Server": 20000000000},
            "gptj-99.9": {"Server": 20000000000},
            "gpt3-99": {"Server": 20000000000},
            "gpt3-99.9": {"Server": 20000000000},
        },
        "min-queries": {
            "resnet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "retinanet": {
                "SingleStream": 1024,
                "MultiStream": 270336,
                "Server": 270336,
                "Offline": 1,
            },
            "rnnt": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "bert-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "dlrm-v2-99": {"Server": 270336, "Offline": 1},
            "dlrm-v2-99.9": {"Server": 270336, "Offline": 1},
            "3d-unet-99": {"SingleStream": 1024, "Offline": 1},
            "3d-unet-99.9": {"SingleStream": 1024, "Offline": 1},
            "gptj-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "gptj-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "gpt3-99": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
            "gpt3-99.9": {"SingleStream": 1024, "Server": 270336, "Offline": 1},
        },
    },
}

VALID_DIVISIONS = ["open", "closed", "network"]
VALID_AVAILABILITIES = ["available", "preview", "rdi"]
REQUIRED_PERF_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
OPTIONAL_PERF_FILES = ["mlperf_log_accuracy.json"]
REQUIRED_PERF_POWER_FILES = ["spl.txt"]
REQUIRED_POWER_FILES = [
    "client.json",
    "client.log",
    "ptd_logs.txt",
    "server.json",
    "server.log",
]
REQUIRED_ACC_FILES = [
    "mlperf_log_summary.txt",
    "mlperf_log_detail.txt",
    "accuracy.txt",
    "mlperf_log_accuracy.json",
]
REQUIRED_MEASURE_FILES = ["mlperf.conf", "user.conf", "README.md"]
REQUIRED_POWER_MEASURE_FILES = ["analyzer_table.*", "power_settings.*"]
MS_TO_NS = 1000 * 1000
S_TO_MS = 1000
FILE_SIZE_LIMIT_MB = 50
MB_TO_BYTES = 1024 * 1024
MAX_ACCURACY_LOG_SIZE = 10 * 1024
OFFLINE_MIN_SPQ = 24576
TEST_DURATION_MS_PRE_1_0 = 60000
TEST_DURATION_MS = 600000
REQUIRED_COMP_PER_FILES = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
REQUIRED_TEST01_ACC_FILES_1 = ["mlperf_log_accuracy.json", "accuracy.txt"]
REQUIRED_TEST01_ACC_FILES = REQUIRED_TEST01_ACC_FILES_1 + [
    "baseline_accuracy.txt",
    "compliance_accuracy.txt",
]

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
    "Server": "Scheduled samples per second",
}

RESULT_FIELD_NEW = {
    "v0.5": {
        "Offline": "result_samples_per_second",
        "SingleStream": "result_90.00_percentile_latency_ns",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "result_99.00_percentile_per_query_latency_ns",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v0.7": {
        "Offline": "result_samples_per_second",
        "SingleStream": "result_90.00_percentile_latency_ns",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "result_99.00_percentile_per_query_latency_ns",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v1.0": {
        "Offline": "result_samples_per_second",
        "SingleStream": "result_90.00_percentile_latency_ns",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "result_99.00_percentile_per_query_latency_ns",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v1.1": {
        "Offline": "result_samples_per_second",
        "SingleStream": "result_90.00_percentile_latency_ns",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "result_99.00_percentile_per_query_latency_ns",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v2.0": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v2.1": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v3.0": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_scheduled_samples_per_sec",
    },
    "v3.1": {
        "Offline": "result_samples_per_second",
        "SingleStream": "early_stopping_latency_ss",
        "MultiStreamLegacy": "effective_samples_per_query",
        "MultiStream": "early_stopping_latency_ms",
        "Server": "result_scheduled_samples_per_sec",
    }
}

ACC_PATTERN = {
    "acc": r"^accuracy=([\d\.]+).*",
    "AUC": r"^AUC=([\d\.]+).*",
    "mAP": r"^mAP=([\d\.]+).*",
    "bleu": r"^BLEU\:\s*([\d\.]+).*",
    "F1": r"^{[\"\']exact_match[\"\']\:\s*[\d\.]+,\s*[\"\']f1[\"\']\:\s*([\d\.]+)}",
    "WER": r"Word Error Rate\:.*, accuracy=([0-9\.]+)%",
    "DICE": r"Accuracy\:\s*mean\s*=\s*([\d\.]+).*",
    "ROUGE1": r".*'rouge1':\s([\d.]+).*",
    "ROUGE2": r".*'rouge2':\s([\d.]+).*",
    "ROUGEL": r".*'rougeL':\s([\d.]+).*",
    "GEN_LEN": r".*'gen_len':\s([\d.]+).*",
}

SYSTEM_DESC_REQUIRED_FIELDS = [
    "division",
    "submitter",
    "status",
    "system_name",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_storage_capacity",
    "host_storage_type",
    "accelerators_per_node",
    "accelerator_model_name",
    "accelerator_memory_capacity",
    "framework",
    "operating_system",
]

SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS = [
    "division",
    "submitter",
    "system_type",
    "status",
    "system_name",
    "number_of_nodes",
    "host_processor_model_name",
    "host_processors_per_node",
    "host_processor_core_count",
    "host_memory_capacity",
    "host_memory_configuration",
    "host_storage_capacity",
    "host_storage_type",
    "host_networking",
    "host_networking_card_count",
    "host_networking_topology",
    "accelerators_per_node",
    "accelerator_model_name",
    "accelerator_memory_capacity",
    "accelerator_host_interconnect",
    "accelerator_memory_configuration",
    "accelerator_interconnect",
    "cooling",
    "framework",
    "operating_system",
    "other_software_stack",
]

SYSTEM_DESC_REQUIRED_FIELDS_SINCE_V1 = [
    "system_type",
    "other_software_stack",
    "host_processor_frequency",
    "host_processor_caches",
    "host_memory_configuration",
    "host_processor_interconnect",
    "host_networking",
    "host_networking_topology",
    "accelerator_frequency",
    "accelerator_host_interconnect",
    "accelerator_interconnect",
    "accelerator_interconnect_topology",
    "accelerator_memory_configuration",
    "accelerator_on-chip_memories",
    "cooling",
    "hw_notes",
    "sw_notes",
]

SYSTEM_DESC_REQUIRED_FIELDS_SINCE_V3_1 = [
    "host_networking_card_count",
    "system_type_detail"
]

SYSTEM_DESC_REQUIRED_FIELDS_POWER = [
    "power_management",
    "filesystem",
    "boot_firmware_version",
    "management_firmware_version",
    "other_hardware",
    "number_of_type_nics_installed",
    "nics_enabled_firmware",
    "nics_enabled_os",
    "nics_enabled_connected",
    "network_speed_mbit",
    "power_supply_quantity_and_rating_watts",
    "power_supply_details",
    "disk_drives",
    "disk_controllers",
]

SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS_POWER = []

SYSTEM_DESC_IS_NETWORK_MODE = "is_network"
SYSTEM_DESC_REQUIRED_FIELDS_NETWORK_MODE = [
    SYSTEM_DESC_IS_NETWORK_MODE,
    "network_type",
    "network_media",
    "network_rate",
    "nic_loadgen",
    "number_nic_loadgen",
    "net_software_stack_loadgen",
    "network_protocol",
    "number_connections",
    "nic_sut",
    "number_nic_sut",
    "net_software_stack_sut",
    "network_topology",
]
NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME = "Network SUT"

SYSTEM_IMP_REQUIRED_FILES = [
    "input_data_types",
    "retraining",
    "starting_weights_filename",
    "weight_data_types",
    "weight_transformations",
]


class Config:
    """Select config value by mlperf version and submission type."""

    def __init__(
        self,
        version,
        extra_model_benchmark_map,
        ignore_uncommited=False,
        skip_power_check=False,
    ):
        self.base = MODEL_CONFIG.get(version)
        self.extra_model_benchmark_map = extra_model_benchmark_map
        self.version = version
        self.models = self.base["models"]
        self.seeds = self.base["seeds"]
        self.test05_seeds = self.base["test05_seeds"]
        self.accuracy_target = self.base["accuracy-target"]
        self.performance_sample_count = self.base["performance-sample-count"]
        self.latency_constraint = self.base.get("latency-constraint", {})
        self.min_queries = self.base.get("min-queries", {})
        self.required = None
        self.optional = None
        self.ignore_uncommited = ignore_uncommited
        self.skip_power_check = skip_power_check

    def set_type(self, submission_type):
        if submission_type is None and self.version in ["v0.5"]:
            return
        elif submission_type == "datacenter":
            self.required = self.base["required-scenarios-datacenter"]
            self.optional = self.base["optional-scenarios-datacenter"]
        elif submission_type == "edge":
            self.required = self.base["required-scenarios-edge"]
            self.optional = self.base["optional-scenarios-edge"]
        elif (
            submission_type == "datacenter,edge" or submission_type == "edge,datacenter"
        ):
            self.required = self.base["required-scenarios-datacenter-edge"]
            self.optional = self.base["optional-scenarios-datacenter-edge"]
        else:
            raise ValueError("invalid system type")

    def get_mlperf_model(self, model, extra_model_mapping=None):
        # preferred - user is already using the official name
        if model in self.models:
            return model

        # simple mapping, ie resnet50->resnet
        mlperf_model = self.base["model_mapping"].get(model)
        if mlperf_model:
            return mlperf_model

        # Custom mapping provided by the submitter
        if extra_model_mapping is not None:
            mlperf_model = extra_model_mapping.get(model)
            if mlperf_model:
                return mlperf_model

        # try to guess, keep this for backwards compatibility
        # TODO: Generalize this guess or remove it completely?
        if (
            "ssdlite" in model
            or "ssd-inception" in model
            or "yolo" in model
            or "ssd-mobilenet" in model
            or "ssd-resnet50" in model
        ):
            model = "ssd-small"
        elif "mobilenet" in model:
            model = "mobilenet"
        elif "efficientnet" in model or "resnet50" in model:
            model = "resnet"
        elif "rcnn" in model:
            model = "ssd-small"
        elif "bert-99.9" in model:
            model = "bert-99.9"
        elif "bert-99" in model:
            model = "bert-99"
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
        if (
            self.ignore_uncommited
            and ("ERROR : Loadgen built with uncommitted " "changes!") in line
        ):
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

    def uses_early_stopping(self, scenario):
        return (self.version not in ["v0.5", "v0.7", "v1.0", "v1.1"]) and (
            scenario in ["Server", "SingleStream", "MultiStream"]
        )

    def has_query_count_in_log(self):
        return self.version not in ["v0.5", "v0.7", "v1.0", "v1.1"]

    def has_power_utc_timestamps(self):
        return self.version not in ["v0.5", "v0.7", "v1.0"]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument(
        "--version",
        default="v3.1",
        choices=list(MODEL_CONFIG.keys()),
        help="mlperf version",
    )
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument("--csv", default="summary.csv", help="csv file with results")
    parser.add_argument(
        "--skip_compliance",
        action="store_true",
        help="Pass this cmdline option to skip checking compliance/ dir",
    )
    parser.add_argument(
        "--extra-model-benchmark-map",
        help="File containing extra custom model mapping. It is assumed to be inside the folder open/<submitter>",
        default="model_mapping.json",
    )
    parser.add_argument("--debug", action="store_true", help="extra debug output")
    parser.add_argument(
        "--submission-exceptions",
        action="store_true",
        help="ignore certain errors for submission",
    )
    parser.add_argument(
        "--skip-power-check",
        action="store_true",
        help="skips Power WG's check.py script on each power submission.",
    )
    parser.add_argument(
        "--skip-meaningful-fields-emptiness-check",
        action="store_true",
        help="skips the check of empty values in required measurement field values",
    )
    parser.add_argument(
        "--skip-check-power-measure-files",
        action="store_true",
        help="skips the check of required measure files for power runs",
    )
    parser.add_argument(
        "--skip-empty-files-check",
        action="store_true",
        help="skips the check of empty required files",
    )
    parser.add_argument(
        "--skip-extra-files-in-root-check",
        action="store_true",
        help="skips the check of extra files inside the root submission dir",
    )
    args = parser.parse_args()
    return args


def list_dir(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def list_files(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def list_empty_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(path) if not dirs and not files]


def list_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(path)]


def list_files_recursively(*path):
    path = os.path.join(*path)
    return [
        os.path.join(dirpath, file)
        for dirpath, dirs, files in os.walk(path)
        for file in files
    ]


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
                        if (
                            "Loadgen built with uncommitted changes!"
                            not in error["value"]
                        ):
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
                            if (
                                "ERROR : Loadgen built with uncommitted changes!"
                                in line
                            ):
                                log.warning("%s contains error: %s", fname, line)
                            continue
                        log.error("%s contains error: %s", fname, line)
                        is_valid = False
    return is_valid


def check_accuracy_dir(config, model, path, verbose):
    is_valid = False
    all_accuracy_valid = True
    acc = None
    hash_val = None
    target = config.get_accuracy_target(model)
    patterns = []
    acc_targets = []
    for i in range(0, len(target), 2):
        acc_type, acc_target = target[i:i+2]
        patterns.append(ACC_PATTERN[acc_type])
        acc_targets.append(acc_target)
    acc_seen = [False for _ in acc_targets]
    with open(os.path.join(path, "accuracy.txt"), "r", encoding="utf-8") as f:
        for line in f:
            for i, (pattern, acc_target) in enumerate(zip(patterns, acc_targets)):
                m = re.match(pattern, line)
                if m:
                    acc = m.group(1)
                m = re.match(r"^hash=([\w\d]+)$", line)
                if m:
                    hash_val = m.group(1)
                if acc is not None and float(acc) >= acc_target:
                    all_accuracy_valid &= True
                    acc_seen[i] = True
                elif acc is not None:
                    all_accuracy_valid = False
                    log.warning("%s accuracy not met: expected=%f, found=%s", path, acc_target, acc)
            if all(acc_seen) and hash_val:
                break;
        is_valid = all_accuracy_valid & all(acc_seen)

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


def get_performance_metric(
    config, model, path, scenario_fixed, division, system_json, has_power=False
):
    #Assumes new logging format

    fname = os.path.join(path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(fname)
    if (
        "result_validity" in mlperf_log.get_keys()
        and mlperf_log["result_validity"] == "VALID"
    ):
        is_valid = True
    scenario = mlperf_log["effective_scenario"]
    scenario_for_res = (
        "MultiStreamLegacy"
        if scenario == "MultiStream" and config.uses_legacy_multistream()
        else scenario
    )
    res = float(mlperf_log[RESULT_FIELD_NEW[config.version][scenario_for_res]])
    return res

def check_performance_dir(
    config, model, path, scenario_fixed, division, system_json, has_power=False
):
    is_valid = False
    rt = {}

    # look for: Result is: VALID
    if config.has_new_logging_format():
        fname = os.path.join(path, "mlperf_log_detail.txt")
        mlperf_log = MLPerfLog(fname)
        if (
            "result_validity" in mlperf_log.get_keys()
            and mlperf_log["result_validity"] == "VALID"
        ):
            is_valid = True
        performance_sample_count = mlperf_log["effective_performance_sample_count"]
        qsl_rng_seed = mlperf_log["effective_qsl_rng_seed"]
        sample_index_rng_seed = mlperf_log["effective_sample_index_rng_seed"]
        schedule_rng_seed = mlperf_log["effective_schedule_rng_seed"]
        scenario = mlperf_log["effective_scenario"]
        scenario_for_res = (
            "MultiStreamLegacy"
            if scenario == "MultiStream" and config.uses_legacy_multistream()
            else scenario
        )
        res = float(mlperf_log[RESULT_FIELD_NEW[config.version][scenario_for_res]])
        latency_99_percentile = mlperf_log["result_99.00_percentile_latency_ns"]
        latency_mean = mlperf_log["result_mean_latency_ns"]
        if scenario in ["MultiStream"]:
            latency_99_percentile = mlperf_log[
                "result_99.00_percentile_per_query_latency_ns"
            ]
            latency_mean = mlperf_log["result_mean_query_latency_ns"]
        min_query_count = mlperf_log["effective_min_query_count"]
        samples_per_query = mlperf_log["effective_samples_per_query"]
        min_duration = mlperf_log["effective_min_duration_ms"]
        if scenario == "SingleStream":
            # qps_wo_loadgen_overhead is only used for inferring Offline from SingleStream; only for old submissions
            qps_wo_loadgen_overhead = mlperf_log["result_qps_without_loadgen_overhead"]
        sut_name = mlperf_log["sut_name"]
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
        performance_sample_count = int(rt["performance_sample_count"])
        qsl_rng_seed = int(rt["qsl_rng_seed"])
        sample_index_rng_seed = int(rt["sample_index_rng_seed"])
        schedule_rng_seed = int(rt["schedule_rng_seed"])
        scenario = rt["Scenario"].replace(" ", "")
        res = float(rt[RESULT_FIELD[scenario]])
        latency_99_percentile = int(rt["99.00 percentile latency (ns)"])
        latency_mean = int(rt["Mean latency (ns)"])
        min_query_count = int(rt["min_query_count"])
        samples_per_query = int(rt["samples_per_query"])
        min_duration = int(rt["min_duration (ms)"])
        if scenario == "SingleStream":
            qps_wo_loadgen_overhead = float(rt["QPS w/o loadgen overhead"])
        sut_name = str(rt["System Under Test (SUT) name: "])

    # check if there are any errors in the detailed log
    fname = os.path.join(path, "mlperf_log_detail.txt")
    if not find_error_in_detail_log(config, fname):
        is_valid = False

    required_performance_sample_count = config.get_performance_sample_count(model)
    if performance_sample_count < required_performance_sample_count:
        log.error(
            "%s performance_sample_count, found %d, needs to be >= %d",
            fname,
            performance_sample_count,
            required_performance_sample_count,
        )
        is_valid = False

    config_seeds = config.seeds if "TEST05" not in fname else config.test05_seeds
    if qsl_rng_seed != config_seeds["qsl_rng_seed"]:
        log.error(
            "%s qsl_rng_seed is wrong, expected=%s, found=%s",
            fname,
            config_seeds["qsl_rng_seed"],
            qsl_rng_seed,
        )
        is_valid = False
    if sample_index_rng_seed != config_seeds["sample_index_rng_seed"]:
        log.error(
            "%s sample_index_rng_seed is wrong, expected=%s, found=%s",
            fname,
            config_seeds["sample_index_rng_seed"],
            sample_index_rng_seed,
        )
        is_valid = False
    if schedule_rng_seed != config_seeds["schedule_rng_seed"]:
        log.error(
            "%s schedule_rng_seed is wrong, expected=%s, found=%s",
            fname,
            config_seeds["schedule_rng_seed"],
            schedule_rng_seed,
        )
        is_valid = False

    if scenario == "SingleStream" or (
        scenario == "MultiStream" and not config.uses_legacy_multistream()
    ):
        res /= MS_TO_NS

    # Check if current scenario (and version) uses early stopping
    uses_early_stopping = config.uses_early_stopping(scenario)

    if config.version != "v0.5":
        # FIXME: for open we script this because open can submit in all scenarios
        # not supported for v0.5

        if uses_early_stopping:
            # check if early_stopping condition was met
            if not mlperf_log["early_stopping_met"]:
                early_stopping_result = mlperf_log["early_stopping_result"]
                log.error(
                    "Early stopping condition was not met, msg=%s",
                    early_stopping_result,
                )

            # If the scenario has a target latency (Server scenario), check
            # that the target latency that was passed to the early stopping
            # is less than the target latency.
            target_latency = config.latency_constraint.get(model, dict()).get(scenario)
            if target_latency:
                early_stopping_latency_ns = mlperf_log["effective_target_latency_ns"]
                log.info(
                    "Target latency: %s, Early Stopping Latency: %s, Scenario: %s",
                    target_latency,
                    early_stopping_latency_ns,
                    scenario,
                )
                if early_stopping_latency_ns > target_latency:
                    log.error(
                        "%s Latency constraint with early stopping not met, expected=%s, found=%s",
                        fname,
                        target_latency,
                        early_stopping_latency_ns,
                    )
                    is_valid = False

        else:
            # check if the benchmark meets latency constraint
            target_latency = config.latency_constraint.get(model, dict()).get(scenario)
            log.info(
                "Target latency: %s, Latency: %s, Scenario: %s",
                target_latency,
                latency_99_percentile,
                scenario,
            )
            if target_latency:
                if latency_99_percentile > target_latency:
                    log.error(
                        "%s Latency constraint not met, expected=%s, found=%s",
                        fname,
                        target_latency,
                        latency_99_percentile,
                    )

        # Check Minimum queries were issued to meet test duration
        # Check if this run uses early stopping. If it does, get the
        # min_queries from the detail log, otherwise get this value
        # from the config
        if not uses_early_stopping:
            required_min_query_count = config.get_min_query_count(model, scenario)
            if required_min_query_count and min_query_count < required_min_query_count:
                log.error(
                    "%s Required minimum Query Count not met by user config, Expected=%s, Found=%s",
                    fname,
                    required_min_query_count,
                    min_query_count,
                )
                is_valid = False

        if scenario == "Offline" and (samples_per_query < OFFLINE_MIN_SPQ):
            log.error(
                "%s Required minimum samples per query not met by user config, Expected=%s, Found=%s",
                fname,
                OFFLINE_MIN_SPQ,
                samples_per_query,
            )
            is_valid = False

        # Test duration of 600s is met
        required_min_duration = (
            TEST_DURATION_MS_PRE_1_0
            if config.version in ["v0.5", "v0.7"]
            else TEST_DURATION_MS
        )
        if min_duration < required_min_duration:
            log.error(
                "%s Test duration less than 600s in user config. expected=%s, found=%s",
                fname,
                required_min_duration,
                min_duration,
            )

    inferred = False
    # special case for results inferred from different scenario
    if scenario_fixed in ["Offline"] and scenario in ["SingleStream"]:
        inferred = True
        res = qps_wo_loadgen_overhead

    if (
        scenario_fixed in ["Offline"] and not config.uses_legacy_multistream()
    ) and scenario in ["MultiStream"]:
        inferred = True
        res = samples_per_query * S_TO_MS / (latency_mean / MS_TO_NS)

    if (
        scenario_fixed in ["MultiStream"] and not config.uses_legacy_multistream()
    ) and scenario in ["SingleStream"]:
        inferred = True
        # samples_per_query does not match with the one reported in the logs
        # when inferring MultiStream from SingleStream
        samples_per_query = 8
        if uses_early_stopping:
            early_stopping_latency_ms = mlperf_log["early_stopping_latency_ms"]
            if early_stopping_latency_ms == 0:
                log.error(
                    "Not enough samples were processed for early stopping to make an estimate"
                )
                is_valid = False
            res = (early_stopping_latency_ms * samples_per_query) / MS_TO_NS
        else:
            res = (latency_99_percentile * samples_per_query) / MS_TO_NS

    is_network_system, is_network_mode_valid = is_system_over_network(
        division, system_json, path
    )
    is_valid &= is_network_mode_valid
    if is_network_system:
        # for network mode verify the SUT name is valid, accodring to the rules (must include "Network SUT" in name)
        if NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME not in sut_name:
            log.error(
                f"{fname} invalid sut name for network mode. expecting the substring '{NETWORK_MODE_REQUIRED_SUBSTRING_IN_SUT_NAME}' got '{sut_name}'"
            )
            is_valid = False

    return is_valid, res, inferred


def get_power_metric(config, scenario_fixed, log_path, is_valid, res):
    # parse the power logs
    if config.has_power_utc_timestamps():
        server_timezone = datetime.timedelta(0)
        client_timezone = datetime.timedelta(0)
    else:
        server_json_fname = os.path.join(power_path, "server.json")
        with open(server_json_fname) as f:
            server_timezone = datetime.timedelta(seconds=json.load(f)["timezone"])
        client_json_fname = os.path.join(power_path, "client.json")
        with open(client_json_fname) as f:
            client_timezone = datetime.timedelta(seconds=json.load(f)["timezone"])
    detail_log_fname = os.path.join(log_path, "mlperf_log_detail.txt")
    mlperf_log = MLPerfLog(detail_log_fname)
    datetime_format = "%m-%d-%Y %H:%M:%S.%f"
    power_begin = (
        datetime.datetime.strptime(mlperf_log["power_begin"], datetime_format)
        + client_timezone
    )
    power_end = (
        datetime.datetime.strptime(mlperf_log["power_end"], datetime_format)
        + client_timezone
    )
    # Obtain the scenario also from logs to check if power is inferred
    if config.has_new_logging_format():
        scenario = mlperf_log["effective_scenario"]
    else:
        rt = {}
        fname = os.path.join(log_path, "mlperf_log_summary.txt")
        with open(fname, "r") as f:
            for line in f:
                m = re.match(r"^Result\s+is\s*\:\s+VALID", line)
                if m:
                    is_valid = True
                m = re.match(r"^\s*([\w\s.\(\)\/]+)\s*\:\s*([\w\+\.][\w\+\.\s]*)", line)
                if m:
                    rt[m.group(1).strip()] = m.group(2).strip()
        scenario = rt["Scenario"].replace(" ", "")
    spl_fname = os.path.join(log_path, "spl.txt")
    power_list = []
    with open(spl_fname) as f:
        for line in f:
            timestamp = (
                datetime.datetime.strptime(line.split(",")[1], datetime_format)
                + server_timezone
            )
            if timestamp > power_begin and timestamp < power_end:
                value = float(line.split(",")[3])
                if value > 0:
                    power_list.append(float(line.split(",")[3]))

    if len(power_list) == 0:
        log.error(
            "%s has no power samples falling in power range: %s - %s",
            spl_fname,
            power_begin,
            power_end,
        )
        is_valid = False
    else:
        avg_power = sum(power_list) / len(power_list)
        power_duration = (power_end - power_begin).total_seconds()
        if scenario_fixed in ["Offline", "Server"]:
            # In Offline and Server scenarios, the power metric is in W.
            power_metric = avg_power
            avg_power_efficiency = res / avg_power

        else:
            # In SingleStream and MultiStream scenarios, the power metric is in mJ/query.
            assert scenario_fixed in [
                "MultiStream",
                "SingleStream",
            ], "Unknown scenario: {:}".format(scenario_fixed)
            if not config.has_query_count_in_log():
                # Before v2.0, LoadGen does NOT print out the actual number of queries in detail logs. There is a
                # "generated_query_count", but LoadGen exits early when the min_duration has been met, so it is not equal to
                # the actual number of queries. To work around it, make use of "result_qps_with_loadgen_overhead", which is
                # defined as: (sample_count - 1) / pr.final_query_issued_time, where final_query_issued_time can be
                # approximated by power_duration (off by one query worth of latency, which is in general negligible compared
                # to 600-sec total runtime and can be offsetted by removing the "+1" when reconstructing the sample_count).
                # As for MultiStream, it always runs for 270336 queries, so using "generated_query_count" as above is fine.
                if scenario_fixed in ["MultiStream"]:
                    num_queries = (
                        mlperf_log["generated_query_count"]
                        * mlperf_log["generated_samples_per_query"]
                    )
                elif scenario_fixed in ["SingleStream"]:
                    num_queries = (
                        mlperf_log["result_qps_with_loadgen_overhead"] * power_duration
                    )
            else:
                # Starting from v2.0, LoadGen logs the actual number of issued queries.
                num_queries = int(mlperf_log["result_query_count"])
            power_metric = avg_power * power_duration * 1000 / num_queries

            if scenario_fixed in ["SingleStream"]:
                samples_per_query = 1
            elif scenario_fixed in ["MultiStream"]:
                samples_per_query = 8

            if (
                scenario_fixed in ["MultiStream"]
                and not config.uses_legacy_multistream()
            ) and scenario in ["SingleStream"]:
                power_metric = (
                    avg_power * power_duration * samples_per_query * 1000 / num_queries
                )

            avg_power_efficiency = (samples_per_query * 1000) / power_metric

    return is_valid, power_metric, scenario, avg_power_efficiency


def check_power_dir(
    power_path,
    ranging_path,
    testing_path,
    scenario_fixed,
    power_res_ranging,
    power_res_testing,
    config,
):
    skip_power_check = config.skip_power_check

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

    (
        is_valid,
        power_metric_ranging,
        scenario,
        power_efficiency_ranging,
    ) = get_power_metric(
        config, scenario_fixed, ranging_path, is_valid, power_res_ranging
    )
    is_valid, power_metric, scenario, power_efficiency_testing = get_power_metric(
        config, scenario_fixed, testing_path, is_valid, power_res_testing
    )

    if not skip_power_check:
        python_version_major = int(sys.version.split(" ")[0].split(".")[0])
        python_version_minor = int(sys.version.split(" ")[0].split(".")[1])
        assert python_version_major == 3 and python_version_minor >= 7, (
            "Power check " " only " "supports " "Python " "3.7+"
        )
        sys.path.insert(0, os.path.join(submission_checker_dir, "power"))
        from power.power_checker import check as check_power_more

        perf_path = os.path.dirname(power_path)
        check_power_result = check_power_more(perf_path)
        sys.stdout.flush()
        sys.stderr.flush()
        if check_power_result != 0:
            log.error("Power WG power_checker.py did not pass for: %s", perf_path)
            is_valid = False

    return is_valid, power_metric, power_efficiency_testing


def files_diff(list1, list2, optional=None):
    """returns a list of files that are missing or added."""
    if not optional:
        optional = []
    optional = optional + ["mlperf_log_trace.json", "results.json", ".gitkeep"]
    return set(list1).symmetric_difference(set(list2)) - set(optional)


def is_system_over_network(division, system_json, path):
    """
    Verify whether the submitted system is over network and whether it is valid
    for the division
    for 'network' division, it is mandatory that the system is over-network
    for 'closed' division, the system must not be over-network
    for 'open' division, the system may be either local or over-network
    """
    is_network_mode_sys_spec_str = system_json.get(SYSTEM_DESC_IS_NETWORK_MODE)
    is_network_system = (
        is_network_mode_sys_spec_str.lower() == "true"
        if is_network_mode_sys_spec_str is not None
        else False
    )
    # verify that the system corresponds the division
    is_valid = True
    expected_state_by_division = {"network": True, "closed": False}
    if division in expected_state_by_division:
        is_valid = expected_state_by_division[division] is is_network_system
    if not is_valid:
        log.error(
            f"{path} incorrect network mode (={is_network_system}) for division '{division}'"
        )
    return is_network_system, is_valid


def check_results_dir(
    config,
    filter_submitter,
    skip_compliance,
    csv,
    debug=False,
    skip_meaningful_fields_emptiness_check=False,
    skip_empty_files_check=False,
    skip_check_power_measure_files=False,
    skip_extra_files_in_root_check=False,
):
    """
    Walk the results directory and do the checking.
    We are called with the cdw at the root of the submission directory.
    level1 division - closed|open|network
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
        if there are errors write a None as result so we can report later what
        failed
    """
    head = [
        "Organization",
        "Availability",
        "Division",
        "SystemType",
        "SystemName",
        "Platform",
        "Model",
        "MlperfModel",
        "Scenario",
        "Result",
        "Accuracy",
        "number_of_nodes",
        "host_processor_model_name",
        "host_processors_per_node",
        "host_processor_core_count",
        "accelerator_model_name",
        "accelerators_per_node",
        "Location",
        "framework",
        "operating_system",
        "notes",
        "compliance",
        "errors",
        "version",
        "inferred",
        "has_power",
        "Units",
    ]
    fmt = ",".join(["{}"] * len(head)) + "\n"
    csv.write(",".join(head) + "\n")
    results = {}
    systems = {}

    def log_result(
        submitter,
        available,
        division,
        system_type,
        system_name,
        system_desc,
        model_name,
        mlperf_model,
        scenario_fixed,
        r,
        acc,
        system_json,
        name,
        compliance,
        errors,
        config,
        inferred=0,
        power_metric=0,
    ):
        notes = system_json.get("hw_notes", "")
        if system_json.get("sw_notes"):
            notes = notes + ". " if notes else ""
            notes = notes + system_json.get("sw_notes")
        unit_dict = {
            "SingleStream": "Latency (ms)",
            "MultiStream": "Latency (ms)",
            "Offline": "Samples/s",
            "Server": "Queries/s",
        }
        power_unit_dict = {
            "SingleStream": "millijoules",
            "MultiStream": "millijoules",
            "Offline": "Watts",
            "Server": "Watts",
        }
        unit = unit_dict[scenario_fixed]
        power_unit = power_unit_dict[scenario_fixed]

        csv.write(
            fmt.format(
                submitter,
                available,
                division,
                '"' + system_type + '"',
                '"' + system_name + '"',
                system_desc,
                model_name,
                mlperf_model,
                scenario_fixed,
                r,
                acc,
                system_json.get("number_of_nodes"),
                '"' + system_json.get("host_processor_model_name") + '"',
                system_json.get("host_processors_per_node"),
                system_json.get("host_processor_core_count"),
                '"' + system_json.get("accelerator_model_name") + '"',
                '"' + str(system_json.get("accelerators_per_node")) + '"',
                name.replace("\\", "/"),
                '"' + system_json.get("framework", "") + '"',
                '"' + system_json.get("operating_system", "") + '"',
                '"' + notes + '"',
                compliance,
                errors,
                config.version,
                inferred,
                power_metric > 0,
                unit,
            )
        )

        if power_metric > 0:
            csv.write(
                fmt.format(
                    submitter,
                    available,
                    division,
                    '"' + system_type + '"',
                    '"' + system_name + '"',
                    system_desc,
                    model_name,
                    mlperf_model,
                    scenario_fixed,
                    power_metric,
                    acc,
                    system_json.get("number_of_nodes"),
                    '"' + system_json.get("host_processor_model_name") + '"',
                    system_json.get("host_processors_per_node"),
                    system_json.get("host_processor_core_count"),
                    '"' + system_json.get("accelerator_model_name") + '"',
                    '"' + str(system_json.get("accelerators_per_node")) + '"',
                    name.replace("\\", "/"),
                    '"' + system_json.get("framework", "") + '"',
                    '"' + system_json.get("operating_system", "") + '"',
                    '"' + notes + '"',
                    compliance,
                    errors,
                    config.version,
                    inferred,
                    power_metric > 0,
                    power_unit,
                )
            )

    # we are at the top of the submission directory
    for division in list_dir("."):
        # we are looking at ./$division, ie ./closed
        if division not in VALID_DIVISIONS:
            if division not in [".git", ".github", "assets"]:
                log.error("invalid division in input dir %s", division)
            continue
        is_closed_or_network = division in ["closed", "network"]
        # Look at files outside the division folder
        files_outside_division = [
            f for f in list_files(".") if not (f.endswith(".md") or f.endswith(".pdf"))
        ]
        if len(files_outside_division) > 0 and not skip_extra_files_in_root_check:
            log.error(
                "Root contains files outside division folder %s",
                division,
                files_outside_division,
            )
            results[f"root"] = None
            break
        # Look at files outside the submitter folder
        files_outside_submitter = list_files(division)
        if len(files_outside_submitter) > 0:
            log.error(
                "%s contains files outside submitter folder %s",
                division,
                files_outside_submitter,
            )
            results[f"{division}"] = None
            continue

        if division not in systems:
            systems[division] = {}
            systems[division]["power"] = {}
            systems[division]["non_power"] = {}

        for submitter in list_dir(division):
            # we are looking at ./$division/$submitter, ie ./closed/mlperf_org
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
                continue

            ## Apply folder checks
            dirs = list_dirs_recursively(division, submitter)
            files = list_files_recursively(division, submitter)

            # Check symbolic links
            broken_symbolic_links = [
                f
                for f in files
                if os.path.islink(f) and not os.path.exists(os.readlink(f))
            ]
            if len(broken_symbolic_links) > 0:
                log.error(
                    "%s/%s contains broken symbolic links: %s",
                    division,
                    submitter,
                    broken_symbolic_links,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for files over 50 MB
            files_over_size_limit = [
                f
                for f in files
                if os.path.getsize(f) > FILE_SIZE_LIMIT_MB * MB_TO_BYTES
            ]
            if len(files_over_size_limit) > 0:
                log.error(
                    "%s/%s contains files with size greater than 50 MB: %s",
                    division,
                    submitter,
                    files_over_size_limit,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check files and folders with git unfriendly names
            dir_names = [(dir_, dir_.split("/")[-1]) for dir_ in dirs]
            file_names = [(file_, file_.split("/")[-1]) for file_ in files]
            git_error_names = [
                name[0] for name in dir_names if name[1].startswith(".")
            ] + [name[0] for name in file_names if name[1].startswith(".")]
            if len(git_error_names) > 0:
                log.error(
                    "%s/%s contains files with git unfriendly name: %s",
                    division,
                    submitter,
                    git_error_names,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check files and folders with spaces names
            space_error_names = [name[0] for name in dir_names if " " in name[1]] + [
                name[0] for name in file_names if " " in name[1]
            ]
            if len(space_error_names) > 0:
                log.error(
                    "%s/%s contains files with spaces in their names: %s",
                    division,
                    submitter,
                    space_error_names,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for pycache folders
            pycache_dirs = [dir for dir in dirs if dir.endswith("__pycache__")]
            if len(pycache_dirs) > 0:
                log.error(
                    "%s/%s has the following __pycache__ directories: %s",
                    division,
                    submitter,
                    pycache_dirs,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for empty folders
            empty_dirs = list_empty_dirs_recursively(division, submitter)
            if len(empty_dirs) > 0:
                log.error(
                    "%s/%s has the following empty directories: %s",
                    division,
                    submitter,
                    empty_dirs,
                )
                results[f"{division}/{submitter}"] = None
                continue

            # Check for extra model mapping
            extra_model_mapping = None
            if division == "open":
                model_mapping_path = (
                    f"{division}/{submitter}/{config.extra_model_benchmark_map}"
                )
                if os.path.exists(model_mapping_path):
                    with open(model_mapping_path) as fp:
                        extra_model_mapping = json.load(fp)

            for system_desc in list_dir(results_path):
                # we are looking at ./$division/$submitter/results/$system_desc, ie ./closed/mlperf_org/results/t4-ort

                #
                # check if system_id is good.
                #
                system_id_json = os.path.join(
                    division, submitter, "systems", system_desc + ".json"
                )
                if not os.path.exists(system_id_json):
                    log.error(
                        "no system_desc for %s/%s/%s", division, submitter, system_desc
                    )
                    results[os.path.join(results_path, system_desc)] = None
                    continue

                name = os.path.join(results_path, system_desc)
                with open(system_id_json) as f:
                    system_json = json.load(f)
                    available = system_json.get("status").lower()
                    if available not in VALID_AVAILABILITIES:
                        log.error(
                            "%s has invalid status (%s)", system_id_json, available
                        )
                        results[name] = None
                        continue
                    system_type = system_json.get("system_type")
                    if config.version not in ["v0.5"]:
                        valid_system_types = ["datacenter", "edge"]
                        if config.version not in ["v0.7"]:
                            valid_system_types += ["datacenter,edge", "edge,datacenter"]
                        if system_type not in valid_system_types:
                            log.error(
                                "%s has invalid system type (%s)",
                                system_id_json,
                                system_type,
                            )
                            results[name] = None
                            continue
                    config.set_type(system_type)
                    if not check_system_desc_id(
                        name,
                        system_json,
                        submitter,
                        division,
                        config.version,
                        skip_meaningful_fields_emptiness_check,
                    ):
                        results[name] = None
                        continue

                #
                # Look at each model
                #
                for model_name in list_dir(results_path, system_desc):
                    # we are looking at ./$division/$submitter/results/$system_desc/$model,
                    #   ie ./closed/mlperf_org/results/t4-ort/bert
                    name = os.path.join(results_path, system_desc, model_name)
                    mlperf_model = config.get_mlperf_model(
                        model_name, extra_model_mapping
                    )

                    if is_closed_or_network and mlperf_model not in config.models:
                        # for closed/network divisions we want the model name to match.
                        # for open division the model_name might be different than the task
                        log.error(
                            "%s has an invalid model %s for closed/network division",
                            name,
                            model_name,
                        )
                        results[name] = None
                        continue

                    #
                    # Look at each scenario
                    #
                    required_scenarios = config.get_required(mlperf_model)
                    if required_scenarios is None:
                        log.error(
                            "%s has an invalid model %s, system_type=%s",
                            name,
                            mlperf_model,
                            system_type,
                        )
                        results[name] = None
                        continue

                    errors = 0
                    all_scenarios = set(
                        list(required_scenarios)
                        + list(config.get_optional(mlperf_model))
                    )
                    for scenario in list_dir(results_path, system_desc, model_name):
                        # some submissions in v0.5 use lower case scenarios - map them for now
                        scenario_fixed = SCENARIO_MAPPING.get(scenario, scenario)

                        # we are looking at ./$division/$submitter/results/$system_desc/$model/$scenario,
                        #   ie ./closed/mlperf_org/results/t4-ort/bert/Offline
                        name = os.path.join(
                            results_path, system_desc, model_name, scenario
                        )
                        results[name] = None
                        if is_closed_or_network and scenario_fixed not in all_scenarios:
                            log.error(
                                "%s ignoring scenario %s (neither required nor optional)",
                                name,
                                scenario,
                            )
                            results[name] = None
                            errors += 1
                            continue

                        # check if this submission has power logs
                        power_path = os.path.join(name, "performance", "power")
                        has_power = os.path.exists(power_path)

                        if has_power:
                            log.info("Detected power logs for %s", name)
                            if config.version in [
                                "v1.0",
                                "v1.1",
                                "v2.0",
                                "v2.1",
                                "v3.0",
                                "v3.1"
                            ]:
                                pass  # Submission checker was not enforcing this
                            # The power related system_desc_fields are not used by submitters currently.
                            # Turning this check off for now
                            elif False and not check_system_desc_id_power(
                                name,
                                system_json,
                                submitter,
                                division,
                                config.version,
                                skip_meaningful_fields_emptiness_check,
                            ):
                                results[name] = None
                                errors += 1
                                continue

                        # check if measurement_dir is good.
                        measurement_dir = os.path.join(
                            division,
                            submitter,
                            "measurements",
                            system_desc,
                            model_name,
                            scenario,
                        )
                        if not os.path.exists(measurement_dir):
                            log.error("no measurement_dir for %s", measurement_dir)
                            results[measurement_dir] = None
                            errors += 1
                            continue
                        else:
                            if not check_measurement_dir(
                                measurement_dir,
                                name,
                                system_desc,
                                os.path.join(division, submitter),
                                model_name,
                                scenario,
                                has_power,
                                skip_meaningful_fields_emptiness_check,
                                skip_empty_files_check,
                                skip_check_power_measure_files,
                            ):
                                log.error(
                                    "%s measurement_dir has issues", measurement_dir
                                )
                                results[measurement_dir] = None
                                errors += 1
                                continue

                        # check accuracy
                        accuracy_is_valid = False
                        acc_path = os.path.join(name, "accuracy")
                        if not os.path.exists(os.path.join(acc_path, "accuracy.txt")):
                            log.error(
                                "%s has no accuracy.txt. Generate it with accuracy-imagenet.py or accuracy-coco.py or "
                                "process_accuracy.py",
                                acc_path,
                            )
                        else:
                            diff = files_diff(list_files(acc_path), REQUIRED_ACC_FILES)
                            if diff:
                                log.error(
                                    "%s has file list mismatch (%s)", acc_path, diff
                                )
                            accuracy_is_valid, acc = check_accuracy_dir(
                                config,
                                mlperf_model,
                                acc_path,
                                debug or is_closed_or_network,
                            )
                            if not accuracy_is_valid and not is_closed_or_network:
                                if debug:
                                    log.warning(
                                        "%s, accuracy not valid but taken for open",
                                        acc_path,
                                    )
                                accuracy_is_valid = True
                            if not accuracy_is_valid:
                                # a little below we'll not copy this into the results csv
                                errors += 1
                                log.error("%s, accuracy not valid", acc_path)

                        inferred = 0
                        if scenario in ["Server"] and config.version in [
                            "v0.5",
                            "v0.7",
                        ]:
                            n = ["run_1", "run_2", "run_3", "run_4", "run_5"]
                        else:
                            n = ["run_1"]

                        for i in n:
                            perf_path = os.path.join(name, "performance", i)
                            if not os.path.exists(perf_path):
                                log.error("%s is missing", perf_path)
                                is_valid, r = False, None
                                continue
                            if has_power:
                                required_perf_files = (
                                    REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
                                )
                            else:
                                required_perf_files = REQUIRED_PERF_FILES
                            diff = files_diff(
                                list_files(perf_path),
                                required_perf_files,
                                OPTIONAL_PERF_FILES,
                            )
                            if diff:
                                log.error(
                                    "%s has file list mismatch (%s)", perf_path, diff
                                )
                                is_valid, r = False, None
                                continue

                            try:
                                is_valid, r, is_inferred = check_performance_dir(
                                    config,
                                    mlperf_model,
                                    perf_path,
                                    scenario_fixed,
                                    division,
                                    system_json,
                                    has_power,
                                )
                                if is_inferred:
                                    inferred = 1
                                    log.info(
                                        "%s has inferred results, qps=%s", perf_path, r
                                    )
                            except Exception as e:
                                log.error(
                                    "%s caused exception in check_performance_dir: %s",
                                    perf_path,
                                    e,
                                )
                                is_valid, r = False, None

                            power_metric = 0
                            if has_power:
                                try:
                                    ranging_path = os.path.join(
                                        name, "performance", "ranging"
                                    )
                                    (
                                        ranging_r
                                    ) = get_performance_metric(
                                        config,
                                        mlperf_model,
                                        ranging_path,
                                        scenario_fixed,
                                        division,
                                        system_json,
                                        has_power,
                                    )
                                except Exception as e:
                                    log.error(
                                        "%s caused exception in check_ranging_dir: %s",
                                        ranging_path,
                                        e,
                                    )
                                    is_valid, r = False, None

                                try:
                                    (
                                        power_is_valid,
                                        power_metric,
                                        power_efficiency,
                                    ) = check_power_dir(
                                        power_path,
                                        ranging_path,
                                        perf_path,
                                        scenario_fixed,
                                        r,
                                        ranging_r,
                                        config,
                                    )
                                    if not power_is_valid:
                                        is_valid = False
                                        power_metric = 0
                                except Exception as e:
                                    log.error(
                                        "%s caused exception in check_power_dir: %s",
                                        perf_path,
                                        e,
                                    )
                                    is_valid, r, power_metric = False, None, 0

                            if is_valid:
                                results[name] = (
                                    r
                                    if r is None or not has_power
                                    else (
                                        "{:f} "
                                        "with "
                                        "power_metric"
                                        " = {:f} and power_efficiency (inf/J) = {:f}"
                                    ).format(r, power_metric, power_efficiency)
                                )

                                system_id = submitter + "_" + system_desc

                                key = "power" if power_metric > 0 else "non_power"
                                if system_id not in systems[division][key]:
                                    systems[division][key][system_id] = 1
                                else:
                                    systems[division][key][system_id] += 1

                                required_scenarios.discard(scenario_fixed)
                            else:
                                log.error("%s has issues", perf_path)
                                errors += 1
                                results[name] = None

                        # check if compliance dir is good for CLOSED division
                        compliance = 0 if is_closed_or_network else 1
                        if is_closed_or_network and not skip_compliance:
                            compliance_dir = os.path.join(
                                division,
                                submitter,
                                "compliance",
                                system_desc,
                                model_name,
                                scenario,
                            )
                            if not os.path.exists(compliance_dir):
                                log.error("no compliance dir for %s", name)
                                results[name] = None
                            else:
                                if not check_compliance_dir(
                                    compliance_dir,
                                    mlperf_model,
                                    scenario_fixed,
                                    config,
                                    division,
                                    system_json,
                                ):
                                    log.error(
                                        "compliance dir %s has issues", compliance_dir
                                    )
                                    results[name] = None
                                else:
                                    compliance = 1

                        if results.get(name):
                            if accuracy_is_valid:
                                log_result(
                                    submitter,
                                    available,
                                    division,
                                    system_type,
                                    system_json.get("system_name"),
                                    system_desc,
                                    model_name,
                                    mlperf_model,
                                    scenario_fixed,
                                    r,
                                    acc,
                                    system_json,
                                    name,
                                    compliance,
                                    errors,
                                    config,
                                    inferred=inferred,
                                    power_metric=power_metric,
                                )
                            else:
                                results[name] = None
                                log.error("%s is OK but accuracy has issues", name)

                    if required_scenarios:
                        name = os.path.join(results_path, system_desc, model_name)
                        if is_closed_or_network:
                            results[name] = None
                            log.error(
                                "%s does not have all required scenarios, missing %s",
                                name,
                                required_scenarios,
                            )
                        elif debug:
                            log.warning(
                                "%s ignoring missing scenarios in open division (%s)",
                                name,
                                required_scenarios,
                            )

    return results, systems


def check_system_desc_id(
    fname,
    systems_json,
    submitter,
    division,
    version,
    skip_meaningful_fields_emptiness_check,
):
    is_valid = True
    # check all required fields
    if version in ["v0.5", "v0.7"]:
        required_fields = SYSTEM_DESC_REQUIRED_FIELDS
    elif version in ["v1.0", "v1.1", "v2.0", "v2.1", "v3.0"]:
        required_fields = (
            SYSTEM_DESC_REQUIRED_FIELDS + SYSTEM_DESC_REQUIRED_FIELDS_SINCE_V1
        )
    else:
        required_fields = (
            SYSTEM_DESC_REQUIRED_FIELDS
            + SYSTEM_DESC_REQUIRED_FIELDS_SINCE_V1
            + SYSTEM_DESC_REQUIRED_FIELDS_SINCE_V3_1
        )

    is_network_system, is_network_mode_valid = is_system_over_network(
        division, systems_json, fname
    )
    is_valid &= is_network_mode_valid
    if is_network_system:
        required_fields += SYSTEM_DESC_REQUIRED_FIELDS_NETWORK_MODE

    check_empty_fields = False if skip_meaningful_fields_emptiness_check else True

    for k in required_fields:
        if k not in systems_json:
            is_valid = False
            log.error("%s, field %s is missing", fname, k)
        elif (
            check_empty_fields
            and k in SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS
            and not systems_json[k]
        ):
            is_valid = False
            log.error(
                "%s, field %s requires a meaningful response but is empty", fname, k
            )

    if version in ["v0.5", "v0.7"]:
        all_fields = required_fields + SYSTEM_DESC_REQUIRED_FIELDS_SINCE_V1
    else:
        # SYSTEM_DESC_REQUIRED_FIELDS_POWER should be mandatory when a submission has power logs, but since we
        # check power submission in check_results_dir, the information is not available yet at this stage and we do
        # this check later
        all_fields = required_fields + SYSTEM_DESC_REQUIRED_FIELDS_POWER
    for k in systems_json.keys():
        if k not in all_fields:
            log.warning("%s, field %s is unknown", fname, k)

    if systems_json.get("submitter").lower() != submitter.lower():
        log.error(
            "%s has submitter %s, directory has %s",
            fname,
            systems_json.get("submitter"),
            submitter,
        )
        is_valid = False
    if systems_json.get("division") != division:
        log.error(
            "%s has division %s, division has %s",
            fname,
            systems_json.get("division"),
            division,
        )
        is_valid = False
    return is_valid


def check_system_desc_id_power(
    fname,
    systems_json,
    submitter,
    division,
    version,
    skip_meaningful_fields_emptiness_check,
):
    is_valid = True

    check_empty_fields = False if skip_meaningful_fields_emptiness_check else True

    for k in SYSTEM_DESC_REQUIRED_FIELDS_POWER:
        if k not in systems_json:
            is_valid = False
            log.error("%s, field %s is missing", fname, k)
        elif (
            check_empty_fields
            and k in SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS_POWER
            and not systems_json[k]
        ):
            is_valid = False
            log.error(
                "%s, field %s requires a meaningful response but is empty", fname, k
            )

    return is_valid


def check_measurement_dir(
    measurement_dir,
    fname,
    system_desc,
    root,
    model,
    scenario,
    has_power,
    skip_meaningful_fields_emptiness_check,
    skip_empty_files_check,
    skip_check_power_measure_files,
):
    files = list_files(measurement_dir)
    system_file = None
    is_valid = True
    check_empty_fields = False if skip_meaningful_fields_emptiness_check else True

    for i in REQUIRED_MEASURE_FILES:
        if i not in files:
            log.error("%s is missing %s", measurement_dir, i)
            is_valid = False
        elif not skip_empty_files_check and (
            os.stat(os.path.join(measurement_dir, i)).st_size == 0
        ):
            log.error("%s is having empty %s", measurement_dir, i)
            is_valid = False

    if has_power and not skip_check_power_measure_files:
        for i in REQUIRED_POWER_MEASURE_FILES:
            file_re = measurement_dir + "/../../../**/" + i
            file_paths = glob(file_re, recursive=True)
            if not file_paths:
                log.error("%s is missing %s", measurement_dir, i)
                is_valid = False
            elif not skip_empty_files_check and all((os.stat(file_path).st_size == 0) for file_path in file_paths):
                log.error("%s is having empty %s", measurement_dir, i)
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

    if not system_file and os.environ.get("INFER_SYSTEM_FILE", "") == "yes":
        for i in files:
            if i.endswith(".json"):
                system_file = system_desc + ".json"
                os.rename(
                    os.path.join(measurement_dir, i),
                    os.path.join(measurement_dir, system_file),
                )
                end = len(".json")
                break

    if system_file:
        with open(os.path.join(measurement_dir, system_file), "r") as f:
            j = json.load(f)
            for k in SYSTEM_IMP_REQUIRED_FILES:
                if k not in j:
                    is_valid = False
                    log.error("%s, field %s is missing", fname, k)
                elif check_empty_fields and not j[k]:
                    is_valid = False
                    log.error("%s, field %s is missing meaningful value", fname, k)

        impl = system_file[len(system_desc) + 1 : -end]
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


def check_compliance_perf_dir(test_dir):
    is_valid = False

    fname = os.path.join(test_dir, "verify_performance.txt")
    if not os.path.exists(fname):
        log.error("%s is missing in %s", fname, test_dir)
        is_valid = False
    else:
        with open(fname, "r") as f:
            for line in f:
                # look for: TEST PASS
                if "TEST PASS" in line:
                    is_valid = True
                    break
        if is_valid == False:
            log.error("Compliance test performance check in %s failed", test_dir)

        # Check performance dir
        test_perf_path = os.path.join(test_dir, "performance", "run_1")
        if not os.path.exists(test_perf_path):
            log.error("%s has no performance/run_1 directory", test_dir)
            is_valid = False
        else:
            diff = files_diff(
                list_files(test_perf_path),
                REQUIRED_COMP_PER_FILES,
                ["mlperf_log_accuracy.json"],
            )
            if diff:
                log.error("%s has file list mismatch (%s)", test_perf_path, diff)
                is_valid = False

    return is_valid


def check_compliance_acc_dir(test_dir, model, config):
    is_valid = False
    acc_passed = False

    fname = os.path.join(test_dir, "verify_accuracy.txt")
    if not os.path.exists(fname):
        log.error("%s is missing in %s", fname, test_dir)
    else:
        # Accuracy can fail for TEST01
        is_valid = True
        with open(fname, "r") as f:
            for line in f:
                # look for: TEST PASS
                if "TEST PASS" in line:
                    acc_passed = True
                    break
        if acc_passed == False:
            log.info(
                "Compliance test accuracy check (deterministic mode) in %s failed",
                test_dir,
            )

        # Check Accuracy dir
        test_acc_path = os.path.join(test_dir, "accuracy")
        if not os.path.exists(test_acc_path):
            log.error("%s has no accuracy directory", test_dir)
            is_valid = False
        else:
            diff = files_diff(
                list_files(test_acc_path),
                REQUIRED_TEST01_ACC_FILES_1
                if acc_passed
                else REQUIRED_TEST01_ACC_FILES,
            )
            if diff:
                log.error("%s has file list mismatch (%s)", test_acc_path, diff)
                is_valid = False
            elif not acc_passed:
                target = config.get_accuracy_target(model)
                patterns = []
                acc_types = []
                for i in range(0, len(target), 2):
                    acc_type = target[i:i+2]
                    acc_types.append(acc_type)
                    patterns.append(ACC_PATTERN[acc_type[0]])
                acc_seen = [False for _ in acc_type]



                more_accurate = model.find("99.9")
                if more_accurate == -1:
                    required_delta_perc = 1
                else:
                    required_delta_perc = 0.1
                
                acc_baseline = {
                    acc_type: 0 for acc_type in acc_types
                }
                acc_compliance = {
                    acc_type: 0 for acc_type in acc_types
                }
                with open(
                    os.path.join(test_acc_path, "baseline_accuracy.txt"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line in f:
                        for acc_type, pattern in zip(acc_types, patterns):
                            m = re.match(pattern, line)
                            if m:
                                acc_baseline[acc_type] = float(m.group(1))
                with open(
                    os.path.join(test_acc_path, "compliance_accuracy.txt"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line in f:
                        for acc_type, pattern in zip(acc_types, patterns):
                            m = re.match(pattern, line)
                            if m:
                                acc_compliance[acc_type] = float(m.group(1))
                for acc_type in acc_types:
                    if acc_baseline[acc_type] == 0 or acc_compliance[acc_type] == 0:
                        is_valid = False
                        break
                    else:
                        delta_perc = abs(1 - acc_baseline[acc_type] / acc_compliance[acc_type]) * 100
                        if delta_perc <= required_delta_perc:
                            is_valid = True
                        else:
                            is_valid = False
                            break

    return is_valid


def check_compliance_dir(
    compliance_dir, model, scenario, config, division, system_json
):
    compliance_perf_pass = True
    compliance_perf_dir_pass = True
    compliance_acc_pass = True
    test_list = ["TEST01", "TEST04", "TEST05"]

    if model in [
        "rnnt",
        "bert-99",
        "bert-99.9",
        "dlrm-99",
        "dlrm-99.9",
        "3d-unet-99",
        "3d-unet-99.9",
        "retinanet",
        "gptj-99",
        "gptj-99.9",
        "gpt3-99",
        "gpt3-99.9",
    ]:
        test_list.remove("TEST04")

    if model in [
        "gptj-99",
        "gptj-99.9",
        "gpt3-99",
        "gpt3-99.9",
    ]:
        test_list.remove("TEST05")

    # Check performance of all Tests
    for test in test_list:
        test_dir = os.path.join(compliance_dir, test)
        if not os.path.exists(test_dir):
            log.error("Missing %s in compliance dir %s", test, compliance_dir)
            compliance_perf_dir_pass = False
        else:
            try:
                compliance_perf_dir = os.path.join(
                    compliance_dir, test, "performance", "run_1"
                )
                compliance_perf_valid, r, is_inferred = check_performance_dir(
                    config, model, compliance_perf_dir, scenario, division, system_json
                )
                if is_inferred:
                    log.info("%s has inferred results, qps=%s", compliance_perf_dir, r)
            except Exception as e:
                log.error(
                    "%s caused exception in check_performance_dir: %s",
                    compliance_perf_dir,
                    e,
                )
                is_valid, r = False, None
            compliance_perf_pass = (
                compliance_perf_pass
                and check_compliance_perf_dir(test_dir)
                and compliance_perf_valid
            )

    # Check accuracy for TEST01
    compliance_acc_pass = check_compliance_acc_dir(
        os.path.join(compliance_dir, "TEST01"), model, config
    )

    return compliance_perf_pass and compliance_acc_pass and compliance_perf_dir_pass


def main():
    args = get_args()

    config = Config(
        args.version,
        args.extra_model_benchmark_map,
        ignore_uncommited=args.submission_exceptions,
        skip_power_check=args.skip_power_check,
    )

    if args.version in [ "v0.5", "v0.7", "v1.0", "v1.1", "v2.0", "v2.1", "v3.0"]:
        args.skip_meaningful_fields_emptiness_check = True
        args.skip_empty_files_check = True
        args.skip_check_power_measure_files = True

    with open(args.csv, "w") as csv:
        os.chdir(args.input)
        # check results directory
        results, systems = check_results_dir(
            config,
            args.submitter,
            args.skip_compliance,
            csv,
            args.debug,
            args.skip_meaningful_fields_emptiness_check,
            args.skip_empty_files_check,
            args.skip_check_power_measure_files,
            args.skip_extra_files_in_root_check
        )

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

    closed_systems = systems.get("closed", {})
    open_systems = systems.get("open", {})
    network_systems = systems.get("network", {})
    closed_power_systems = closed_systems.get("power", {})
    closed_non_power_systems = closed_systems.get("non_power", {})
    open_power_systems = open_systems.get("power", {})
    open_non_power_systems = open_systems.get("non_power", {})
    network_power_systems = network_systems.get("power", {})
    network_non_power_systems = network_systems.get("non_power", {})

    number_closed_power_systems = len(closed_power_systems)
    number_closed_non_power_systems = len(closed_non_power_systems)
    number_closed_systems = (
        number_closed_power_systems + number_closed_non_power_systems
    )
    number_open_power_systems = len(open_power_systems)
    number_open_non_power_systems = len(open_non_power_systems)
    number_open_systems = number_open_power_systems + number_open_non_power_systems
    number_network_power_systems = len(network_power_systems)
    number_network_non_power_systems = len(network_non_power_systems)
    number_network_systems = (
        number_network_power_systems + number_network_non_power_systems
    )

    def merge_two_dict(x, y):
        z = x.copy()
        for key in y:
            if key not in z:
                z[key] = y[key]
            else:
                z[key] += y[key]
        return z

    # systems can be repeating in open, closed and network
    unique_closed_systems = merge_two_dict(
        closed_power_systems, closed_non_power_systems
    )
    unique_open_systems = merge_two_dict(open_power_systems, open_non_power_systems)
    unique_network_systems = merge_two_dict(
        network_power_systems, network_non_power_systems
    )

    unique_systems = merge_two_dict(unique_closed_systems, unique_open_systems)
    unique_systems = merge_two_dict(unique_systems, unique_network_systems)

    # power systems can be repeating in open, closed and network
    unique_power_systems = merge_two_dict(closed_power_systems, open_power_systems)
    unique_power_systems = merge_two_dict(unique_power_systems, network_power_systems)

    number_systems = len(unique_systems)
    number_power_systems = len(unique_power_systems)

    # Counting the number of closed,open and network results
    def sum_dict_values(x):
        count = 0
        for key in x:
            count += x[key]
        return count

    count_closed_power_results = sum_dict_values(closed_power_systems)
    count_closed_non_power_results = sum_dict_values(closed_non_power_systems)
    count_closed_results = count_closed_power_results + count_closed_non_power_results

    count_open_power_results = sum_dict_values(open_power_systems)
    count_open_non_power_results = sum_dict_values(open_non_power_systems)
    count_open_results = count_open_power_results + count_open_non_power_results

    count_network_power_results = sum_dict_values(network_power_systems)
    count_network_non_power_results = sum_dict_values(network_non_power_systems)
    count_network_results = (
        count_network_power_results + count_network_non_power_results
    )

    count_power_results = (
        count_closed_power_results
        + count_open_power_results
        + count_network_power_results
    )

    # print summary
    log.info("---")
    log.info(
        "Results=%d, NoResults=%d, Power Results=%d",
        with_results,
        len(results) - with_results,
        count_power_results,
    )

    log.info("---")
    log.info(
        "Closed Results=%d, Closed Power Results=%d\n",
        count_closed_results,
        count_closed_power_results,
    )
    log.info(
        "Open Results=%d, Open Power Results=%d\n",
        count_open_results,
        count_open_power_results,
    )
    log.info(
        "Network Results=%d, Network Power Results=%d\n",
        count_network_results,
        count_network_power_results,
    )
    log.info("---")

    log.info("Systems=%d, Power Systems=%d", number_systems, number_power_systems)
    log.info(
        "Closed Systems=%d, Closed Power Systems=%d",
        number_closed_systems,
        number_closed_power_systems,
    )
    log.info(
        "Open Systems=%d, Open Power Systems=%d",
        number_open_systems,
        number_open_power_systems,
    )
    log.info(
        "Network Systems=%d, Network Power Systems=%d",
        number_network_systems,
        number_network_power_systems,
    )
    log.info("---")
    if len(results) != with_results:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
