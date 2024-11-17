
"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
from queue import Queue

import mlperf_loadgen as lg
import numpy as np
import torch

import subprocess
from py_demo_server_lon import main as server_main

import dataset
import coco

from concurrent.futures import ThreadPoolExecutor, as_completed

# from sut_over_network_demo import main as 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

SUPPORTED_DATASETS = {
    "coco-1024": (
        coco.Coco,
        dataset.preprocess,
        coco.PostProcessCoco(),
        {"image_size": [3, 1024, 1024]},
    )
}


SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "debug": {
        "dataset": "coco-1024",
        "backend": "debug",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-pytorch": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-pytorch-dist": {
        "dataset": "coco-1024",
        "backend": "pytorch-dist",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-migraphx": {
        "dataset": "coco-1024",
        "backend": "migraphx",
        "model-name": "stable-diffusion-xl",
    },
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sut-server', default=['http://t004-005:8008', "http://t006-001:8008"], nargs='+', help='A list of server address & port') #'http://t004-006:8008'
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )
    parser.add_argument("--backend", help="Name of the backend", default="migraphx")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--model-path", help="Path to model weights")

    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="dtype of the model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "rocm"],
        help="device to run the benchmark",
    )
    parser.add_argument(
        "--latent-framework",
        default="torch",
        choices=["torch", "numpy"],
        help="framework to load the latents",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # file for LoadGen audit settings
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    # arguments to save images
    # pass this argument for official submission
    # parser.add_argument("--output-images", action="store_true", help="Store a subset of the generated images")
    # do not modify this argument for official submission
    parser.add_argument("--ids-path", help="Path to caption ids", default="tools/sample_ids.txt")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--performance-sample-count", type=int, help="performance sample count", default=5000
    )
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )
    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args




def main(): 
    
    args = get_args()
    log.info(args)
    
    # Define the command and arguments
    # command = ['python', 'script_to_run.py', '--num', '10', '--text', 'Hello, world!']
    
    server_main (args)
    
    # command = ['python', 
    #            'py_demo_server_lon.py', 
    #            '--sut-server http://t007-001:8888 http://t006-001:8888',
    #            '--dataset=coco-1024', 
    #            '--dataset-path=/work1/zixian/ziw081/inference/text_to_image/coco2014',
    #            '--profile=stable-diffusion-xl-pytorch',
    #            '--dtype=fp16',
    #            '--device=cuda',
    #            '--time=30',
    #            '--scenario=Offline',
    #            '--max-batchsize=4'
    #         ]


    # # Run the command
    # subprocess.run(command)
    
    
    

if __name__ == "__main__":
    main()
