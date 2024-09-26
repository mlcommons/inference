#! /usr/bin/env python3
# coding=utf-8
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2021 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import subprocess

from pathlib import Path

import mlperf_loadgen as lg


__doc__ = """
Run 3D UNet performing KiTS19 Kidney Tumore Segmentation task.
Dataset needs to be prepared through preprocessing (python preprocess.py --help).

Run inference in performance mode:
    python3 run.py --backend=$(BACKEND) --scenario=$(SCENARIO) --model=$(MODEL)

Run inference in accuracy mode:
    python3 run.py --backend=$(BACKEND) --scenario=$(SCENARIO) --model=$(MODEL) --accuracy

$(BACKEND): tensorflow, pytorch, or onnxruntime
$(SCENARIO): Offline, SingleStream, MultiStream, or Server (Note: MultiStream may be deprecated)
$(MODEL) should point to correct model for the chosen backend

If run for the accuracy, DICE scores will be summarized and printed at the end of the test, and 
inference results will be stored as NIFTI files.

Performance run can be more specific as:
    python3  run.py --backend=$(BACKEND) --scenario=$(SCENARIO) --model=$(MODEL)
                    --preprocessed_data_dir=$(PREPROCESSED_DATA_DIR)
                    --postprocessed_data_dir=$(POSTPROCESSED_DATA_DIR)
                    --mlperf_conf=$(MLPERF_CONF)
                    --user_conf=$(USER_CONF)
                    --performance_count=$(PERF_CNT)

$(MLPERF_CONF) contains various configurations MLPerf-Inference needs and used to configure LoadGen
$(USER_CONF) contains configurations such as target QPS for LoadGen and overrides part of $(MLPERF_CONF)
$(PERF_CNT) sets number of query samples guaranteed to fit in memory

More info for the above LoadGen related configs can be found at:
https://github.com/mlcommons/inference/tree/master/loadgen
"""


def get_args():
    """
    Args used for running 3D UNet KITS19
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--backend",
                        choices=["pytorch", "pytorch_checkpoint", "onnxruntime", "tensorflow"],
                        default="pytorch",
                        help="Backend")
    parser.add_argument("--scenario",
                        choices=["SingleStream", "Offline"],
                        default="Offline",
                        help="Scenario")
    parser.add_argument("--accuracy",
                        action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--mlperf_conf",
                        default="build/mlperf.conf",
                        help="mlperf rules config")
    parser.add_argument("--user_conf",
                        default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--audit_conf",
                        default="audit.conf",
                        help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--model",
                        default="build/model/3dunet_kits19_pytorch.ptc",
                        help="Path to PyTorch, ONNX, or TF model")
    parser.add_argument("--preprocessed_data_dir",
                        default="build/preprocessed_data",
                        help="path to preprocessed data")
    parser.add_argument("--performance_count",
                        type=int,
                        default=None,
                        help="performance count")
    args = parser.parse_args()
    return args


def main():
    """
    Runs 3D UNet performing KiTS19 Kidney Tumore Segmentation task as below:

    1. instantiate SUT and QSL for the chosen backend
    2. configure LoadGen for the chosen scenario
    3. configure MLPerf logger
    4. start LoadGen
    5. collect logs and if needed evaluate inference results
    6. clean up
    """
    # scenarios in LoadGen
    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "Offline": lg.TestScenario.Offline,
        "Server": lg.TestScenario.Server,
        "MultiStream": lg.TestScenario.MultiStream
    }

    args = get_args()

    # instantiate SUT as per requested backend; QSL is also instantiated
    if args.backend == "pytorch":
        from pytorch_SUT import get_sut
    elif args.backend == "pytorch_checkpoint":
        from pytorch_checkpoint_SUT import get_sut
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_sut
    elif args.backend == "tensorflow":
        from tensorflow_SUT import get_sut
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))
    sut = get_sut(args.model, args.preprocessed_data_dir,
                  args.performance_count)

    # setup LoadGen
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "3d-unet", args.scenario)
    settings.FromConfig(args.user_conf, "3d-unet", args.scenario)
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    # set up mlperf logger
    log_path = Path(os.environ.get("LOG_PATH", os.path.join("build", "logs"))).absolute()
    log_path.mkdir(parents=True, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = str(log_path)
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    # start running test, from LoadGen
    print("Running Loadgen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings, args.audit_conf)

    # if needed check accuracy
    if args.accuracy and not os.environ.get('SKIP_VERIFY_ACCURACY', False):
        print("Checking accuracy...")
        cmd = f"python3 accuracy_kits.py --preprocessed_data_dir={args.preprocessed_data_dir} --log_file={os.path.join(log_path, 'mlperf_log_accuracy.json')}"
        subprocess.check_call(cmd, shell=True)

    # all done
    print("Done!")

    # cleanup
    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)
    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)


if __name__ == "__main__":
    main()
