# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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
import mlperf_loadgen as lg
import subprocess

from squad_QSL import get_squad_QSL

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tf","pytorch","onnxruntime","tf_estimator"], default="tf", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline", "Server", "MultiStream"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--quantized", action="store_true", help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true", help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument("--mlperf_conf", default="build/mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    args = parser.parse_args()
    return args

scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}

def main():
    args = get_args()

    if args.backend == "pytorch":
        assert not args.quantized, "Quantized model is only supported by onnxruntime backend!"
        assert not args.profile, "Profiling is only supported by onnxruntime backend!"
        from pytorch_SUT import get_pytorch_sut
        sut = get_pytorch_sut()
    elif args.backend == "tf":
        assert not args.quantized, "Quantized model is only supported by onnxruntime backend!"
        assert not args.profile, "Profiling is only supported by onnxruntime backend!"
        from tf_SUT import get_tf_sut
        sut = get_tf_sut()
    elif args.backend == "tf_estimator":
        assert not args.quantized, "Quantized model is only supported by onnxruntime backend!"
        assert not args.profile, "Profiling is only supported by onnxruntime backend!"
        from tf_estimator_SUT import get_tf_estimator_sut
        sut = get_tf_estimator_sut()
    elif args.backend == "onnxruntime":
        from onnxruntime_SUT import get_onnxruntime_sut
        sut = get_onnxruntime_sut(args)
    else:
        raise ValueError("Unknown backend: {:}".format(args.backend))

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "bert", args.scenario)
    settings.FromConfig(args.user_conf, "bert", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings

    print("Running LoadGen test...")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl.qsl, settings, log_settings)

    if args.accuracy:
        cmd = "python3 accuracy-squad.py"
        subprocess.check_call(cmd, shell=True)

    print("Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl.qsl)

if __name__ == "__main__":
    main()
