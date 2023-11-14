import subprocess
import mlperf_loadgen as lg
import argparse
import os

import sys
from SUT import SUT
sys.path.insert(0, os.getcwd())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["SingleStream", "Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", type=str, default="tiiuae/falcon-40b-instruct", help="Model name")
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
    parser.add_argument("--dtype", type=str, default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--device", type=str,  choices=["cpu", "cuda:0"], default="cpu", help="device to use")
    parser.add_argument("--audit-conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--mlperf-conf", type=str, default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--total-sample-count", type=int, default=24576, help="Number of samples to use in benchmark.") # TODO: This interpretation of 'total-sample-count' is a little misleading. Fix it

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

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "falcon-40b", args.scenario)
    settings.FromConfig(args.user_conf, "falcon-40b", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    log_path = os.environ.get("LOG_PATH")
    if not log_path:
        log_path = "build/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True

    sut = SUT(
        model_path=args.model_path,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        device=args.device,
    )

    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings, args.audit_conf)

    print("Test Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut)

    print("Destroying QSL...")
    lg.DestroyQSL(qsl)


if __name__ == "__main__":
    main()
