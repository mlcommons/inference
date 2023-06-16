import subprocess
import mlperf_loadgen as lg
import argparse
import os

import sys
from backend import get_SUT
sys.path.insert(0, os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["Offline", "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", default="/sax/test/gpt3", help="")
    parser.add_argument("--dataset-path", default="gs://cnn_dailymail_public/mlperf/cnn_dailymail/seqio_cache_tasks/cnn_dailymail_3.0.0/cnn_dailymail-validation.tfrecord-00000-of-00001", help="")
    parser.add_argument("--spm-path", default="gs://cnn_dailymail_public/mlperf/vocab/c4_en_301_5Mexp2_spm.model", help="")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--mlperf-conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--log-path", default="./logs", help="log path")
    args = parser.parse_args()
    return args


scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():

    args = get_args()

    sut = get_SUT(
        scenario=args.scenario,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        spm_path=args.spm_path
    )

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "gpt-3", args.scenario)
    settings.FromConfig(args.user_conf, "gpt-3", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = True

    print("Start Testing!")
    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings)
    print("Test Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()