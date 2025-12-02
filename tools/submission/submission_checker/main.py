import argparse
import logging
import os

from .constants import MODEL_CONFIG
from .configuration.configuration import Config
from .loader import Loader
from .checks.performance_check import PerformanceCheck

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument(
        "--version",
        default="v5.1",
        choices=list(MODEL_CONFIG.keys()),
        help="mlperf version",
    )
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument(
        "--csv",
        default="summary.csv",
        help="csv file with results")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="extra debug output")
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
    parser.add_argument(
        "--scenarios-to-skip",
        help="Delimited list input of scenarios to skip. i.e. if you only have Offline results, pass in 'Server'",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    config = Config(
        args.version,
        args.extra_model_benchmark_map,
        ignore_uncommited=args.submission_exceptions,
        skip_power_check=args.skip_power_check,
    )

    if args.scenarios_to_skip:
        scenarios_to_skip = [
            scenario for scenario in args.scenarios_to_skip.split(",")]
    else:
        scenarios_to_skip = []

    loader = Loader(args.input, args.version)
    for logs in loader.load():
        performance_checks = PerformanceCheck(log, logs.loader_data["perf_path"], config, logs)
        performance_checks.run_checks()

    with open(args.csv, "w") as csv:
        # Output summary
        pass

    # log results
    results = {}
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
    log.info(
        "Results=%d, NoResults=%d, Power Results=%d",
        with_results,
        len(results) - with_results,
        0,
    )

    log.info("---")
    log.info(
        "Closed Results=%d, Closed Power Results=%d\n",
        0,
        0,
    )
    log.info(
        "Open Results=%d, Open Power Results=%d\n",
        0,
        0,
    )
    log.info(
        "Network Results=%d, Network Power Results=%d\n",
        0,
        0,
    )
    log.info("---")

    log.info(
        "Systems=%d, Power Systems=%d",
        0,
        0)
    log.info(
        "Closed Systems=%d, Closed Power Systems=%d",
        0,
        0,
    )
    log.info(
        "Open Systems=%d, Open Power Systems=%d",
        0,
        0,
    )
    log.info(
        "Network Systems=%d, Network Power Systems=%d",
        0,
        0,
    )
    log.info("---")
    if len(results) != with_results:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0


if __name__ == "__main__":
    main()
