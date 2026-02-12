import argparse
import logging
import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    __package__ = "submission_checker"

from .constants import MODEL_CONFIG
from .configuration.configuration import Config
from .loader import Loader
from .checks.performance_check import PerformanceCheck
from .checks.accuracy_check import AccuracyCheck
from .checks.system_check import SystemCheck
from .checks.measurements_checks import MeasurementsCheck
from .checks.compliance_check import ComplianceCheck
from .checks.power_check import PowerCheck
from .results import ResultExporter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


def get_args():
    """Parse command-line arguments for the submission checker.

    Sets up an ArgumentParser with options for input directory, version,
    filtering, output files, and various skip flags for different checks.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
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
        "--skip-extra-accuracy-files-check",
        action="store_true",
        help="skips the check of extra accuracy files like the images folder of SDXL",
    )
    parser.add_argument(
        "--scenarios-to-skip",
        help="Delimited list input of scenarios to skip. i.e. if you only have Offline results, pass in 'Server'",
        type=str,
    )
    parser.add_argument(
        "--skip-all-systems-have-results-check",
        action="store_true",
        help="skips the check that all the systems in the systems and measurements folder should have results",
    )
    parser.add_argument(
        "--skip-calibration-check",
        action="store_true",
        help="skips the check that the calibration documentation should exist",
    )
    parser.add_argument(
        "--skip-dataset-size-check",
        action="store_true",
        help="skips dataset size check, only for backwards compatibility",
    )
    args = parser.parse_args()
    return args


def main():
    """Run the MLPerf submission checker on the provided directory.

    Parses arguments, initializes configuration and loader, iterates
    through all submissions, runs validation checks (performance,
    accuracy, system, measurements, power), collects results, and
    exports summaries. Logs pass/fail status and statistics.

    Returns:
        int: 0 if all submissions pass checks, 1 if any errors found.
    """
    args = get_args()

    config = Config(
        args.version,
        args.extra_model_benchmark_map,
        ignore_uncommited=args.submission_exceptions,
        skip_compliance=args.skip_power_check,
        skip_power_check=args.skip_power_check,
        skip_meaningful_fields_emptiness_check=args.skip_meaningful_fields_emptiness_check,
        skip_check_power_measure_files=args.skip_check_power_measure_files,
        skip_empty_files_check=args.skip_empty_files_check,
        skip_extra_files_in_root_check=args.skip_extra_files_in_root_check,
        skip_extra_accuracy_files_check=args.skip_extra_accuracy_files_check,
        skip_all_systems_have_results_check=args.skip_all_systems_have_results_check,
        skip_calibration_check=args.skip_calibration_check,
        skip_dataset_size_check=args.skip_dataset_size_check
    )

    if args.scenarios_to_skip:
        scenarios_to_skip = [
            scenario for scenario in args.scenarios_to_skip.split(",")]
    else:
        scenarios_to_skip = []

    loader = Loader(args.input, args.version, config)
    exporter = ResultExporter(args.csv, config)
    results = {}
    systems = {}
    for division in ["closed", "open", "network"]:
        systems[division] = {}
        systems[division]["power"] = {}
        systems[division]["non_power"] = {}

    # Main loop over all the submissions
    for logs in loader.load():
        # Initialize check classes
        performance_checks = PerformanceCheck(
            log, logs.loader_data["perf_path"], config, logs)
        accuracy_checks = AccuracyCheck(
            log, logs.loader_data["acc_path"], config, logs)
        system_checks = SystemCheck(
            log, logs.loader_data["system_path"], config, logs)
        measurements_checks = MeasurementsCheck(
            log, logs.loader_data["measurements_path"], config, logs)
        power_checks = PowerCheck(
            log, logs.loader_data["power_dir_path"], config, logs)
        # Run checks
        valid = True
        valid &= performance_checks()
        valid &= accuracy_checks()
        valid &= system_checks()
        valid &= measurements_checks()
        valid &= power_checks()
        # Add results to summary
        if valid:
            # Results dictionary
            results[logs.loader_data.get("perf_path")] = logs.loader_data.get(
                "performance_metric")
            # System dictionary
            system_id = logs.loader_data.get("system")
            if os.path.exists(logs.loader_data.get("power_dir_path", "")):
                if system_id in systems[logs.loader_data.get(
                        "division")]["power"]:
                    systems[logs.loader_data.get(
                        "division")]["power"][system_id] += 1
                else:
                    systems[logs.loader_data.get(
                        "division")]["power"][system_id] = 1
            else:
                if system_id in systems[logs.loader_data.get(
                        "division")]["non_power"]:
                    systems[logs.loader_data.get(
                        "division")]["non_power"][system_id] += 1
                else:
                    systems[logs.loader_data.get(
                        "division")]["non_power"][system_id] = 1
            # CSV exporter
            exporter.add_result(logs)
        else:
            results[logs.loader_data.get("perf_path")] = None
    # Export results
    exporter.export()

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
    unique_open_systems = merge_two_dict(
        open_power_systems, open_non_power_systems)
    unique_network_systems = merge_two_dict(
        network_power_systems, network_non_power_systems
    )

    unique_systems = merge_two_dict(unique_closed_systems, unique_open_systems)
    unique_systems = merge_two_dict(unique_systems, unique_network_systems)

    # power systems can be repeating in open, closed and network
    unique_power_systems = merge_two_dict(
        closed_power_systems, open_power_systems)
    unique_power_systems = merge_two_dict(
        unique_power_systems, network_power_systems)

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
    count_network_non_power_results = sum_dict_values(
        network_non_power_systems)
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

    log.info(
        "Systems=%d, Power Systems=%d",
        number_systems,
        number_power_systems)
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
