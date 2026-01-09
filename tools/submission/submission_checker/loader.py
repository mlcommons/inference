import os
from .constants import *
from .utils import list_dir
from .parsers.loadgen_parser import LoadgenParser
from typing import Generator, Literal
from .utils import *
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
)


class SubmissionLogs:
    def __init__(self, performance_log=None, accuracy_log=None, accuracy_result=None,
                 accuracy_json=None, system_json=None, measurements_json=None, loader_data={}) -> None:
        self.performance_log = performance_log
        self.accuracy_log = accuracy_log
        self.accuracy_result = accuracy_result
        self.accuracy_json = accuracy_json
        self.system_json = system_json
        self.loader_data = loader_data
        self.measurements_json = measurements_json


class Loader:
    def __init__(self, root, version) -> None:
        self.root = root
        self.version = version
        self.logger = logging.getLogger("LoadgenParser")
        self.perf_log_path = os.path.join(
            self.root, PERFORMANCE_LOG_PATH.get(
                version, PERFORMANCE_LOG_PATH["default"]))
        self.perf_summary_path = os.path.join(
            self.root, PERFORMANCE_SUMMARY_PATH.get(
                version, PERFORMANCE_SUMMARY_PATH["default"]))
        self.acc_log_path = os.path.join(
            self.root, ACCURACY_LOG_PATH.get(
                version, ACCURACY_LOG_PATH["default"]))
        self.acc_result_path = os.path.join(
            self.root, ACCURACY_RESULT_PATH.get(
                version, ACCURACY_RESULT_PATH["default"]))
        self.acc_json_path = os.path.join(
            self.root, ACCURACY_JSON_PATH.get(
                version, ACCURACY_JSON_PATH["default"]))
        self.system_log_path = os.path.join(
            self.root, SYSTEM_PATH.get(
                version, SYSTEM_PATH["default"]))
        self.measurements_path = os.path.join(
            self.root, MEASUREMENTS_PATH.get(
                version, MEASUREMENTS_PATH["default"]))
        self.compliance_path = os.path.join(
            self.root, COMPLIANCE_PATH.get(
                version, COMPLIANCE_PATH["default"]))
        self.test01_perf_path = os.path.join(
            self.root, TEST01_PERF_PATH.get(
                version, TEST01_PERF_PATH["default"]))
        self.test01_acc_path = os.path.join(
            self.root, TEST01_ACC_PATH.get(
                version, TEST01_ACC_PATH["default"]))
        self.test04_perf_path = os.path.join(
            self.root, TEST04_PERF_PATH.get(
                version, TEST04_PERF_PATH["default"]))
        self.test04_acc_path = os.path.join(
            self.root, TEST04_ACC_PATH.get(
                version, TEST04_ACC_PATH["default"]))
        self.test06_acc_path = os.path.join(
            self.root, TEST06_ACC_PATH.get(
                version, TEST06_ACC_PATH["default"]))
        self.power_dir_path = os.path.join(
            self.root, POWER_DIR_PATH.get(
                version, POWER_DIR_PATH["default"]))
        self.src_path = os.path.join(
            self.root, SRC_PATH.get(
                version, SRC_PATH["default"]))

    def get_measurement_path(self, path, division,
                             submitter, system, benchmark, scenario):
        measurements_file = None
        if "{file}" in path:
            files = list_files(
                str(
                    os.path.dirname(path)).format(
                    division=division,
                    submitter=submitter,
                    system=system,
                    benchmark=benchmark,
                    scenario=scenario))
            for i in files:
                if i.startswith(system) and i.endswith(
                        "_" + scenario + ".json"):
                    measurements_file = i
                    # end = len("_" + scenario + ".json")
                    break
                elif i.startswith(system) and i.endswith(".json"):
                    measurements_file = i
                    # end = len(".json")
                    break
            return path.format(division=division, submitter=submitter, system=system,
                               benchmark=benchmark, scenario=scenario, file=measurements_file)
        return path.format(division=division, submitter=submitter,
                           system=system, benchmark=benchmark, scenario=scenario)

    def load_single_log(self, path, log_type: Literal["Performance", "Accuracy",
                        "AccuracyResult", "AccuracyJSON", "Test", "System", "Measurements"]):
        log = None
        if os.path.exists(path):
            self.logger.info("Loading %s log from %s", log_type, path)
            if log_type in ["Performance", "Accuracy", "Test"]:
                log = LoadgenParser(path)
            elif log_type in ["System", "Measurements"]:
                with open(path) as f:
                    log = json.load(f)
            elif log_type in ["AccuracyResult"]:
                with open(path) as f:
                    log = f.readlines()
            elif log_type in ["AccuracyJSON"]:
                log = path
            else:
                self.logger.info(
                    "Could not load %s log from %s, log type not recognized",
                    log_type,
                    path)
        else:
            self.logger.info(
                "Could not load %s log from %s, path does not exist",
                log_type,
                path)
        return log

    def load(self) -> Generator[SubmissionLogs, None, None]:
        for division in list_dir(self.root):
            if division not in VALID_DIVISIONS:
                continue
            division_path = os.path.join(self.root, division)
            for submitter in list_dir(division_path):
                results_path = os.path.join(
                    division_path, submitter, "results")
                model_mapping = {}
                if division == "open" and os.path.exists(os.path.join(
                        division_path, submitter, "model_mapping.json")):
                    model_mapping = self.load_single_log(os.path.join(
                        division_path, submitter, "model_mapping.json"), "System")
                for system in list_dir(results_path):
                    system_path = os.path.join(results_path, system)
                    system_json_path = self.system_log_path.format(
                        division=division, submitter=submitter, system=system)
                    system_json = self.load_single_log(
                        system_json_path, "System")
                    for benchmark in list_dir(system_path):
                        benchmark_path = os.path.join(system_path, benchmark)
                        for scenario in list_dir(benchmark_path):
                            scenario_path = os.path.join(
                                benchmark_path, benchmark)
                            # Format Paths for a specific submission
                            perf_path = self.perf_log_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            acc_path = self.acc_log_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            acc_result_path = self.acc_result_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            acc_json_path = self.acc_json_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            power_dir_path = self.power_dir_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            measurements_path = self.get_measurement_path(
                                self.measurements_path,
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            compliance_path = self.compliance_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            test01_perf_path = self.test01_perf_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            test01_acc_path = self.test01_acc_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            test04_perf_path = self.test04_perf_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            test04_acc_path = self.test04_acc_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            test06_acc_path = self.test06_acc_path.format(
                                division=division,
                                submitter=submitter,
                                system=system,
                                benchmark=benchmark,
                                scenario=scenario)
                            src_path = self.src_path.format(
                                division=division, submitter=submitter)

                            # Load logs
                            perf_log = self.load_single_log(
                                perf_path, "Performance")
                            acc_log = self.load_single_log(
                                acc_path, "Accuracy")
                            acc_result = self.load_single_log(
                                acc_result_path, "AccuracyResult")
                            acc_json = self.load_single_log(
                                acc_json_path, "AccuracyJSON")
                            measurements_json = self.load_single_log(
                                measurements_path, "Measurements")

                            # Load test logs
                            test01_perf_log = self.load_single_log(
                                test01_perf_path, "Performance")
                            test01_acc_result = self.load_single_log(
                                test01_acc_path, "AccuracyResult")
                            test04_perf_log = self.load_single_log(
                                test04_perf_path, "Performance")
                            test04_acc_result = self.load_single_log(
                                test04_acc_path, "AccuracyResult")
                            test06_acc_result = self.load_single_log(
                                test06_acc_path, "AccuracyResult")

                            loader_data = {
                                # Submission info
                                "division": division,
                                "submitter": submitter,
                                "system": system,
                                "benchmark": benchmark,
                                "scenario": scenario,
                                # Submission paths
                                "perf_path": perf_path,
                                "acc_path": acc_path,
                                "system_path": system_path,
                                "measurements_path": measurements_path,
                                "measurements_dir": os.path.dirname(measurements_path),
                                "compliance_path": compliance_path,
                                "model_mapping": model_mapping,
                                "power_dir_path": power_dir_path,
                                "src_path": src_path,
                                # Test paths
                                "TEST01_perf_path": test01_perf_path,
                                "TEST01_acc_path": test01_acc_path,
                                "TEST04_perf_path": test04_perf_path,
                                "TEST04_acc_path": test04_acc_path,
                                "TEST06_acc_path": test06_acc_path,
                                # Test logs
                                "TEST01_perf_log": test01_perf_log,
                                "TEST01_acc_result": test01_acc_result,
                                "TEST04_perf_log": test04_perf_log,
                                "TEST04_acc_result": test04_acc_result,
                                "TEST06_acc_result": test06_acc_result,
                            }
                            yield SubmissionLogs(perf_log, acc_log, acc_result, acc_json, system_json, measurements_json, loader_data)
