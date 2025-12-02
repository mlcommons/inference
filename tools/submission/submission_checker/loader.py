import os
from .constants import PERFORMANCE_LOG_PATH, PERFORMANCE_SUMMARY_PATH, ACCURACY_LOG_PATH, SYSTEM_PATH, VALID_DIVISIONS
from .utils import list_dir
from .parsers.loadgen_parser import LoadgenParser
from typing import Generator, Literal
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
)


class SubmissionLogs:
    def __init__(self, performance_log, accuracy_log, system_json, loader_data = {}) -> None:
        self.performance_log = performance_log
        self.accuracy_log = accuracy_log
        self.system_json = system_json
        self.loader_data = loader_data


class Loader:
    def __init__(self, root, version) -> None:
        self.root = root
        self.version = version
        self.logger = logging.getLogger("LoadgenParser")
        self.perf_log_path = os.path.join(self.root, PERFORMANCE_LOG_PATH.get(version, PERFORMANCE_LOG_PATH["default"]))
        self.perf_summary_path = os.path.join(self.root, PERFORMANCE_SUMMARY_PATH.get(version, PERFORMANCE_SUMMARY_PATH["default"]))
        self.acc_log_path = os.path.join(self.root, ACCURACY_LOG_PATH.get(version, ACCURACY_LOG_PATH["default"]))
        self.system_log_path = os.path.join(self.root, SYSTEM_PATH.get(version, SYSTEM_PATH["default"]))

    def load_single_log(self, path, log_type: Literal["Performance", "Accuracy", "Test", "System"]):
        log = None
        if os.path.exists(path):
            self.logger.info("Loading %s log from %s", log_type, path)
            if log_type in ["Performance", "Accuracy", "Test"]:
                log = LoadgenParser(path)
            if log_type in ["System"]:
                with open(path) as f:
                    log = json.load(f)
        else:
            self.logger.info("Could not load %s log from %s, path does not exist", log_type, path)
        return log


    def load(self) -> Generator[SubmissionLogs, None, None]:
        for division in list_dir(self.root):
            if division not in VALID_DIVISIONS:
                continue
            division_path = os.path.join(self.root, division)
            for submitter in list_dir(division_path):
                results_path = os.path.join(division_path, submitter, "results")
                model_mapping = {}
                if division == "open" and os.path.exists(os.path.join(division_path, submitter, "model_mapping.json")):
                    model_mapping = self.load_single_log(os.path.join(division_path, submitter, "model_mapping.json"), "System")
                for system in list_dir(results_path):
                    system_path = os.path.join(results_path, system)
                    system_json_path = self.system_log_path.format(division = division, submitter = submitter, system = system)
                    system_json = self.load_single_log(system_json_path, "System")
                    for benchmark in list_dir(system_path):
                        benchmark_path = os.path.join(system_path, benchmark)
                        for scenario in list_dir(benchmark_path):
                            scenario_path = os.path.join(benchmark_path, benchmark)
                            perf_path = self.perf_log_path.format(division = division, submitter = submitter, system = system, benchmark = benchmark, scenario = scenario)
                            acc_path = self.acc_log_path.format(division = division, submitter = submitter, system = system, benchmark = benchmark, scenario = scenario)
                            perf_log = self.load_single_log(perf_path, "Performance")
                            acc_log = self.load_single_log(acc_path, "Accuracy")
                            loader_data = {
                                "division": division,
                                "submitter": submitter,
                                "system": system,
                                "benchmark": benchmark,
                                "scenario": scenario,
                                "perf_path": perf_path,
                                "model_mapping": model_mapping,
                            }
                            yield SubmissionLogs(perf_log, acc_log, system_json, loader_data)
