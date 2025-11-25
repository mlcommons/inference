import os
from .constants import PERFORMANCE_LOG_PATH, PERFORMANCE_SUMMARY_PATH, ACCURACY_LOG_PATH, VALID_DIVISIONS
from .utils import list_dir
from .parsers.loadgen_parser import LoadgenParser
from typing import Generator


class SubmissionLogs:
    def __init__(self, performance_log, accuracy_log) -> None:
        self.performance_log = performance_log
        self.accuracy_log = accuracy_log


class Loader:
    def __init__(self, root, version) -> None:
        self.root = root
        self.version = version
        self.perf_log_path = PERFORMANCE_LOG_PATH.get(version, PERFORMANCE_LOG_PATH["default"])
        self.perf_summary_path = PERFORMANCE_SUMMARY_PATH.get(version, PERFORMANCE_SUMMARY_PATH["default"])
        self.acc_log_path = ACCURACY_LOG_PATH.get(version, ACCURACY_LOG_PATH["default"])


    def load(self) -> Generator[SubmissionLogs, None]:
        for division in list_dir(self.root):
            if division not in VALID_DIVISIONS:
                continue
            division_path = os.path.join(self.root, division)
            for submitter in list_dir(division):
                results_path = os.path.join(division_path, submitter, "results")
                for system in list_dir(results_path):
                    system_path = os.path.join(results_path, system)
                    for benchmark in list_dir(system_path):
                        benchmark_path = os.path.join(system_path, benchmark)
                        for scenario in benchmark_path:
                            scenario_path = os.path.join(benchmark_path, benchmark)
                            perf_log = LoadgenParser(self.perf_log_path.format(division = division, submitter = submitter, system = system, benchmark = system, scenario = scenario))
                            acc_log = LoadgenParser(self.acc_log_path.format(division = division, submitter = submitter, system = system, benchmark = system, scenario = scenario))
                            yield SubmissionLogs(perf_log, acc_log)
        yield None

