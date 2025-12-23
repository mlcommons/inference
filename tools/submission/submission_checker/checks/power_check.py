from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
from .power.power_checker import check as check_power_more
from ..utils import *
import os
import sys
import datetime

class PowerCheck(BaseCheck):
    def __init__(self, log, path, config: Config, submission_logs: SubmissionLogs):
        super().__init__(log, path)
        self.config = config
        self.submission_logs = submission_logs
        self.mlperf_log = self.submission_logs.performance_log
        self.scenario_fixed = self.submission_logs.loader_data.get("scenario", "")
        self.power_path = self.submission_logs.loader_data.get("power_dir_path", "")
        self.testing_path = os.path.dirname(self.submission_logs.loader_data.get("perf_path", ""))
        self.ranging_path = os.path.join(os.path.dirname(self.testing_path), "ranging")
        self.has_power = os.path.exists(self.power_path)
        self.setup_checks()

    def setup_checks(self):
        self.checks.append(self.required_files_check)
        self.checks.append(self.external_power_check)
        self.checks.append(self.get_power_metric_check)

    
    def required_files_check(self):
        if not self.has_power:
            return True
        
        self.log.info("Checking necessary power files for %s", self.path)
        is_valid = True
        required_files = REQUIRED_PERF_FILES + REQUIRED_PERF_POWER_FILES
        diff = files_diff(
            list_files(self.testing_path),
            required_files,
            OPTIONAL_PERF_FILES)
        if diff:
            self.log.error("%s has file list mismatch (%s)", self.testing_path, diff)
            is_valid = False
        diff = files_diff(
            list_files(self.ranging_path),
            required_files,
            OPTIONAL_PERF_FILES)
        if diff:
            self.log.error("%s has file list mismatch (%s)", self.ranging_path, diff)
            is_valid = False
        diff = files_diff(list_files(self.power_path), REQUIRED_POWER_FILES)
        if diff:
            self.log.error("%s has file list mismatch (%s)", self.power_path, diff)
            is_valid = False
        return is_valid

    def external_power_check(self):
        if not self.config.skip_power_check and self.has_power:
            self.log.info("Running external power checks for %s", self.path)
            python_version_major = int(sys.version.split(" ")[0].split(".")[0])
            python_version_minor = int(sys.version.split(" ")[0].split(".")[1])
            assert python_version_major == 3 and python_version_minor >= 7, (
                "Power check " " only " "supports " "Python " "3.7+"
            )
            perf_path = os.path.dirname(self.power_path)
            check_power_result = check_power_more(perf_path)
            sys.stdout.flush()
            sys.stderr.flush()
            if check_power_result != 0:
                self.log.error(
                    "Power WG power_checker.py did not pass for: %s",
                    perf_path)
                return False
        return True
    
    def get_power_metric_check(self):
        if not self.has_power:
            return True
        # parse the power logs
        is_valid = True
        server_timezone = datetime.timedelta(0)
        client_timezone = datetime.timedelta(0)

        datetime_format = "%m-%d-%Y %H:%M:%S.%f"
        power_begin = (
            datetime.datetime.strptime(self.mlperf_log["power_begin"], datetime_format)
            + client_timezone
        )
        power_end = (
            datetime.datetime.strptime(self.mlperf_log["power_end"], datetime_format)
            + client_timezone
        )
        # Obtain the scenario also from logs to check if power is inferred
        scenario = self.mlperf_log["effective_scenario"]

        log_path = self.testing_path
        spl_fname = os.path.join(log_path, "spl.txt")
        power_list = []
        with open(spl_fname) as f:
            for line in f:
                if not line.startswith("Time"):
                    continue
                timestamp = (
                    datetime.datetime.strptime(line.split(",")[1], datetime_format)
                    + server_timezone
                )
                if timestamp > power_begin and timestamp < power_end:
                    value = float(line.split(",")[3])
                    if value > 0:
                        power_list.append(float(line.split(",")[3]))

        if len(power_list) == 0:
            self.log.error(
                "%s has no power samples falling in power range: %s - %s",
                spl_fname,
                power_begin,
                power_end,
            )
            is_valid = False
        else:
            avg_power = sum(power_list) / len(power_list)
            power_duration = (power_end - power_begin).total_seconds()
            if self.scenario_fixed.lower() in ["offline", "server", "interactive"]:
                # In Offline and Server scenarios, the power metric is in W.
                power_metric = avg_power
                avg_power_efficiency = self.submission_logs.loader_data["performance_metric"] / avg_power

            else:
                # In SingleStream and MultiStream scenarios, the power metric is in
                # mJ/query.
                assert self.scenario_fixed.lower() in [
                    "multistream",
                    "singlestream",
                ], "Unknown scenario: {:}".format(self.scenario_fixed)

                num_queries = int(self.mlperf_log["result_query_count"])

                power_metric = avg_power * power_duration * 1000 / num_queries

                if self.scenario_fixed.lower() in ["singlestream"]:
                    samples_per_query = 1
                elif self.scenario_fixed.lower() in ["multistream"]:
                    samples_per_query = 8

                if (self.scenario_fixed.lower() in ["multistream"]
                        ) and scenario.lower() in ["singlestream"]:
                    power_metric = (
                        avg_power * power_duration * samples_per_query * 1000 / num_queries
                    )

                avg_power_efficiency = (samples_per_query * 1000) / power_metric

        self.submission_logs.loader_data["power_metric"] = power_metric
        self.submission_logs.loader_data["avg_power_efficiency"] = avg_power_efficiency
        return is_valid