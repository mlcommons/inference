from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
from .power.power_checker import check as check_power_more
from ..utils import *
import os
import sys

class PowerCheck(BaseCheck):
    def __init__(self, log, path, config: Config, submission_logs: SubmissionLogs):
        super().__init__(log, path)
        self.config = config
        self.submission_logs = submission_logs
        self.power_path = self.submission_logs.loader_data.get("power_dir_path", "")
        self.testing_path = os.path.dirname(self.submission_logs.loader_data.get("perf_path", ""))
        self.ranging_path = os.path.join(os.path.dirname(self.testing_path), "ranging")
        self.has_power = os.path.exists(self.power_path)
        self.setup_checks()

    def setup_checks(self):
        self.checks.append(self.required_files_check)
        self.checks.append(self.external_power_check)

    
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