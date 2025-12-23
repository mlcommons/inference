import os

from submission_checker.checks.base import BaseCheck
from submission_checker.constants import *
from submission_checker.loader import SubmissionLogs
from submission_checker.configuration.configuration import Config
from submission_checker.utils import *


class MeasurementsCheck(BaseCheck):
    def __init__(self, log, path, config: Config, submission_logs: SubmissionLogs):
        super().__init__(log, path)
        self.name = "measurement checks"
        self.submission_logs = submission_logs
        self.measurements_json = self.submission_logs.measurements_json
        self.submitter = self.submission_logs.loader_data.get("submitter", "")
        self.division = self.submission_logs.loader_data.get("division", "")
        self.measurements_dir = self.submission_logs.loader_data.get("measurements_dir", "")
        self.config = config
        self.setup_checks()

    def setup_checks(self):
        self.checks.append(self.missing_check)
        self.checks.append(self.required_files_check)
        self.checks.append(self.required_fields_check)
    

    def missing_check(self):
        if self.measurements_json is None:
            self.log.error(
                "%s measurements json file not found",
                self.path
            )
            return False
        return True
    
    def required_files_check(self):
        is_valid = True
        files = list_files(self.measurements_dir)
        for i in REQUIRED_MEASURE_FILES:
            if i not in files:
                self.log.error("%s is missing %s", self.measurements_dir, i)
                is_valid = False
            elif not self.config.skip_empty_files_check and (
                os.stat(os.path.join(self.measurements_dir, i)).st_size == 0
            ):
                self.log.error("%s is having empty %s", self.measurements_dir, i)
                is_valid = False
        return is_valid

    def required_fields_check(self):
        is_valid = True
        check_empty_fields = False if self.config.skip_meaningful_fields_emptiness_check else True
        for k in SYSTEM_IMP_REQUIRED_FILES:
            if k not in self.measurements_json:
                is_valid = False
                self.log.error("%s, field %s is missing", self.path, k)
            elif check_empty_fields and not self.measurements_json[k]:
                is_valid = False
                self.log.error(
                    "%s, field %s is missing meaningful value", self.path, k)
        return is_valid