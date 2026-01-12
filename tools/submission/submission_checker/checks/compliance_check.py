
from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
from .performance_check import PerformanceCheck
from .accuracy_check import AccuracyCheck
from ..utils import *
import re
import os


class ComplianceCheck(BaseCheck):
    def __init__(self, log, path, config: Config,
                 submission_logs: SubmissionLogs):
        super().__init__(log, path)
        self.submission_logs = submission_logs
        self.config = config
        self.model = self.submission_logs.loader_data.get("benchmark", "")
        self.model_mapping = self.submission_logs.loader_data.get(
            "model_mapping", {})
        self.compliance_dir = self.submission_logs.loader_data.get(
            "compliance_path", {})
        self.division = self.submission_logs.loader_data.get("division", "")
        self.model = self.config.get_mlperf_model(
            self.model, self.model_mapping)
        self.test_list = self.get_test_list(self.model)
        self.setup_checks()

    def setup_checks(self):
        self.checks.append(self.dir_exists_check)
        self.checks.append(self.performance_check)
        self.checks.append(self.accuracy_check)
        self.checks.append(self.compliance_performance_check)

    def get_test_list(self, model):
        test_list = []
        if model in self.config.base["models_TEST01"]:
            test_list.append("TEST01")
        if model in self.config.base["models_TEST04"]:
            test_list.append("TEST04")
        if model in self.config.base["models_TEST06"]:
            test_list.append("TEST06")
        return test_list

    def dir_exists_check(self):
        if self.division.lower() == "open":
            self.log.info(
                "Compliance tests not needed for open division. Skipping tests on %s",
                self.path)
            return True
        is_valid = True
        for test in self.test_list:
            test_dir = os.path.join(self.compliance_dir, test)
            acc_path = os.path.join(
                self.compliance_dir, test, "verify_accuracy.txt")
            perf_comp_path = os.path.join(
                self.compliance_dir, test, "verify_performance.txt")
            perf_path = os.path.join(
                self.compliance_dir,
                test,
                "performance",
                "run_1",
                "mlperf_log_detail.txt")
            if not os.path.exists(test_dir):
                self.log.error(
                    "Missing %s in compliance dir %s",
                    test,
                    self.compliance_dir)
                is_valid = False
            if test in ["TEST01", "TEST06"]:
                if not os.path.exists(acc_path):
                    self.log.error(
                        "Missing accuracy file in compliance dir. Needs file %s", acc_path)
                    is_valid = False
            if test in ["TEST01", "TEST04"]:
                if not os.path.exists(perf_comp_path):
                    self.log.error(
                        "Missing performance file in compliance dir. Needs file %s",
                        perf_comp_path)
                    is_valid = False
                if not os.path.exists(perf_path):
                    self.log.error(
                        "Missing perfomance file in compliance dir. Needs file %s", perf_path)
                    is_valid = False
        return is_valid

    def performance_check(self):
        if self.division.lower() == "open":
            self.log.info(
                "Compliance tests not needed for open division. Skipping tests on %s",
                self.path)
            return True
        is_valid = True
        for test in self.test_list:
            if test in ["TEST01", "TEST04"]:
                test_data = {
                    "division": self.submission_logs.loader_data.get("division", ""),
                    "benchmark": self.submission_logs.loader_data.get("benchmark", ""),
                    "scenario": self.submission_logs.loader_data.get("scenario", ""),
                    "model_mapping": self.submission_logs.loader_data.get("model_mapping", {})
                }
                test_logs = SubmissionLogs(
                    self.submission_logs.loader_data[f"{test}_perf_log"], None, None, None, self.submission_logs.system_json, None, test_data)
                perf_check = PerformanceCheck(self.log, os.path.join(
                    self.compliance_dir, test), self.config, test_logs)
                is_valid &= perf_check()
        return is_valid

    def accuracy_check(self):
        if self.division.lower() == "open":
            self.log.info(
                "Compliance tests not needed for open division. Skipping tests on %s",
                self.path)
            return True
        is_valid = True
        for test in self.test_list:
            test_dir = os.path.join(self.compliance_dir, test)
            if test == "TEST01":
                lines = self.submission_logs.loader_data[f"{test}_acc_result"]
                lines = [line.strip() for line in lines]
                if "TEST PASS" in lines:
                    self.log.info(
                        "Compliance test accuracy check (deterministic mode) in %s passed",
                        test_dir,
                    )
                else:
                    self.log.info(
                        "Compliance test accuracy check (deterministic mode) in %s failed",
                        test_dir,
                    )
                    test_acc_path = os.path.join(test_dir, "accuracy")
                    if not os.path.exists(test_acc_path):
                        self.log.error(
                            "%s has no accuracy directory", test_dir)
                        is_valid = False
                    else:
                        diff = files_diff(
                            list_files(
                                test_acc_path), REQUIRED_TEST01_ACC_FILES,
                        )
                        if diff:
                            self.log.error(
                                "%s has file list mismatch (%s)",
                                test_acc_path,
                                diff)
                            is_valid = False
                        else:
                            target = self.config.get_accuracy_target(
                                self.model)
                            patterns, acc_targets, acc_types, acc_limits, up_patterns, acc_upper_limit = self.config.get_accuracy_values(
                                self.model)
                            acc_limit_check = True

                            acc_seen = [False for _ in acc_targets]
                            acc_baseline = {
                                acc_type: 0 for acc_type in acc_types}
                            acc_compliance = {
                                acc_type: 0 for acc_type in acc_types}
                            with open(
                                os.path.join(
                                    test_acc_path, "baseline_accuracy.txt"),
                                "r",
                                encoding="utf-8",
                            ) as f:
                                for line in f:
                                    for acc_type, pattern in zip(
                                            acc_types, patterns):
                                        m = re.match(pattern, line)
                                        if m:
                                            acc_baseline[acc_type] = float(
                                                m.group(1))
                            with open(
                                os.path.join(
                                    test_acc_path, "compliance_accuracy.txt"),
                                "r",
                                encoding="utf-8",
                            ) as f:
                                for line in f:
                                    for acc_type, pattern in zip(
                                            acc_types, patterns):
                                        m = re.match(pattern, line)
                                        if m:
                                            acc_compliance[acc_type] = float(
                                                m.group(1))
                            for acc_type in acc_types:
                                if acc_baseline[acc_type] == 0 or acc_compliance[acc_type] == 0:
                                    is_valid = False
                                    break
                                else:
                                    required_delta_perc = self.config.get_delta_perc(
                                        self.model, acc_type
                                    )
                                    delta_perc = (
                                        abs(
                                            1
                                            - acc_baseline[acc_type] /
                                            acc_compliance[acc_type]
                                        )
                                        * 100
                                    )
                                    if delta_perc <= required_delta_perc:
                                        is_valid = True
                                    else:
                                        self.log.error(
                                            "Compliance test accuracy check (non-deterministic mode) in %s failed",
                                            test_dir,
                                        )
                                        is_valid = False
                                        break
            elif test == "TEST06":
                lines = self.submission_logs.loader_data[f"{test}_acc_result"]
                lines = [line.strip() for line in lines]
                first_token_pass = (
                    "First token check pass: True" in lines
                    or "First token check pass: Skipped" in lines
                )
                eos_pass = "EOS check pass: True" in lines
                length_check_pass = "Sample length check pass: True" in lines
                is_valid &= (
                    first_token_pass and eos_pass and length_check_pass)
                if not is_valid:
                    self.log.error(
                        f"TEST06 accuracy check failed. first_token_check: {first_token_pass} eos_check: {eos_pass} length_check: {length_check_pass}."
                    )
            else:
                self.log.info(f"{test_dir} does not require accuracy check")
        return is_valid

    def compliance_performance_check(self):
        if self.division.lower() == "open":
            self.log.info(
                "Compliance tests not needed for open division. Skipping tests on %s",
                self.path)
            return True
        is_valid = True
        for test in self.test_list:
            test_dir = os.path.join(self.compliance_dir, test)
            if test in ["TEST01", "TEST04"]:
                fname = os.path.join(test_dir, "verify_performance.txt")
                if not os.path.exists(fname):
                    self.log.error("%s is missing in %s", fname, test_dir)
                    is_valid = False
                else:
                    with open(fname, "r") as f:
                        for line in f:
                            # look for: TEST PASS
                            if "TEST PASS" in line:
                                is_valid = True
                                break
                    if is_valid == False:
                        self.log.error(
                            "Compliance test performance check in %s failed",
                            test_dir)

                    # Check performance dir
                    test_perf_path = os.path.join(
                        test_dir, "performance", "run_1")
                    if not os.path.exists(test_perf_path):
                        self.log.error(
                            "%s has no performance/run_1 directory", test_dir)
                        is_valid = False
                    else:
                        diff = files_diff(
                            list_files(test_perf_path),
                            REQUIRED_COMP_PER_FILES,
                            ["mlperf_log_accuracy.json"],
                        )
                        if diff:
                            self.log.error(
                                "%s has file list mismatch (%s)",
                                test_perf_path,
                                diff)
                            is_valid = False
        return is_valid
