
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
    """Validate compliance test artifacts for a submission.

    The `ComplianceCheck` class runs a set of validations against the
    compliance directory produced with a submission. It verifies the
    presence of required test subdirectories and files, runs delegated
    performance and accuracy checks for compliance tests, and inspects
    compliance-specific performance outputs.

    The class delegates some checks to `PerformanceCheck` and
    `AccuracyCheck` helpers when relevant. Results and file lists are
    logged via the provided logger.
    """

    def __init__(self, log, path, config: Config,
                 submission_logs: SubmissionLogs):
        """Initialize the compliance checker.

        Args:
            log: Logger used to emit informational, warning, and error
                messages about the compliance checks.
            path: Filesystem path to the submission root being checked.
            config (Config): Configuration provider for models and
                compliance expectations.
            submission_logs (SubmissionLogs): Parsed submission log
                artifacts and loader metadata.
        """
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
        """Register the sequence of compliance checks to run.

        Appends the per-submission validation callables to `self.checks` in
        the order they should be executed by the checking framework.
        """
        self.checks.append(self.dir_exists_check)
        self.checks.append(self.performance_check)
        self.checks.append(self.accuracy_check)
        self.checks.append(self.compliance_performance_check)

    def get_test_list(self, model):
        """Return the list of compliance tests applicable to `model`.

        The mapping of models to tests is read from the configuration
        (`self.config.base`) using the pre-defined keys
        `models_TEST01`, `models_TEST04`, `models_TEST06`, and `models_TEST08`.

        Args:
            model (str): MLPerf benchmark/model identifier.

        Returns:
            list[str]: Ordered list of compliance test names to execute.
        """

        test_list = []
        if model in self.config.base.get("models_TEST01", []):
            test_list.append("TEST01")
        if model in self.config.base.get("models_TEST04", []):
            test_list.append("TEST04")
        if model in self.config.base.get("models_TEST06", []):
            test_list.append("TEST06")
        if model in self.config.base.get("models_TEST07", []):
            test_list.append("TEST07")
        if model in self.config.base.get("models_TEST09", []):
            test_list.append("TEST09")
        if model in self.config.base.get("models_TEST08", []):
            test_list.append("TEST08")
        if model in self.config.base.get("models_TEST07", []):
            test_list.append("TEST07")
        if model in self.config.base.get("models_TEST09", []):
            test_list.append("TEST09")
        return test_list

    def dir_exists_check(self):
        """Verify required compliance directories and files exist.

        Skips checks for the 'open' division. For each test in
        `self.test_list`, ensures the expected test directory exists and
        that required verification files are present depending on the
        test type (accuracy/performance files for specific tests).

        Returns:
            bool: True if all required files and directories are present,
                False otherwise.
        """

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
            # TEST01, TEST06, TEST07 and TEST08 require verify_accuracy.txt
            if test in ["TEST01", "TEST06", "TEST07", "TEST08"]:
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
            if test == "TEST09":
                output_len_path = os.path.join(
                    self.compliance_dir, test, "verify_output_len.txt")
                if not os.path.exists(output_len_path):
                    self.log.error(
                        "Missing output length verification file in compliance dir. Needs file %s",
                        output_len_path)
                    is_valid = False
        return is_valid

    def performance_check(self):
        """Run performance compliance checks for applicable tests.

        For each test that requires a performance check (TEST01 and
        TEST04), construct a `SubmissionLogs` object pointing at the
        test's performance log and delegate to `PerformanceCheck`.

        Returns:
            bool: True if all delegated performance checks pass, False
                if any fail.
        """

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
        """Run accuracy compliance checks for applicable tests.

        For TEST01, verifies deterministic-mode pass lines and checks the
        `accuracy` directory contents and baseline/compliance accuracy
        values against model-specific delta thresholds.

        For TEST06, inspects the pre-parsed result lines for first-token,
        EOS, and sample-length checks.

        Returns:
            bool: True if all required accuracy checks pass, False
                otherwise.
        """

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
            elif test == "TEST07":
                # TEST07: Verify accuracy in performance mode
                # Check verify_accuracy.txt for TEST PASS
                acc_path = os.path.join(test_dir, "verify_accuracy.txt")
                if os.path.exists(acc_path):
                    with open(acc_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if "TEST PASS" in content:
                        self.log.info(
                            "TEST07 accuracy check in %s passed", test_dir)
                    else:
                        self.log.error(
                            "TEST07 accuracy check in %s failed", test_dir)
                        is_valid = False
                else:
                    self.log.error(
                        "TEST07 verify_accuracy.txt missing in %s", test_dir)
                    is_valid = False
            elif test == "TEST09":
                # TEST09: Verify output token length in performance mode
                # Check verify_output_len.txt for TEST PASS
                output_len_path = os.path.join(
                    test_dir, "verify_output_len.txt")
                if os.path.exists(output_len_path):
                    with open(output_len_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if "TEST PASS" in content:
                        self.log.info(
                            "TEST09 output length check in %s passed", test_dir)
                    else:
                        self.log.error(
                            "TEST09 output length check in %s failed", test_dir)
                        is_valid = False
                else:
                    self.log.error(
                        "TEST09 verify_output_len.txt missing in %s", test_dir)
                    is_valid = False
            elif test == "TEST08":
                # TEST08 is used for dlrm-v3 streaming dataset compliance
                # It verifies that NE values match between accuracy and
                # performance runs
                lines = self.submission_logs.loader_data.get(
                    f"{test}_acc_result")
                if lines is None:
                    self.log.error(
                        "TEST08 accuracy result file not found for %s", test_dir)
                    is_valid = False
                else:
                    lines = [line.strip() for line in lines]
                    if "TEST PASS" in lines:
                        self.log.info(
                            "Compliance test TEST08 accuracy check in %s passed",
                            test_dir,
                        )
                    else:
                        self.log.error(
                            "Compliance test TEST08 accuracy check in %s failed. "
                            "Expected 'TEST PASS' in verify_accuracy.txt",
                            test_dir,
                        )
                        is_valid = False
            elif test == "TEST07":
                # TEST07: Verify accuracy in performance mode
                # Check verify_accuracy.txt for TEST PASS
                acc_path = os.path.join(test_dir, "verify_accuracy.txt")
                if os.path.exists(acc_path):
                    with open(acc_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if "TEST PASS" in content:
                        self.log.info(
                            "TEST07 accuracy check in %s passed", test_dir)
                    else:
                        self.log.error(
                            "TEST07 accuracy check in %s failed", test_dir)
                        is_valid = False
                else:
                    self.log.error(
                        "TEST07 verify_accuracy.txt missing in %s", test_dir)
                    is_valid = False
            elif test == "TEST09":
                # TEST09: Verify output token length in performance mode
                # Check verify_output_len.txt for TEST PASS
                output_len_path = os.path.join(
                    test_dir, "verify_output_len.txt")
                if os.path.exists(output_len_path):
                    with open(output_len_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if "TEST PASS" in content:
                        self.log.info(
                            "TEST09 output length check in %s passed", test_dir)
                    else:
                        self.log.error(
                            "TEST09 output length check in %s failed", test_dir)
                        is_valid = False
                else:
                    self.log.error(
                        "TEST09 verify_output_len.txt missing in %s", test_dir)
                    is_valid = False
            else:
                self.log.info(f"{test_dir} does not require accuracy check")
        return is_valid

    def compliance_performance_check(self):
        """Inspect compliance performance verification outputs.

        For TEST01 and TEST04, checks the `verify_performance.txt` file for
        a passing indicator and ensures the `performance/run_1` directory
        contains the expected files (with optional exclusions).

        Returns:
            bool: True if all compliance performance checks pass, False
                if any check fails.
        """

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
