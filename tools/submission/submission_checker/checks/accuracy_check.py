from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
import re
import os


class AccuracyCheck(BaseCheck):
    """Checks accuracy-related submission artifacts and reports issues.

    The `AccuracyCheck` class performs a set of validations on submission
    accuracy outputs. It inspects the parsed MLPerf log and accompanying
    accuracy artifacts provided via `SubmissionLogs` and the test
    `Config` to ensure that reported accuracy metrics meet configured
    targets and limits, that the accuracy JSON exists and is properly
    truncated, that Loadgen did not report blocking errors, and that the
    accuracy run covered the expected dataset size.

    Main check methods:
    - `accuracy_result_check`: Parses `accuracy.txt` lines to validate
        reported metrics against targets, upper limits, and hash presence.
    - `accuracy_json_check`: Ensures the accuracy JSON file exists and is
        within allowed size limits.
    - `loadgen_errors_check`: Fails if Loadgen reported non-ignored errors.
    - `dataset_check`: Verifies the reported sample count matches the
        configured dataset size unless the check is skipped.

    Attributes:
            submission_logs (SubmissionLogs): Holder for submission log paths
                    and parsed contents (accuracy logs, results, json, loader data).
            mlperf_log: Parsed MLPerf log object used to inspect errors and
                    run metadata.
            accuracy_result (list[str]): Lines from `accuracy.txt` used to
                    extract reported accuracy values.
            accuracy_json (str): Path to the accuracy JSON file.
            config (Config): Configuration helper providing target values and
                    dataset sizes.
    """

    def __init__(
        self, log, path, config: Config, submission_logs: SubmissionLogs
    ):
        """Initialize the accuracy check helper.

        Args:
            log: Logger instance used to report messages.
            path: Path to the submission being checked.
            config (Config): Configuration provider for targets and limits.
            submission_logs (SubmissionLogs): Parsed submission logs and
                artifact paths (accuracy logs, results, json, loader data).
        """
        super().__init__(log, path)
        self.name = "accuracy checks"
        self.submission_logs = submission_logs
        self.mlperf_log = self.submission_logs.accuracy_log
        self.accuracy_result = self.submission_logs.accuracy_result
        self.accuracy_json = self. submission_logs.accuracy_json
        self.config = config
        self.model = self.submission_logs.loader_data.get("benchmark", "")
        self.model_mapping = self.submission_logs.loader_data.get(
            "model_mapping", {})
        self.model = self.config.get_mlperf_model(
            self.model, self.model_mapping)
        self.scenario_fixed = self.submission_logs.loader_data.get(
            "scenario", "")
        self.scenario = self.mlperf_log["effective_scenario"]
        self.division = self.submission_logs.loader_data.get("division", "")
        self.setup_checks()

    def setup_checks(self):
        """Register individual accuracy-related checks.

        Adds the per-submission validation callables to `self.checks` in
        the order they should be executed.
        """
        self.checks.append(self.accuracy_result_check)
        self.checks.append(self.accuracy_json_check)
        self.checks.append(self.loadgen_errors_check)
        self.checks.append(self.dataset_check)

    def accuracy_result_check(self):
        """Validate reported accuracy metrics in `accuracy.txt`.

        Parses lines from `self.accuracy_result` using configured patterns
        and compares found values against targets and optional upper
        limits. Also ensures a hash value is present and records the
        observed accuracy metrics in `submission_logs.loader_data`.

        Returns:
            bool: True if accuracy checks passed (or division is 'open'),
                False otherwise.
        """

        patterns, acc_targets, acc_types, acc_limits, up_patterns, acc_upper_limit = self.config.get_accuracy_values(
            self.model
        )
        acc = None
        hash_val = None
        result_acc = {}
        acc_limit_check = True
        all_accuracy_valid = True
        acc_seen = [False for _ in acc_targets]
        for line in self.accuracy_result:
            for i, (pattern, acc_target, acc_type) in enumerate(
                zip(patterns, acc_targets, acc_types)
            ):
                m = re.match(pattern, line)
                if m:
                    acc = m.group(1)
                m = re.match(r"^hash=([\w\d]+)$", line)
                if m:
                    hash_val = m.group(1)
                if acc is not None and float(acc) >= acc_target:
                    all_accuracy_valid &= True
                    acc_seen[i] = True
                elif acc is not None:
                    all_accuracy_valid = False
                    self.log.warning(
                        "%s accuracy not met: expected=%f, found=%s",
                        self.path,
                        acc_target,
                        acc,
                    )
                if acc:
                    result_acc[acc_type] = acc
                acc = None

            if acc_upper_limit is not None:
                for i, (pattern, acc_limit) in enumerate(
                        zip(up_patterns, acc_limits)):
                    m = re.match(pattern, line)
                    if m:
                        acc = m.group(1)
                    m = re.match(r"^hash=([\w\d]+)$", line)
                    if m:
                        hash_val = m.group(1)
                    if (
                        acc is not None
                        and acc_upper_limit is not None
                        and float(acc) > acc_limit
                    ):
                        acc_limit_check = False
                        self.log.warning(
                            "%s accuracy not met: upper limit=%f, found=%s",
                            self.path,
                            acc_limit,
                            acc,
                        )
                    acc = None
            if all(acc_seen) and hash_val:
                break
        is_valid = all_accuracy_valid & all(acc_seen)
        if acc_upper_limit is not None:
            is_valid &= acc_limit_check
        if not hash_val:
            self.log.error("%s not hash value for accuracy.txt", self.path)
            is_valid = False
        self.submission_logs.loader_data["accuracy_metrics"] = result_acc
        if self.division.lower() == "open":
            return True
        return is_valid

    def accuracy_json_check(self):
        """Check that the accuracy JSON exists and is within size limits.

        Returns:
            bool: True if the JSON file exists and its size does not
                exceed `MAX_ACCURACY_LOG_SIZE`, False otherwise.
        """
        if not os.path.exists(self.accuracy_json):
            self.log.error("%s is missing", self.accuracy_json)
            return False
        else:
            if os.stat(self.accuracy_json).st_size > MAX_ACCURACY_LOG_SIZE:
                self.log.error("%s is not truncated", self.accuracy_json)
                return False
        return True

    def loadgen_errors_check(self):
        """Detect Loadgen errors reported in the MLPerf log.

        If errors are present and not ignored by configuration, logs the
        error messages and returns False to indicate failure.

        Returns:
            bool: True if no blocking Loadgen errors are present,
                False otherwise.
        """
        if self.mlperf_log.has_error():
            if self.config.ignore_uncommited:
                has_other_errors = False
                for error in self.mlperf_log.get_errors():
                    if "Loadgen built with uncommitted changes!" not in error["value"]:
                        has_other_errors = True
            self.log.error("%s contains errors:", self.path)
            for error in self.mlperf_log.get_errors():
                self.log.error("%s", error["value"])

            if not self.config.ignore_uncommited or has_other_errors:
                self.log.error(
                    "%s has loadgen errors, number of errors: %s", self.path, self.mlperf_log.num_errors()
                )
                return False
        return True

    def dataset_check(self):
        """Verify the accuracy run covered the expected dataset size.

        If `skip_dataset_size_check` is enabled in the configuration,
        this check is skipped and returns True. Otherwise compares the
        `qsl_reported_total_count` from the MLPerf log to the expected
        dataset size for the model.

        Returns:
            bool: True if the dataset sizes match or the check is skipped,
                False if the reported count differs from expected.
        """
        if self.config.skip_dataset_size_check:
            self.log.info(
                "%s Skipping dataset size check", self.path
            )
            return True
        expected_qsl_total_count = self.config.get_accuracy_sample_count(
            self.model)
        if "effective_accuracy_sample_count" in self.mlperf_log.get_keys():
            qsl_total_count = self.mlperf_log["effective_accuracy_sample_count"]
        else:
            qsl_total_count = self.mlperf_log["qsl_reported_total_count"]

        if qsl_total_count != expected_qsl_total_count:
            self.log.error(
                "%s accurcy run does not cover all dataset, accuracy samples: %s, dataset size: %s", self.path, qsl_total_count, expected_qsl_total_count
            )
            return False
        return True
