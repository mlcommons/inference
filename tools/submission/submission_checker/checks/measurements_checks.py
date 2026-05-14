from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
from ..utils import *
import os


class MeasurementsCheck(BaseCheck):
    """Validate measurement artifacts included in a submission.

    The `MeasurementsCheck` class verifies the presence and basic
    correctness of measurement-related files and fields produced by a
    submission. It ensures the measurements JSON exists, required files
    are present (and optionally non-empty), the source directory exists,
    and that required metadata fields inside the measurements JSON have
    meaningful values.

    Attributes:
        submission_logs (SubmissionLogs): Parsed submission logs and
            metadata used to locate measurement artifacts.
        measurements_json (dict): Parsed contents of the measurements JSON.
        measurements_dir (str): Path to the measurements directory to
            validate file contents.
        src_dir (str): Path to the submission source directory expected to
            be present in the submission bundle.
        config (Config): Configuration provider toggling optional checks.
    """

    def __init__(self, log, path, config: Config,
                 submission_logs: SubmissionLogs):
        """Initialize the measurements checker.

        Args:
            log: Logger used to emit info/warning/error messages.
            path: Path to the submission root being validated.
            config (Config): Configuration helper containing feature
                toggles for skipped checks.
            submission_logs (SubmissionLogs): Parsed submission artifacts
                and loader metadata.
        """
        super().__init__(log, path)
        self.name = "measurement checks"
        self.submission_logs = submission_logs
        self.measurements_json = self.submission_logs.measurements_json
        self.submitter = self.submission_logs.loader_data.get("submitter", "")
        self.division = self.submission_logs.loader_data.get("division", "")
        self.measurements_dir = self.submission_logs.loader_data.get(
            "measurements_dir", "")
        self.src_dir = self.submission_logs.loader_data.get("src_path", "")
        self.config = config
        self.setup_checks()

    def setup_checks(self):
        """Register per-submission measurement checks.

        Appends the callable checks to `self.checks` in the order they
        should be executed by the submission validation framework.
        """
        self.checks.append(self.missing_check)
        self.checks.append(self.directory_exist_check)
        self.checks.append(self.required_files_check)
        self.checks.append(self.required_fields_check)

    def missing_check(self):
        """Ensure a measurements JSON was provided.

        Returns:
            bool: True if `measurements_json` is present, False otherwise.
        """
        if self.measurements_json is None:
            self.log.error(
                "%s measurements json file not found",
                self.path
            )
            return False
        return True

    def directory_exist_check(self):
        """Verify the expected source directory exists in the submission.

        Returns:
            bool: True if `src_dir` exists, False otherwise.
        """
        if not os.path.exists(self.src_dir):
            self.log.error(
                "%s src directory does not exist",
                self.src_dir
            )
            return False
        return True

    def required_files_check(self):
        """Confirm required measurement files exist and are non-empty.

        Respects the `skip_empty_files_check` configuration flag; when that
        flag is False, files with zero size will cause the check to fail.

        Returns:
            bool: True if all required files are present (and non-empty when
                configured), False otherwise.
        """
        is_valid = True
        files = list_files(self.measurements_dir)
        for i in REQUIRED_MEASURE_FILES:
            if i not in files:
                self.log.error("%s is missing %s", self.measurements_dir, i)
                is_valid = False
            elif not self.config.skip_empty_files_check and (
                os.stat(os.path.join(self.measurements_dir, i)).st_size == 0
            ):
                self.log.error(
                    "%s is having empty %s",
                    self.measurements_dir,
                    i)
                is_valid = False
        return is_valid

    def required_fields_check(self):
        """Validate presence and meaningfulness of required JSON fields.

        If `skip_meaningful_fields_emptiness_check` is False in the
        configuration, this will also fail when required fields are empty.

        Returns:
            bool: True if all required fields exist (and contain meaningful
                values when configured), False otherwise.
        """
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
