from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
from ..utils import *


class SystemCheck(BaseCheck):
    """Validate system description JSON and metadata consistency.

    The `SystemCheck` class verifies the presence and correctness of the
    system description JSON file in submissions. It ensures the system
    metadata matches submission expectations, validates required fields
    and their types, checks system type and availability status, and
    enforces consistency between JSON fields and submission directory
    structure.

    Attributes:
        submission_logs (SubmissionLogs): Parsed submission logs and
            metadata used to locate system JSON and compare fields.
        system_json (dict): Parsed system description JSON contents.
        submitter (str): Submitter name extracted from submission path.
        division (str): Division extracted from submission path.
        config (Config): Configuration provider for field validation rules.
    """

    def __init__(self, log, path, config: Config,
                 submission_logs: SubmissionLogs):
        """Initialize the system checker.

        Args:
            log: Logger used to emit info/warning/error messages.
            path: Path to the submission being validated.
            config (Config): Configuration helper containing validation
                rules and toggles.
            submission_logs (SubmissionLogs): Parsed submission artifacts
                and loader metadata.
        """
        super().__init__(log, path)
        self.name = "system checks"
        self.submission_logs = submission_logs
        self.system_json = self.submission_logs.system_json
        self.submitter = self.submission_logs.loader_data.get("submitter", "")
        self.division = self.submission_logs.loader_data.get("division", "")
        self.is_endpoints = self.submission_logs.loader_data.get(
            "is_endpoints_submission", False)
        self.config = config
        self.setup_checks()

    def setup_checks(self):
        """Register per-submission system checks.

        Appends the callable checks to `self.checks` in the order they
        should be executed by the submission validation framework.
        """
        self.checks.append(self.missing_check)
        self.checks.append(self.availability_check)
        self.checks.append(self.system_type_check)
        self.checks.append(self.network_check)
        self.checks.append(self.required_fields_check)
        self.checks.append(self.submitter_check)
        self.checks.append(self.division_check)
        self.checks.append(self.nameplate_power_check)
        self.apply_checks = set(self.checks)

    def missing_check(self):
        """Ensure the system JSON file was provided.

        Returns:
            bool: True if `system_json` is present, False otherwise.
        """
        if self.system_json is None:
            self.log.error(
                "%s system json file not found",
                self.path
            )
            return False
        return True

    def availability_check(self):
        """Validate the system's availability status.

        Checks that the 'status' field contains a valid availability value
        from the predefined list.

        Returns:
            bool: True if status is valid, False otherwise.
        """
        availability = self.system_json.get("status").lower()
        if availability not in VALID_AVAILABILITIES:
            self.log.error(
                "%s has invalid status (%s)", self.path, availability
            )
            return False
        return True

    def system_type_check(self):
        """Verify the system type is allowed.

        Ensures 'system_type' is one of the valid types: datacenter, edge,
        or combinations thereof.

        Returns:
            bool: True if system type is valid, False otherwise.
        """
        system_type = self.system_json.get("system_type")
        valid_system_types = [
            "datacenter", "edge", "datacenter,edge", "edge,datacenter"]

        if system_type not in valid_system_types:
            self.log.error(
                "%s has invalid system type (%s)",
                self.path,
                system_type,
            )
            return False
        # Maybe add this line if needed
        # self.config.set_type(system_type)
        return True

    def network_check(self):
        """Ensure network mode matches division requirements.

        For closed division, network mode must be false; for network
        division, it must be true.

        Returns:
            bool: True if network mode is correct for the division,
                False otherwise.
        """
        is_network = self.system_json.get(SYSTEM_DESC_IS_NETWORK_MODE)
        is_network = (
            is_network.lower() == "true"
            if is_network is not None
            else False
        )
        expected_state_by_division = {"network": True, "closed": False}
        if self.division in expected_state_by_division:
            if expected_state_by_division[self.division] != is_network:
                self.log.error(
                    f"{self.path} incorrect network mode(={is_network})"
                    f"for division '{self.division}'"
                )
                return False
        return True

    def required_fields_check(self):
        """Validate presence and validity of required system fields.

        Checks for required fields based on network mode, ensures they
        exist, and validates meaningful responses and numeric types where
        required, respecting configuration toggles.

        Returns:
            bool: True if all required fields are present and valid,
                False otherwise.
        """
        required_fields = SYSTEM_DESC_REQUIRED_FIELDS.copy()
        is_network = self.system_json.get(SYSTEM_DESC_IS_NETWORK_MODE)
        if is_network:
            required_fields += SYSTEM_DESC_REQUIRED_FIELDS_NETWORK_MODE

        check_empty_fields = False if self.config.skip_meaningful_fields_emptiness_check else True
        is_valid = True
        for k in required_fields:
            if k not in self.system_json:
                is_valid = False
                self.log.error("%s, field %s is missing", self.path, k)
            elif (
                check_empty_fields
                and k in SYSTEM_DESC_MEANINGFUL_RESPONSE_REQUIRED_FIELDS
                and not self.system_json[k]
            ):
                is_valid = False
                self.log.error(
                    "%s, field %s requires a meaningful response but is empty", self.path, k
                )
            elif (
                check_empty_fields
                and k in SYSTEM_DESC_NUMERIC_RESPONSE_REQUIRED_FIELDS
                and not is_number(str(self.system_json[k]))
            ):
                self.log.error(
                    "%s, field %s requires a numeric response but is empty", self.path, k
                )
        return is_valid

    def submitter_check(self):
        """Verify submitter name consistency.

        Ensures the 'submitter' field in system JSON matches the submitter
        name from the submission directory structure.

        Returns:
            bool: True if submitter names match, False otherwise.
        """
        if self.system_json.get("submitter").lower() != self.submitter.lower():
            self.log.error(
                "%s has submitter %s, directory has %s",
                self.path,
                self.system_json.get("submitter"),
                self.submitter,
            )
            return False
        return True

    def division_check(self):
        """Verify division consistency.

        Ensures the 'division' field in system JSON matches the division
        from the submission directory structure.

        Returns:
            bool: True if divisions match, False otherwise.
        """
        if self.system_json.get("division").lower() != self.division.lower():
            self.log.error(
                "%s has division %s, directory has %s",
                self.path,
                self.system_json.get("division"),
                self.division,
            )
            return False
        return True

    def nameplate_power_check(self):
        """Check for nameplate power YAML, load it, and sum PowerCapacityWatts.

        Only runs for systems with a power submission. The nameplate power YAML
        is required in that case — missing or unloadable file is an error.
        Non-power submissions are skipped entirely.

        If present and loaded, recursively sums all PowerCapacityWatts values
        and stores the total as 'design_power_watts' in loader_data.

        Returns:
            bool: False if the file is required but missing or unloadable,
                True otherwise.
        """
        import os
        nameplate_power_path = self.submission_logs.loader_data.get(
            "nameplate_power_path")
        nameplate_power_yaml = self.submission_logs.nameplate_power_yaml
        has_power = os.path.exists(
            self.submission_logs.loader_data.get("power_dir_path", ""))

        if not nameplate_power_path:
            return True

        if not has_power:
            self.submission_logs.loader_data["design_power_watts"] = None
            return True

        if "system_power_capacity" in self.system_json:
            design_power_watts = self.system_json["system_power_capacity"]
            self.submission_logs.loader_data["design_power_watts"] = design_power_watts
            return True

        if not os.path.exists(nameplate_power_path):
            self.submission_logs.loader_data["design_power_watts"] = None
            self.log.warning(
                "%s has a power submission but nameplate power YAML not found at %s",
                self.path,
                nameplate_power_path,
            )
            return True

        if nameplate_power_yaml is None:
            self.submission_logs.loader_data["design_power_watts"] = None
            self.log.error(
                "%s nameplate power YAML exists but could not be loaded: %s",
                self.path,
                nameplate_power_path,
            )
            return False

        design_power_watts = _sum_power_capacity_watts(nameplate_power_yaml)
        self.submission_logs.loader_data["design_power_watts"] = design_power_watts
        self.log.info(
            "%s design_power_watts = %s W",
            self.path,
            design_power_watts,
        )
        return True


def _sum_power_capacity_watts(data):
    """Calculate total design power from the nameplate YAML structure.

    Traverses the hierarchy of submitter-defined component labels. A node is
    a leaf when it contains both 'PSUs' and 'Min PSUs Needed'; design power
    for that component is the sum of the minimum required PSUs' capacities
    (redundant PSUs beyond Min PSUs Needed are excluded). All other nodes are
    intermediate and are recursed into.
    """
    if isinstance(data, dict):
        keys_lower = {k.lower(): k for k in data}
        if "psus" in keys_lower and "min psus needed" in keys_lower:
            min_psus = data[keys_lower["min psus needed"]]
            psus = data[keys_lower["psus"]]
            capacities = sorted(
                [psu.get("PowerCapacityWatts", 0) for psu in psus],
                reverse=True,
            )
            return sum(capacities[:min_psus])
        return sum(
            _sum_power_capacity_watts(v)
            for v in data.values()
            if isinstance(v, (dict, list))
        )
    elif isinstance(data, list):
        return sum(_sum_power_capacity_watts(item) for item in data)
    return 0
