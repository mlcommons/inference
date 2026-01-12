from .base import BaseCheck
from ..constants import *
from ..loader import SubmissionLogs
from ..configuration.configuration import Config
from ..utils import *


class SystemCheck(BaseCheck):
    def __init__(self, log, path, config: Config,
                 submission_logs: SubmissionLogs):
        super().__init__(log, path)
        self.name = "system checks"
        self.submission_logs = submission_logs
        self.system_json = self.submission_logs.system_json
        self.submitter = self.submission_logs.loader_data.get("submitter", "")
        self.division = self.submission_logs.loader_data.get("division", "")
        self.config = config
        self.setup_checks()

    def setup_checks(self):
        self.checks.append(self.missing_check)
        self.checks.append(self.availability_check)
        self.checks.append(self.system_type_check)
        self.checks.append(self.network_check)
        self.checks.append(self.required_fields_check)
        self.checks.append(self.submitter_check)
        self.checks.append(self.division_check)

    def missing_check(self):
        if self.system_json is None:
            self.log.error(
                "%s system json file not found",
                self.path
            )
            return False
        return True

    def availability_check(self):
        availability = self.system_json.get("status").lower()
        if availability not in VALID_AVAILABILITIES:
            self.log.error(
                "%s has invalid status (%s)", self.path, availability
            )
            return False
        return True

    def system_type_check(self):
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
                    f"{
                        self.path} incorrect network mode (={is_network}) for division '{
                        self.division}'"
                )
                return False
        return True

    def required_fields_check(self):
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
        if self.system_json.get("division").lower() != self.division.lower():
            self.log.error(
                "%s has division %s, directory has %s",
                self.path,
                self.system_json.get("division"),
                self.division,
            )
            return False
        return True
