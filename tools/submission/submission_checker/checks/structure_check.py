from .base import BaseCheck


class StructureCheck(BaseCheck):
    def __init__(self, log, path, parsed_log):
        super().__init__(log, path)
        self.parsed_log = parsed_log
        self.checks.append(self.sample_check)

    def sample_check(self):
        return True
