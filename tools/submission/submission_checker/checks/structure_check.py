from .base import BaseCheck


class StructureCheck(BaseCheck):
    """Simple Sample Check to test the structure of the checker. Not used in actual checking."""
    def __init__(self, log, path, parsed_log):
        """Initialize Sample Checker.

        Args:
            log (str): Path to log
            path (str): Path to results
            parsed_log (str): Parsed log
        """
        super().__init__(log, path)
        self.parsed_log = parsed_log
        self.checks.append(self.sample_check)

    def sample_check(self):
        """Simple check that always returns true.

        Returns:
            bool: True
        """
        return True
