from abc import ABC, abstractmethod


class BaseCheck(ABC):
    """
    A generic check class meant to be inherited by concrete check implementations.
    Subclasses must implement the `run()` method.
    """

    def __init__(self, log, path):
        self.checks = []
        self.log = log
        self.path = path
        self.name = "base checks"
        pass

    def run_checks(self):
        """
        Execute the check.
        Must be implemented by subclasses.
        Should return a CheckResult instance.
        """
        valid = True
        errors = []
        for check in self.checks:
            try:
                v = self.execute(check)
                valid &= v
            except BaseException:
                valid &= False
                self.log.error(
                    "Execution occurred in running check %s. Running %s in %s",
                    self.path,
                    check.__name__,
                    self.__class__.__name__)
        return valid

    def execute(self, check):
        return check()

    def __call__(self):
        """Allows the check instance to be called like a function."""
        self.log.info("Starting %s for: %s", self.name, self.path)
        valid = self.run_checks()
        if valid:
            self.log.info("All %s checks passed for: %s", self.name, self.path)
        else:
            self.log.error(
                "Some %s Checks failed for: %s",
                self.name,
                self.path)
        return valid
