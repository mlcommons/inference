from abc import ABC, abstractmethod


class BaseCheck(ABC):
    """
    A generic check class meant to be inherited by concrete check implementations.
    Subclasses must register their check methods into `self.checks`.
    """

    def __init__(self, log, path):
        self.checks = []
        self.log = log
        self.path = path
        self.name = "base checks"
        pass

    def run_checks(self):
        """
        Execute all registered checks. Returns True if all checks pass, False otherwise.
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
                    "Execution occurred in running check {check_name}. Running {check_name} in {class_name}",
                    path=self.path,
                    check_name=check.__name__,
                    class_name=self.__class__.__name__)
        return valid

    def execute(self, check):
        """Custom execution of a single check method."""
        return check()

    def __call__(self):
        """Allows the check instance to be called like a function."""
        self.log.info("Starting {name} for: {path}", name=self.name, path=self.path)
        valid = self.run_checks()
        if valid:
            self.log.info("All {name} checks passed for: {path}", name=self.name, path=self.path)
        else:
            self.log.error(
                "Some {name} Checks failed for: {path}",
                name=self.name,
                path=self.path)
        return valid
