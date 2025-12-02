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
            v = self.execute(check)
            valid &= v
            if not valid:
                return False
        return valid
            
    def execute(self, check):
        return check()
    
    def __call__(self):
        """Allows the check instance to be called like a function."""
        self.log("Starting check...")
        valid = self.run_checks()
        if valid:
            self.log.info("Checks passed")
        else:
            self.log.error("%s Checks failed", self.path)
        return valid