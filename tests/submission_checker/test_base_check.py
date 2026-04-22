import logging
import pytest
from submission_checker.checks.base import BaseCheck


log = logging.getLogger("test")


class AlwaysPassCheck(BaseCheck):
    def __init__(self):
        super().__init__(log, "/fake/path")
        self.checks = [self.check_a, self.check_b]

    def check_a(self):
        return True

    def check_b(self):
        return True


class SomeFailCheck(BaseCheck):
    def __init__(self):
        super().__init__(log, "/fake/path")
        self.checks = [self.pass_check, self.fail_check]

    def pass_check(self):
        return True

    def fail_check(self):
        return False


class ExceptionCheck(BaseCheck):
    def __init__(self):
        super().__init__(log, "/fake/path")
        self.checks = [self.boom]

    def boom(self):
        raise RuntimeError("intentional failure")


class EmptyCheck(BaseCheck):
    def __init__(self):
        super().__init__(log, "/fake/path")
        self.checks = []


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------

class TestRunChecks:
    def test_all_pass_returns_true(self):
        assert AlwaysPassCheck().run_checks() is True

    def test_any_fail_returns_false(self):
        assert SomeFailCheck().run_checks() is False

    def test_exception_treated_as_failure(self):
        assert ExceptionCheck().run_checks() is False

    def test_no_checks_returns_true(self):
        assert EmptyCheck().run_checks() is True


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------

class TestCall:
    def test_callable_returns_true_when_all_pass(self):
        assert AlwaysPassCheck()() is True

    def test_callable_returns_false_when_any_fail(self):
        assert SomeFailCheck()() is False


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

def test_execute_delegates_to_check_method():
    checker = AlwaysPassCheck()
    assert checker.execute(checker.check_a) is True


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def test_path_stored():
    checker = AlwaysPassCheck()
    assert checker.path == "/fake/path"


def test_log_stored():
    checker = AlwaysPassCheck()
    assert checker.log is log
