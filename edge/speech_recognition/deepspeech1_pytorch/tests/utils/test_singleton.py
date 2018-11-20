import pytest

from deepspeech.utils.singleton import Singleton
from deepspeech.utils.singleton import SingletonNotExistError
from deepspeech.utils.singleton import SingletonRefsExistError


@pytest.fixture
def singleton_cls(request):
    """Returns a unique Singleton class with `check_args=True` for each request.

    A Singleton class defined at the module level would be shared by all tests
    making them dependent on each other due to it's stateful nature. Using a
    unique class per test ensures isolation.
    """
    def __init__(self, val):
        self.val = val

    cls_name = request.function.__name__ + '_singleton'
    cls = Singleton(cls_name, (), {'__init__': __init__}, check_args=True)

    return cls


def test_singleton_created_once(singleton_cls):
    a = singleton_cls(3)
    b = singleton_cls(3)
    assert a is b


def test_singleton_check_args_matching_ok(singleton_cls):
    a = singleton_cls(5)
    b = singleton_cls(val=5)
    assert a is b


def test_singleton_check_args_nonmatching_raises_value_error(singleton_cls):
    singleton_cls(val=5)
    with pytest.raises(ValueError):
        singleton_cls(6)


def test_get_singleton_raises_error_when_not_exist(singleton_cls):
    with pytest.raises(SingletonNotExistError):
        singleton_cls.get_singleton()


def test_get_singleton_returns_singleton(singleton_cls):
    a = singleton_cls(4)
    b = singleton_cls.get_singleton()
    assert a is b


def test_get_or_init_singleton_creates_singleton_when_not_exist(singleton_cls):
    a = singleton_cls.get_or_init_singleton(val=10)
    assert a.val == 10


def test_get_or_init_singleton_returns_singleton_when_exist(singleton_cls):
    a = singleton_cls.get_or_init_singleton(val=10)
    b = singleton_cls.get_or_init_singleton(val=15)
    assert a is b
    assert a.val == b.val == 10


def test_reset_singleton_removes_class_ref_when_no_other_refs(singleton_cls):
    singleton_cls(val=5)  # create Singleton instance but keep no reference

    # ref to Singleton instance in class should be deleted
    singleton_cls._reset_singleton()

    with pytest.raises(SingletonNotExistError):
        assert singleton_cls.get_singleton()

    # should be able to create a new Singleton instance
    a = singleton_cls(val=10)

    b = singleton_cls.get_singleton()
    assert a is b


def test_reset_singleton_raises_error_when_other_refs(singleton_cls):
    # create Singleton instance and _keep_ reference
    a = singleton_cls(val=10)  # noqa: F841
    with pytest.raises(SingletonRefsExistError):
        singleton_cls._reset_singleton()
