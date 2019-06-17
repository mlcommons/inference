import inspect
import sys


class SingletonNotExistError(Exception):
    """Raised when getting a Singleton that does not exist.

    Args:
        singleton_cls_name: `__name__` of the Singleton class that has no
            instance.
        message (optional): Explanation of the error.
    """
    def __init__(self, singleton_cls_name, message=None):
        self.singleton_cls_name = singleton_cls_name
        self.message = message


class SingletonRefsExistError(Exception):
    """Raised when resetting Singleton class but refs to current instance exist.

    Args:
        singleton_cls_name: `__name__` of the Singleton class that still has
            refs to the current Singleton instance.
        message (optional): Explanation of the error.
    """
    def __init__(self, singleton_cls_name, message=None):
        self.singleton_cls_name = singleton_cls_name
        self.message = message


class Singleton(type):
    """A Singleton metaclass ensures at most one instance of a class exists.

    Args:
        check_args (optional, bool): If `True` when passed as a kwd (see
            example below) then it verifies that each call to the classes
            `__init__` method has the same set of arguments. If not, a
            `ValueError` is raised. Default: `False`.

    Example:

        >>> class Foo(metaclass=Singleton, check_args=True):
        ...     def __init__(self, val):
        ...         self.val = val
        ...
        >>> Foo.get_singleton()
        Traceback (most recent call last):
            ...
        deepspeech.utils.singleton.SingletonNotExistError: Foo
        >>> a = Foo(val=6)
        >>> b = Foo(6)
        >>> a is b
        True
        >>> Foo(val=8)
        Traceback (most recent call last):
            ...
        ValueError: Foo instance already exists but previously initialised
        differently...
        >>> c = Foo.get_singleton()
        >>> a is c
        True
        >>> d = Foo.get_or_init_singleton(val=8)  # `check_args` skipped!
        >>> a is d
        True
    """
    def __new__(metacls, name, bases, namespace, **kwds):
        cls_methods = ['get_singleton',
                       'get_or_init_singleton',
                       '_reset_singleton']
        for cls_method in cls_methods:
            namespace[cls_method] = classmethod(getattr(metacls, cls_method))
        return type.__new__(metacls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwds):
        cls.__check_args = 'check_args' in kwds
        cls.__instance = None

    def get_singleton(cls):
        """Returns the Singleton instance if it exists.

        Raises:
            SingletonNotExistError: Singleton instance yet to be created.
        """
        if cls.__instance is None:
            raise SingletonNotExistError(cls.__name__)
        return cls.__instance

    def get_or_init_singleton(cls, *args, **kwargs):
        """Returns the Singleton instance if it exists else calls `__init__`.

        Warning: The arguments are not used, and hence not checked, if the
        Singleton instance already exists even if `check_args=True`.
        """
        if cls.__instance is None:
            return Singleton.__call__(cls, *args, **kwargs)
        return cls.__instance

    def _reset_singleton(cls):
        """Removes the Singleton class's reference to the Singleton instance.

        Raises:
            SingletonRefsExistError: Raised if there exist objects that refer
                to the current Singleton instance if it exists.
        """
        if cls.__instance is not None:
            # sub 2 to remove `getrefcount` arg ref and `cls.__instance` ref
            n_refs = sys.getrefcount(cls.__instance) - 2
            if n_refs > 0:
                err_msg = ('failed to reset %s: %d ref(s) to the Singleton '
                           'instance still exist') % (cls.__name__, n_refs)
                raise SingletonRefsExistError(cls.__name__, err_msg)

            cls.__instance = None
            if cls.__check_args:
                del cls.__args

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__call__(*args, **kwargs)
            if cls.__check_args:
                cls.__args = _get_init_arguments(cls, *args, **kwargs)
        elif cls.__check_args:
            err_msg = (cls.__name__ + ' instance already exists but '
                       'previously initialised differently - '
                       'instance: %s, call: %s')
            args = _get_init_arguments(cls, *args, **kwargs)
            if args != cls.__args:
                raise ValueError(err_msg % (cls.__args, args))

        return cls.__instance


def _get_init_arguments(cls, *args, **kwargs):
    """Returns an OrderedDict of args passed to cls.__init__ given [kw]args."""
    init_args = inspect.signature(cls.__init__)
    bound_args = init_args.bind(None, *args, **kwargs)
    bound_args.apply_defaults()
    arg_dict = bound_args.arguments
    del arg_dict['self']
    return arg_dict
