import logging


class LoggerMixin:
    """Adds a logging.Logger to a class.

    Attributes:
        _logger: A logging.Logger with name equal to the concatenation of the
            class module and class name.
    """
    @property
    def _logger(self):
        name = '.'.join([self.__module__, self.__class__.__name__])
        return logging.getLogger(name)


class log_call:
    """A decorator that logs a method call plus it's args and kwargs.

    The class that the method belongs to must include LoggerMixin as a base.

    Args:
        level (int): The logging level (e.g. logging.DEBUG).
    """
    def __init__(self, level):
        self._level = level

    def __call__(self, f):
        def wrapper(f_self, *args, **kwargs):
            msg = 'entering %r, args: %r, kwargs: %r'
            f_self._logger.log(self._level, msg, f.__name__, args, kwargs)

            result = f(f_self, *args, **kwargs)

            f_self._logger.log(self._level, 'exited %r', f.__name__)

            return result
        return wrapper


def log_call_debug(f):
    return log_call(logging.DEBUG)(f)


def log_call_info(f):
    return log_call(logging.INFO)(f)


def log_call_warning(f):
    return log_call(logging.WARNING)(f)


def log_call_error(f):
    return log_call(logging.ERROR)(f)


def log_call_critical(f):
    return log_call(logging.CRITICAL)(f)
