import argparse
import logging


class LogLevelAction(argparse.Action):
    """An `argparse.Action` for levels in the `logging` module.

    Example usage:

        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--loglevel', action=LogLevelAction)

    """

    LEVELS = {'DEBUG':    logging.DEBUG,
              'INFO':     logging.INFO,
              'WARNING':  logging.WARNING,
              'ERROR':    logging.ERROR,
              'CRITICAL': logging.CRITICAL}

    def __init__(self, option_strings, dest, nargs=None, const=None,
                 type=str, choices=LEVELS.keys(), default='DEBUG', **kwargs):
        if nargs is not None:
            raise ValueError('nargs must be None')
        if const is not None:
            raise ValueError('const must be None')
        if type is not str:
            raise ValueError('type must be str')
        if any([choice not in self.LEVELS for choice in choices]):
            raise ValueError('choices=%r must be a subset of %r' % (
                choices, list(self.LEVELS.keys())))
        if default not in choices:
            raise ValueError('default=%r must be in %r' % (default,
                                                           list(choices)))

        super().__init__(option_strings, dest, nargs=nargs, const=const,
                         type=type, choices=choices, default=default, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        if value not in self.choices:
            name = self.metavar
            if name is None:
                name = self.dest.upper()
            raise ValueError('%s must be in %s' % (name, self.choices))
        setattr(namespace, self.dest, self._str_to_int(value))

    @classmethod
    def _str_to_int(cls, str_level):
        return cls.LEVELS[str_level]
