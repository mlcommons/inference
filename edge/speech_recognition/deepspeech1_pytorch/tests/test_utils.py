import contextlib
import os


@contextlib.contextmanager
def environ(**env):
    """A context manager that temporarily changes `os.environ`."""
    saved = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
