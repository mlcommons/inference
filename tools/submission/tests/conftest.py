"""Pytest configuration for the submission-checker tests.

The snapshot test (``test_snapshot.py``) needs a cloned submission repo,
supplied via ``MLPERF_SUBMISSION_DIR``. When that variable is not set we skip
*collecting* the module entirely rather than skipping the test at runtime:
a runtime skip would leave syrupy's snapshot marked "unused", which makes
pytest exit non-zero. Ignoring collection keeps a plain ``pytest`` run of this
directory green when the repo isn't available.
"""

import os

collect_ignore = []
if not os.environ.get("MLPERF_SUBMISSION_DIR"):
    collect_ignore.append("test_snapshot.py")
