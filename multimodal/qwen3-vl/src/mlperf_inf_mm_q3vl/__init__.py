"""Reference Implementation for the Qwen3-VL (Q3VL) Benchmark."""

from __future__ import annotations

import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("mlperf-inf-mm-q3vl")
