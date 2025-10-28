"""Reference Implementation for the Vision-language-to-language (VL2L) Benchmark."""

from __future__ import annotations

import contextlib
from importlib.metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("mlperf-inference-multimodal-vl2l")
