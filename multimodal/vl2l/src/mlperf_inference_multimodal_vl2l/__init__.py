"""Reference Implementation for the Vision-language-to-language (VL2L) Benchmark"""

from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlperf-inference-multimodal-vl2l")
except PackageNotFoundError:
    pass
