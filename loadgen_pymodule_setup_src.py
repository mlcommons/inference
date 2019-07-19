# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""MLPerf Inference LoadGen python bindings.

Creates a module that python can import.
All source files are compiled by python's C++ toolchain  without depending
on a loadgen lib.
"""

from setuptools import Extension
from setuptools import setup

sources = [
  "bindings/python_api.cc",
  "loadgen.cc",
  "logging.cc",
]

mlperf_loadgen_module = Extension(
        "mlperf_loadgen",
        define_macros=[("MAJOR_VERSION", "0"), ("MINOR_VERSION", "5")],
        include_dirs=["gen", "gen/third_party/pybind/include"],
        sources=["gen/loadgen/" + s for s in sources ])

setup(name="mlperf_loadgen",
      version="0.5a0",
      description="MLPerf Inference LoadGen python bindings",
      url = "https://mlperf.org",
      ext_modules=[mlperf_loadgen_module])
