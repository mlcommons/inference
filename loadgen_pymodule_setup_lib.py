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
Wraps a loadgen library binary (compiled with an arbitrary toolchain)
with bindings compiled by Python's C++ toolchain.
"""

from setuptools import setup, Extension

mlperf_loadgen_module = Extension('mlperf_loadgen',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '5')],
                    library_dirs = ['obj/loadgen'],
                    libraries = ['mlperf_loadgen'],
                    include_dirs = [ 'gen', 'gen/third_party/pybind/include' ],
                    sources = [ 'gen/loadgen/bindings/python_api.cc' ])

setup (name = 'mlperf_loadgen',
       version = '0.5a0',
       description = 'MLPerf Inference LoadGen python bindings',
       url = 'https://mlperf.org',
       ext_modules = [mlperf_loadgen_module])
