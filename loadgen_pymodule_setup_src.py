"""MLPerf Inference LoadGen python bindings.

Creates a module that python can import.
All source files are compiled by python's C++ toolchain  without depending
on a loadgen lib.
"""

from setuptools import setup, Extension

sources = [
  "bindings/python_api.cc",
  "loadgen.cc",
  "logging.cc",
]

mlperf_loadgen_module = Extension('mlperf_loadgen',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '5')],
                    include_dirs = [ 'gen', 'gen/third_party/pybind/include' ],
                    sources = [ "gen/loadgen/" + s for s in sources ])

setup (name = 'mlperf_loadgen',
       version = '0.5a0',
       description = 'MLPerf Inference LoadGen python bindings',
       url = 'https://mlperf.org',
       ext_modules = [mlperf_loadgen_module])
