"""MLPerf Inference LoadGen python bindings.

Creates a module that python can import.
All source files are compiled by python's C++ toolchain  without depending
on a loadgen lib.

This setup.py can be used stand-alone, without the use of an external
build system. This will polute your source tree with output files
and binaries. Use one of the gn build targets instead if you want
to avoid poluting the source tree.
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
                    include_dirs = [ '.', '../third_party/pybind/include' ],
                    sources = sources)

setup (name = 'mlperf_loadgen',
       version = '0.5a0',
       description = 'MLPerf Inference LoadGen python bindings',
       url = 'https://mlperf.org',
       ext_modules = [mlperf_loadgen_module])
