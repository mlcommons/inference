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
from version_generator import generate_loadgen_version_definitions

generated_version_source_filename = "generated/version.cc"
generate_loadgen_version_definitions(generated_version_source_filename)

public_headers = [
  "loadgen.h",
  "query_sample.h",
  "query_sample_library.h",
  "system_under_test.h",
  "test_settings.h",
]

lib_headers = [
  "logging.h",
  "test_settings_internal.h",
  "trace_generator.h",
  "utils.h",
  "version.h",
]

lib_sources = [
  "loadgen.cc",
  "logging.cc",
]

mlperf_loadgen_headers = public_headers + lib_headers
mlperf_loadgen_sources_no_gen = lib_sources
mlperf_loadgen_sources = \
    mlperf_loadgen_sources_no_gen + [generated_version_source_filename]

sources = [
    "bindings/python_api.cc",
    "generated/version.cc",
    "loadgen.cc",
    "logging.cc",
]

mlperf_loadgen_module = Extension('mlperf_loadgen',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '5')],
                    include_dirs = [ '.', '../third_party/pybind/include' ],
                    sources = mlperf_loadgen_sources + sources,
                    depends = mlperf_loadgen_headers)

setup (name = 'mlperf_loadgen',
       version = '0.5a0',
       description = 'MLPerf Inference LoadGen python bindings',
       url = 'https://mlperf.org',
       ext_modules = [mlperf_loadgen_module])
