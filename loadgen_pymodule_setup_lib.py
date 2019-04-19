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
