#!/usr/bin/env python

import os

import torch
from setuptools import find_packages
from setuptools import setup
import distutils.command.build

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(this_dir, 'lib')
    if not os.path.isdir(lib_dir):
        os.mkdir(lib_dir)
    extensions_dir = os.path.join(this_dir, 'csrc')

    extension = CppExtension

    custom_ops_sources = [os.path.join(extensions_dir, 'vision.cpp')]
    custom_ops_sources += [os.path.join(extensions_dir, 'cpu', 'nms_cpu.cpp')]
    custom_ops_sources_cuda = [os.path.join(extensions_dir, 'cuda', 'nms.cu')]

    extra_compile_args = {'cxx': []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        custom_ops_sources += custom_ops_sources_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "lib.custom_ops",
            custom_ops_sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ]

    return ext_modules

setup(
    name="custom_nms_ops",
    version="0.1",
    author="bowbao",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension,
              "build": distutils.command.build.build}
)