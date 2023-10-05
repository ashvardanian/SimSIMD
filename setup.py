import os
import sys
import platform
from os.path import dirname
from setuptools import setup, Extension

import glob
import numpy

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "simsimd"

compile_args = []
link_args = []
macros_args = []
source_args = ["python/lib.c", "src/autovec.c"]
libraries_args = []
include_args = ["include", numpy.get_include()]

if sys.platform == "linux":
    compile_args.append("-std=c11")
    compile_args.append("-O3")
    compile_args.append("-ffast-math")
    compile_args.append("-fdiagnostics-color=always")

    # Disable warnings
    compile_args.append("-w")


if sys.platform == "darwin":
    compile_args.append("-std=c11")
    compile_args.append("-O3")
    compile_args.append("-ffast-math")

    # Disable warnings
    compile_args.append("-w")

if sys.platform == "win32":
    compile_args.append("/std:c11")
    compile_args.append("/O2")
    compile_args.append("/fp:fast")


arch = platform.machine()

if arch == "x86_64":
    libraries_args.append("simsimd_x86_avx2")
    libraries_args.append("simsimd_x86_avx512")
elif arch == "aarch64" or arch == "arm64":
    libraries_args.append("simsimd_arm_neon")
    libraries_args.append("simsimd_arm_sve")


ext_modules = [
    Extension(
        "simsimd",
        sources=source_args,
        libraries=libraries_args,
        library_dirs=["src"],
        include_dirs=include_args,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros_args,
    ),
]

this_directory = os.path.abspath(dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf8") as f:
    long_description = f.read()

setup(
    name=__lib_name__,
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/simsimd",
    description="SIMD-accelerated similarity measures for x86 and Arm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    ext_modules=ext_modules,
    zip_safe=False,
)
