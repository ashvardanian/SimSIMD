import os
import sys
from os.path import dirname
from setuptools import setup, Extension

import numpy

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "simsimd"

compile_args = []
link_args = []
macros_args = []

if sys.platform == "linux":
    compile_args.append("-std=c11")
    compile_args.append("-O3")
    compile_args.append("-ffast-math")
    compile_args.append("-fdiagnostics-color=always")

    # Disable warnings
    compile_args.append("-w")

    # Enable OpenMP for Linux
    compile_args.append("-fopenmp")
    link_args.append("-lgomp")

    # Add vectorized `logf` implementation from the `glibc`
    link_args.append("-lm")

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

    # Dealing with MinGW linking errors
    # https://cibuildwheel.readthedocs.io/en/stable/faq/#windows-importerror-dll-load-failed-the-specific-module-could-not-be-found
    compile_args.append("/d2FH4-")

ext_modules = [
    Extension(
        "simsimd",
        sources=["python/lib.c"],
        include_dirs=["include", numpy.get_include()],
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
    description="Fastest SIMD-Accelerated Vector Similarity Functions for x86 and Arm",
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
