import os
import sys
from os.path import dirname
from setuptools import setup, Extension

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "simsimd"

compile_args = []
link_args = []
macros_args = [
    ("SIMSIMD_NATIVE_F16", "0"),
    ("SIMSIMD_DYNAMIC_DISPATCH", "1"),
]

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
        sources=["python/lib.c", "c/lib.c"],
        include_dirs=["include"],
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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Programming Language :: C",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    zip_safe=False,
)
