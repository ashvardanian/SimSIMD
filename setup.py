import os
from os.path import dirname
from setuptools import setup, Extension

__version__ = open("VERSION", "r").read().strip()
__lib_name__ = "simsimd"


this_directory = os.path.abspath(dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf8") as f:
    long_description = f.read()

"-march=armv8.2-a+sve"
"-march=sapphirerapids"

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
    ext_modules=[
        Extension(
            "simsimd",
            sources=["python/python.c"],
            include_dirs=["include"],
        )
    ],
    zip_safe=False,
)
