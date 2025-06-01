# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import platform
import tempfile
from pathlib import Path

from setuptools import setup, Extension
from distutils import ccompiler, sysconfig
from distutils.errors import CompileError

__lib_name__ = "simsimd"
__version__ = Path("VERSION").read_text().strip()

# --------------------------------------------------------------------------- #
#  macOS developer‑tools sanity check                                         #
# --------------------------------------------------------------------------- #
# Users occasionally end up with DEVELOPER_DIR="public" (or another bogus
# path) when installing via package managers that sandbox the tool‑chain.  In
# that state *every* call to `xcrun` fails before the compiler even starts.
# We proactively unset that var so AppleClang falls back to xcode‑select’s
# default path.
# --------------------------------------------------------------------------- #
if sys.platform == "darwin":
    _bad_dev_dir = os.environ.get("DEVELOPER_DIR")
    if _bad_dev_dir and (_bad_dev_dir == "public" or not Path(_bad_dev_dir).exists()):
        print(f"[SimSIMD] Ignoring invalid DEVELOPER_DIR={_bad_dev_dir!r}")
        os.environ.pop("DEVELOPER_DIR", None)
        
# --------------------------------------------------------------------------- #
# Compiler and linker flags common across attempts                            #
# --------------------------------------------------------------------------- #

COMPILE_ARGS: list[str] = []
LINK_ARGS: list[str] = []
MACROS_COMMON: list[tuple[str, str]] = [
    ("SIMSIMD_NATIVE_F16", "0"),
    ("SIMSIMD_NATIVE_BF16", "0"),
    ("SIMSIMD_DYNAMIC_DISPATCH", "1"),
]

if sys.platform == "linux":
    COMPILE_ARGS += [
        "-std=c11",
        "-O3",
        "-ffast-math",
        "-fdiagnostics-color=always",
        "-fvisibility=default",
        "-fPIC",
        "-w",  # Hush warnings
        "-fopenmp",  # Enable OpenMP for parallelization
    ]
    LINK_ARGS += [
        "-shared",
        "-fopenmp",  # Link against OpenMP
        "-lm",  # Add vectorized `logf` implementation from the `glibc`
    ]

elif sys.platform == "darwin":
    COMPILE_ARGS += [
        "-std=c11",
        "-O3",
        "-ffast-math",
        "-w",  # Hush warnings
    ]

elif sys.platform == "win32":
    COMPILE_ARGS += [
        "/std:c11",
        "/O2",
        "/fp:fast",
        "/EXPORT:*",
        # Dealing with MinGW linking errors
        # https://cibuildwheel.readthedocs.io/en/stable/faq/#windows-importerror-dll-load-failed-the-specific-module-could-not-be-found
        "/d2FH4-",
        "/w",
    ]

# --------------------------------------------------------------------------- #
# Pick the order in which to disable targets                                  #
# --------------------------------------------------------------------------- #

ARCH = platform.machine().lower()
if ARCH in ("arm64", "aarch64", "armv8l", "armv7l", "arm"):
    TARGETS_PRIORITY: list[str] = [
        "SIMSIMD_TARGET_SVE2",
        "SIMSIMD_TARGET_SVE_BF16",
        "SIMSIMD_TARGET_SVE_F16",
        "SIMSIMD_TARGET_SVE_I8",
        "SIMSIMD_TARGET_SVE",
        "SIMSIMD_TARGET_NEON_BF16",
        "SIMSIMD_TARGET_NEON_F16",
        "SIMSIMD_TARGET_NEON_I8",
        "SIMSIMD_TARGET_NEON",
    ]
else:
    TARGETS_PRIORITY = [
        "SIMSIMD_TARGET_SIERRA",
        "SIMSIMD_TARGET_TURIN",
        "SIMSIMD_TARGET_SAPPHIRE",
        "SIMSIMD_TARGET_GENOA",
        "SIMSIMD_TARGET_ICE",
        "SIMSIMD_TARGET_SKYLAKE",
        "SIMSIMD_TARGET_HASWELL",
    ]


def _compiles_with(targets_statuses: list[tuple[str, str]]) -> bool:
    """Try to compile the full library with the given targets disabled."""
    compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(compiler)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            compiler.compile(
                ["c/lib.c"],
                output_dir=tmp,
                macros=targets_statuses,
                include_dirs=["include"],
                extra_postargs=COMPILE_ARGS,
            )
        return True
    except CompileError:
        return False


targets_disabled = set()
targets_enabled = [name for name in TARGETS_PRIORITY]
targets_statuses = [(name, "1") for name in targets_enabled]
while not _compiles_with(targets_statuses) and len(targets_enabled) > 0:
    targets_disabled.add(targets_enabled.pop(0))
    targets_statuses = [(name, "1") for name in targets_enabled] + [(name, "0") for name in targets_disabled]

# Final verification - should succeed, otherwise bail early.
if not _compiles_with(targets_statuses):
    raise RuntimeError(
        "SimSIMD failed to compile even after disabling all known SIMD targets.\n"
        "Consider filing an issue with your compiler/architecture details."
    )

if len(targets_disabled) > 0:
    print("[SimSIMD] Disabled targets: ", ", ".join(targets_disabled))


# --------------------------------------------------------------------------- #
# Check if editable install is being performed and skip bundled type stubs    #
# --------------------------------------------------------------------------- #


def _is_editable_install() -> bool:
    if "develop" in sys.argv or ("install" in sys.argv and "-e" in sys.argv):
        return True
    for p in sys.path:
        if Path(p, f"{__lib_name__}.egg-link").exists():
            return True
    return False


SETUP_KWARGS = (
    {
        "packages": ["simsimd"],
        "package_dir": {"simsimd": "python/annotations"},
        "package_data": {"simsimd": ["__init__.pyi", "py.typed"]},
    }
    if not _is_editable_install()
    else {}
)

if _is_editable_install():
    print("[SimSIMD] Editable install detected - skipping bundled type stubs.")

# --------------------------------------------------------------------------- #
# Putting it all together                                                     #
# --------------------------------------------------------------------------- #

EXT_MODULES = [
    Extension(
        "simsimd",
        sources=["python/lib.c", "c/lib.c"],
        include_dirs=["include"],
        language="c",
        extra_compile_args=COMPILE_ARGS,
        extra_link_args=LINK_ARGS,
        define_macros=targets_statuses,
    )
]

LONG_DESCRIPTION = Path("README.md").read_text(encoding="utf8")

setup(
    name=__lib_name__,
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/simsimd",
    description="Portable mixed-precision BLAS-like vector math library for x86 and ARM",
    long_description=LONG_DESCRIPTION,
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=EXT_MODULES,
    zip_safe=False,
    include_package_data=True,
    **SETUP_KWARGS,
)
