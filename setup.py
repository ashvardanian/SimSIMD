# -*- coding: utf-8 -*-
"""
SimSIMD build configuration.

This file configures wheels compilation for SimSIMD CPython bindings.
The architecture detection uses environment variable overrides (set via cibuildwheel)
to support cross-compilation scenarios like building ARM64 wheels on x64 hosts.
"""
from __future__ import annotations

import os
import sys
import platform
from pathlib import Path
from typing import List, Tuple

from setuptools import setup, Extension

__lib_name__ = "simsimd"
__version__ = Path("VERSION").read_text().strip()

# --------------------------------------------------------------------------- #
# macOS developer-tools sanity check                                          #
# --------------------------------------------------------------------------- #
if sys.platform == "darwin":
    _bad_dev_dir = os.environ.get("DEVELOPER_DIR")
    if _bad_dev_dir and (_bad_dev_dir == "public" or not Path(_bad_dev_dir).exists()):
        print(f"[SimSIMD] Ignoring invalid DEVELOPER_DIR={_bad_dev_dir!r}")
        os.environ.pop("DEVELOPER_DIR", None)


# --------------------------------------------------------------------------- #
# Architecture detection with environment override support                     #
# --------------------------------------------------------------------------- #


def is_64bit_x86() -> bool:
    """Detect x86-64 architecture with environment override support."""
    override = os.environ.get("SIMSIMD_TARGET_X86")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)


def is_64bit_arm() -> bool:
    """Detect ARM64 architecture with environment override support."""
    override = os.environ.get("SIMSIMD_TARGET_ARM64")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)


# --------------------------------------------------------------------------- #
# Per-platform build settings                                                 #
# --------------------------------------------------------------------------- #


def linux_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for Linux."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-ffast-math",
        "-fdiagnostics-color=always",
        "-fvisibility=default",
        "-fPIC",
        "-w",  # Hush warnings
        "-fopenmp",  # Enable OpenMP for parallelization
    ]
    link_args = [
        "-shared",
        "-fopenmp",  # Link against OpenMP
        "-lm",  # Add vectorized `logf` implementation from the `glibc`
    ]
    # On Linux with GCC, enable all SIMD targets for the detected architecture
    macros = [
        ("SIMSIMD_DYNAMIC_DISPATCH", "1"),
        ("SIMSIMD_NATIVE_F16", "0"),
        ("SIMSIMD_NATIVE_BF16", "0"),
        # x86 targets
        ("SIMSIMD_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_ICE", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_GENOA", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_SAPPHIRE", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_TURIN", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_SIERRA", "0"),  # avx2vnni not supported by manylinux GCC
        # ARM targets
        ("SIMSIMD_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_I8", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_F16", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_BF16", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_SVE", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_SVE_I8", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_SVE_F16", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_SVE_BF16", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_SVE2", "1" if is_64bit_arm() else "0"),
    ]
    return compile_args, link_args, macros


def darwin_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for macOS."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-ffast-math",
        "-w",  # Hush warnings
    ]
    link_args: List[str] = []
    # macOS: no SVE, conservative AVX-512 (not widely available)
    macros = [
        ("SIMSIMD_DYNAMIC_DISPATCH", "1"),
        ("SIMSIMD_NATIVE_F16", "0"),
        ("SIMSIMD_NATIVE_BF16", "0"),
        # x86 targets - conservative for macOS compatibility
        ("SIMSIMD_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_SKYLAKE", "0"),  # AVX-512 not common on Mac
        ("SIMSIMD_TARGET_ICE", "0"),
        ("SIMSIMD_TARGET_GENOA", "0"),
        ("SIMSIMD_TARGET_SAPPHIRE", "0"),
        ("SIMSIMD_TARGET_TURIN", "0"),
        ("SIMSIMD_TARGET_SIERRA", "0"),
        # ARM targets - NEON only, no SVE on Apple Silicon
        ("SIMSIMD_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_I8", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_F16", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_BF16", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_SVE", "0"),
        ("SIMSIMD_TARGET_SVE_I8", "0"),
        ("SIMSIMD_TARGET_SVE_F16", "0"),
        ("SIMSIMD_TARGET_SVE_BF16", "0"),
        ("SIMSIMD_TARGET_SVE2", "0"),
    ]
    return compile_args, link_args, macros


def windows_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for Windows."""
    compile_args = [
        "/std:c11",
        "/O2",
        "/fp:fast",
        # Dealing with MinGW linking errors
        # https://cibuildwheel.readthedocs.io/en/stable/faq/#windows-importerror-dll-load-failed-the-specific-module-could-not-be-found
        "/d2FH4-",
        "/w",
    ]
    link_args: List[str] = []
    # Windows: no SVE, conservative x86 SIMD, as MSVC lacks BF16/FP16 intrinsics support
    macros = [
        ("SIMSIMD_DYNAMIC_DISPATCH", "1"),
        ("SIMSIMD_NATIVE_F16", "0"),
        ("SIMSIMD_NATIVE_BF16", "0"),
        # x86 targets - conservative for MSVC compatibility
        ("SIMSIMD_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_ICE", "1" if is_64bit_x86() else "0"),
        ("SIMSIMD_TARGET_GENOA", "0"),  # BF16 intrinsics broken in MSVC
        ("SIMSIMD_TARGET_SAPPHIRE", "0"),  # FP16 intrinsics broken in MSVC
        ("SIMSIMD_TARGET_TURIN", "0"),  # `VP2INTERSECT` limited in MSVC
        ("SIMSIMD_TARGET_SIERRA", "0"),  # AVX2 VNNI limits in MSVC
        ("SIMSIMD_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_I8", "1" if is_64bit_arm() else "0"),
        ("SIMSIMD_TARGET_NEON_F16", "0"),  # MSVC lacks `float16_t` intrinsics
        ("SIMSIMD_TARGET_NEON_BF16", "0"),  # MSVC lacks `bfloat16x8_t` intrinsics
        ("SIMSIMD_TARGET_SVE", "0"),
        ("SIMSIMD_TARGET_SVE_I8", "0"),
        ("SIMSIMD_TARGET_SVE_F16", "0"),
        ("SIMSIMD_TARGET_SVE_BF16", "0"),
        ("SIMSIMD_TARGET_SVE2", "0"),
    ]
    # MSVC requires architecture-specific macros for winnt.h
    if is_64bit_arm():
        macros.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros.append(("_AMD64_", "1"))

    return compile_args, link_args, macros


# --------------------------------------------------------------------------- #
# Platform dispatch                                                           #
# --------------------------------------------------------------------------- #

if sys.platform == "linux":
    compile_args, link_args, macros = linux_settings()
elif sys.platform == "darwin":
    compile_args, link_args, macros = darwin_settings()
elif sys.platform == "win32":
    compile_args, link_args, macros = windows_settings()
else:
    compile_args, link_args, macros = [], [], []


# --------------------------------------------------------------------------- #
# Editable install detection                                                  #
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
# Extension module                                                            #
# --------------------------------------------------------------------------- #

ext_modules = [
    Extension(
        "simsimd",
        sources=["python/lib.c", "c/lib.c"],
        include_dirs=["include"],
        language="c",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros,
    )
]

# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #

setup(
    name=__lib_name__,
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/simsimd",
    description="Portable mixed-precision BLAS-like vector math library for x86 and ARM",
    long_description=Path("README.md").read_text(encoding="utf8"),
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    zip_safe=False,
    include_package_data=True,
    **SETUP_KWARGS,
)
