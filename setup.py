# -*- coding: utf-8 -*-
"""
NumKong build configuration.

This file configures wheels compilation for NumKong CPython bindings.
The architecture detection uses environment variable overrides (set via cibuildwheel)
to support cross-compilation scenarios like building ARM64 wheels on x64 hosts.
"""

from __future__ import annotations

import os
import sys
import platform
import glob
import subprocess
from pathlib import Path
from typing import List, Tuple

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

__lib_name__ = "numkong"
__version__ = Path("VERSION").read_text().strip()

if sys.platform == "darwin":
    _bad_dev_dir = os.environ.get("DEVELOPER_DIR")
    if _bad_dev_dir and (_bad_dev_dir == "public" or not Path(_bad_dev_dir).exists()):
        print(f"[NumKong] Ignoring invalid DEVELOPER_DIR={_bad_dev_dir!r}")
        os.environ.pop("DEVELOPER_DIR", None)


def is_64bit_x86() -> bool:
    """Detect x86-64 architecture with environment override support."""
    override = os.environ.get("NK_TARGET_X86_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("x86_64", "x64", "amd64")) and (sys.maxsize > 2**32)


def is_64bit_arm() -> bool:
    """Detect ARM64 architecture with environment override support."""
    override = os.environ.get("NK_TARGET_ARM_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("arm64", "aarch64")) and (sys.maxsize > 2**32)


def is_64bit_riscv() -> bool:
    """Detect RISC-V 64-bit architecture with environment override support."""
    override = os.environ.get("NK_TARGET_RISCV_")
    if override is not None:
        return override == "1"
    arch = platform.machine().lower()
    return (arch in ("riscv64",)) and (sys.maxsize > 2**32)


def detect_apple_clang_version() -> Tuple[int, int]:
    """Detect Apple Clang version for SME support (AppleClang 16+ / Clang 18+)."""
    import re

    try:
        result = subprocess.run(["clang", "--version"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "version" in line.lower():
                match = re.search(r"version\s+(\d+)\.(\d+)", line)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
    except Exception:
        pass
    return (0, 0)


def linux_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for Linux."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-fdiagnostics-color=always",
        "-fvisibility=default",
        "-fPIC",
        "-w",  # Hush warnings
    ]
    # On RISC-V, GCC needs `-march` with V extension for vector types to be
    # available at translation-unit scope (`#pragma GCC target` only affects
    # codegen, not type declarations).
    # In manylinux_2_39 riscv64 CI we build with Clang + LLD to support
    # zvfh/zvfbfwma/zvbb in this target string.
    if is_64bit_riscv():
        compile_args.append("-march=rv64gcv_zvfh_zvfbfwma_zvbb")
    link_args = [
        "-shared",
        "-lm",  # Add vectorized `logf` implementation from the `glibc`
    ]
    # On Linux with GCC, enable all SIMD targets for the detected architecture
    macros = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
        # x86 targets
        ("NK_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_GENOA", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SAPPHIRE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_TURIN", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_ALDER", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SIERRA", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SAPPHIREAMX", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_GRANITEAMX", "1" if is_64bit_x86() else "0"),
        # ARM NEON targets
        ("NK_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONSDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONBFDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONFHM", "1" if is_64bit_arm() else "0"),
        # ARM SVE targets
        ("NK_TARGET_SVE", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVEHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVEBFDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVESDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVE2", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVE2P1", "1" if is_64bit_arm() else "0"),
        # ARM SME targets
        ("NK_TARGET_SME", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SME2", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SME2P1", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEF64", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEBF16", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEBI32", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMELUT2", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEFA64", "1" if is_64bit_arm() else "0"),
        # RISC-V targets
        ("NK_TARGET_RVV", "1" if is_64bit_riscv() else "0"),
        ("NK_TARGET_RVVHALF", "1" if is_64bit_riscv() else "0"),
        ("NK_TARGET_RVVBF16", "1" if is_64bit_riscv() else "0"),
        ("NK_TARGET_RVVBB", "1" if is_64bit_riscv() else "0"),
    ]
    return compile_args, link_args, macros


def darwin_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for macOS."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-w",  # Hush warnings
    ]
    link_args: List[str] = []
    # SME available on M4+ with AppleClang 16+ (Xcode 16) or upstream Clang 18+
    clang_major, _ = detect_apple_clang_version()
    has_sme = is_64bit_arm() and clang_major >= 16
    # macOS: no SVE, conservative AVX-512 (not widely available)
    macros = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
        # x86 targets - conservative for macOS compatibility
        ("NK_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SKYLAKE", "0"),  # AVX-512 not common on Mac
        ("NK_TARGET_ICELAKE", "0"),
        ("NK_TARGET_GENOA", "0"),
        ("NK_TARGET_SAPPHIRE", "0"),
        ("NK_TARGET_TURIN", "0"),
        ("NK_TARGET_ALDER", "0"),
        ("NK_TARGET_SIERRA", "0"),
        ("NK_TARGET_SAPPHIREAMX", "0"),
        ("NK_TARGET_GRANITEAMX", "0"),
        # ARM NEON targets - NEON only on Apple Silicon
        ("NK_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONSDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONBFDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONFHM", "1" if is_64bit_arm() else "0"),
        # ARM SVE targets - not available on Apple Silicon
        ("NK_TARGET_SVE", "0"),
        ("NK_TARGET_SVEHALF", "0"),
        ("NK_TARGET_SVEBFDOT", "0"),
        ("NK_TARGET_SVESDOT", "0"),
        ("NK_TARGET_SVE2", "0"),
        ("NK_TARGET_SVE2P1", "0"),
        # ARM SME targets - M4+ with AppleClang 16+ (Xcode 16)
        ("NK_TARGET_SME", "1" if has_sme else "0"),
        ("NK_TARGET_SME2", "1" if has_sme else "0"),
        ("NK_TARGET_SME2P1", "1" if has_sme else "0"),
        ("NK_TARGET_SMEF64", "1" if has_sme else "0"),
        ("NK_TARGET_SMEHALF", "1" if has_sme else "0"),
        ("NK_TARGET_SMEBF16", "1" if has_sme else "0"),
        ("NK_TARGET_SMEBI32", "1" if has_sme else "0"),
        ("NK_TARGET_SMELUT2", "1" if has_sme else "0"),
        ("NK_TARGET_SMEFA64", "1" if has_sme else "0"),
        # RISC-V targets - not available on macOS
        ("NK_TARGET_RVV", "0"),
        ("NK_TARGET_RVVHALF", "0"),
        ("NK_TARGET_RVVBF16", "0"),
        ("NK_TARGET_RVVBB", "0"),
    ]
    return compile_args, link_args, macros


def freebsd_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for FreeBSD."""
    compile_args = [
        "-std=c11",
        "-O3",
        "-fdiagnostics-color=always",
        "-fvisibility=default",
        "-fPIC",
        "-w",  # Hush warnings
    ]
    link_args = [
        "-shared",
        "-lm",  # Math library
    ]
    # FreeBSD: Similar to Linux, enable all SIMD targets for detected architecture
    macros = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
        # x86 targets
        ("NK_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_ICELAKE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_GENOA", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SAPPHIRE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_TURIN", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_ALDER", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SIERRA", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SAPPHIREAMX", "0"),  # AMX may not be available on FreeBSD
        ("NK_TARGET_GRANITEAMX", "0"),
        # ARM NEON targets
        ("NK_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONSDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONBFDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONFHM", "1" if is_64bit_arm() else "0"),
        # ARM SVE targets
        ("NK_TARGET_SVE", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVEHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVEBFDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVESDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVE2", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SVE2P1", "1" if is_64bit_arm() else "0"),
        # ARM SME targets (may require newer FreeBSD)
        ("NK_TARGET_SME", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SME2", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SME2P1", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEF64", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEHALF", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEBF16", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEBI32", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMELUT2", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_SMEFA64", "1" if is_64bit_arm() else "0"),
        # RISC-V targets
        ("NK_TARGET_RVV", "1" if is_64bit_riscv() else "0"),
        ("NK_TARGET_RVVHALF", "1" if is_64bit_riscv() else "0"),
        ("NK_TARGET_RVVBF16", "1" if is_64bit_riscv() else "0"),
        ("NK_TARGET_RVVBB", "1" if is_64bit_riscv() else "0"),
    ]
    return compile_args, link_args, macros


def detect_msvc_version() -> Tuple[int, int]:
    """Detect MSVC version from cl.exe or environment variables."""
    try:
        result = subprocess.run(["cl"], capture_output=True, text=True, shell=True)
        # Parse version from output like "Microsoft (R) C/C++ Optimizing Compiler Version 19.30.30705"
        for line in result.stderr.split("\n"):
            if "Version" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Version" and i + 1 < len(parts):
                        version_str = parts[i + 1]
                        # Version format is like 19.30.30705
                        version_parts = version_str.split(".")
                        if len(version_parts) >= 2:
                            major = int(version_parts[0])
                            minor = int(version_parts[1])
                            print(f"[NumKong] Detected MSVC version {major}.{minor}")
                            return (major, minor)
    except:
        pass

    # Fallback to checking _MSC_VER from environment or defaults
    # MSVC 2019: 19.20-19.29
    # MSVC 2022: 19.30-19.39
    print("[NumKong] MSVC version detection failed, using conservative defaults (MSVC 2019)")
    return (19, 20)  # Conservative default to MSVC 2019


def windows_settings() -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Build settings for Windows."""
    compile_args = [
        "/std:c11",
        "/O2",
        # Dealing with MinGW linking errors
        # https://cibuildwheel.readthedocs.io/en/stable/faq/#windows-importerror-dll-load-failed-the-specific-module-could-not-be-found
        "/d2FH4-",
        "/w",
    ]
    link_args: List[str] = []

    # Detect MSVC version for feature support
    msvc_major, msvc_minor = detect_msvc_version()
    # MSVC 19.44+ (VS 2022 17.14+): all AVX-512 intrinsics available without /arch:AVX512
    has_full_avx512 = msvc_major >= 19 and msvc_minor >= 44

    # Windows: SVE/SME not supported, x86 SIMD support varies by MSVC version
    macros = [
        ("NK_DYNAMIC_DISPATCH", "1"),
        ("NK_NATIVE_F16", "0"),
        ("NK_NATIVE_BF16", "0"),
        # x86 targets - base support
        ("NK_TARGET_HASWELL", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_SKYLAKE", "1" if is_64bit_x86() else "0"),
        ("NK_TARGET_ICELAKE", "1" if is_64bit_x86() else "0"),
        # Advanced x86 targets - require MSVC 19.44+ for full AVX-512 FP16/BF16/VNNI
        ("NK_TARGET_GENOA", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        ("NK_TARGET_SAPPHIRE", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        ("NK_TARGET_TURIN", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        ("NK_TARGET_ALDER", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        ("NK_TARGET_SIERRA", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        ("NK_TARGET_SAPPHIREAMX", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        ("NK_TARGET_GRANITEAMX", "1" if (is_64bit_x86() and has_full_avx512) else "0"),
        # ARM NEON targets
        ("NK_TARGET_NEON", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONHALF", "0"),  # MSVC lacks `float16_t` intrinsics
        ("NK_TARGET_NEONSDOT", "1" if is_64bit_arm() else "0"),
        ("NK_TARGET_NEONBFDOT", "0"),  # MSVC lacks `bfloat16x8_t` intrinsics
        ("NK_TARGET_NEONFHM", "0"),  # MSVC lacks FHM intrinsics
        # ARM SVE targets - not supported on Windows
        ("NK_TARGET_SVE", "0"),
        ("NK_TARGET_SVEHALF", "0"),
        ("NK_TARGET_SVEBFDOT", "0"),
        ("NK_TARGET_SVESDOT", "0"),
        ("NK_TARGET_SVE2", "0"),
        ("NK_TARGET_SVE2P1", "0"),
        # ARM SME targets - not supported on Windows
        ("NK_TARGET_SME", "0"),
        ("NK_TARGET_SME2", "0"),
        ("NK_TARGET_SME2P1", "0"),
        ("NK_TARGET_SMEF64", "0"),
        ("NK_TARGET_SMEHALF", "0"),
        ("NK_TARGET_SMEBF16", "0"),
        ("NK_TARGET_SMEBI32", "0"),
        ("NK_TARGET_SMELUT2", "0"),
        ("NK_TARGET_SMEFA64", "0"),
        # RISC-V targets - not supported on Windows
        ("NK_TARGET_RVV", "0"),
        ("NK_TARGET_RVVHALF", "0"),
        ("NK_TARGET_RVVBF16", "0"),
        ("NK_TARGET_RVVBB", "0"),
    ]
    # MSVC requires architecture-specific macros for winnt.h
    if is_64bit_arm():
        macros.append(("_ARM64_", "1"))
    elif is_64bit_x86():
        macros.append(("_AMD64_", "1"))

    return compile_args, link_args, macros


if sys.platform == "linux":
    compile_args, link_args, macros = linux_settings()
elif sys.platform.startswith("freebsd"):
    # FreeBSD platform strings can be "freebsd11", "freebsd12", etc.
    compile_args, link_args, macros = freebsd_settings()
elif sys.platform == "darwin":
    compile_args, link_args, macros = darwin_settings()
elif sys.platform == "win32":
    compile_args, link_args, macros = windows_settings()
else:
    # Default to minimal settings for unknown platforms
    compile_args, link_args, macros = [], [], []


def _is_editable_install() -> bool:
    if "develop" in sys.argv or ("install" in sys.argv and "-e" in sys.argv):
        return True
    for p in sys.path:
        if Path(p, f"{__lib_name__}.egg-link").exists():
            return True
    return False


SETUP_KWARGS = (
    {
        "packages": ["numkong"],
        "package_dir": {"numkong": "python/annotations"},
        "package_data": {"numkong": ["__init__.pyi", "py.typed"]},
    }
    if not _is_editable_install()
    else {}
)

if _is_editable_install():
    print("[NumKong] Editable install detected - skipping bundled type stubs.")


# Use glob to find all dispatch files
base_sources = [
    "python/numkong.c",
    "python/tensor.c",
    "python/matrix.c",
    "python/types.c",
    "python/distance.c",
    "python/each.c",
    "python/mesh.c",
    "c/numkong.c",
]

dispatch_sources = sorted(glob.glob("c/dispatch_*.c"))

ext_modules = [
    Extension(
        "numkong",
        sources=base_sources + dispatch_sources,
        include_dirs=["include", "python"],
        language="c",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros,
    )
]


class ParallelBuildExt(build_ext):
    def initialize_options(self):
        super().initialize_options()
        # In Docker containers (e.g. cibuildwheel), `os.cpu_count()` returns the
        # *host* core count, not the container's allocated vCPUs.  Launching dozens
        # of heavy SIMD compilation jobs in parallel OOMs the container (exit 143).
        self.parallel = int(os.environ.get("NK_BUILD_PARALLEL", min(os.cpu_count() or 1, 4)))


setup(
    name=__lib_name__,
    cmdclass={"build_ext": ParallelBuildExt},
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://github.com/ashvardanian/numkong",
    description="Portable mixed-precision BLAS-like vector math library for x86 and ARM",
    long_description=Path("README.md").read_text(encoding="utf8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
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
    python_requires=">=3.9",
    ext_modules=ext_modules,
    zip_safe=False,
    include_package_data=True,
    **SETUP_KWARGS,
)
