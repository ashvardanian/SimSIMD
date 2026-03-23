# LoongArch 64 GNU toolchain for NumKong.
#
# Usage:
#   cmake -B build_loongarch -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-loongarch64-gnu.cmake
#   cmake --build build_loongarch
#
# Optional inputs:
#   -D LOONGARCH_TOOLCHAIN_PATH=/path/to/loongarch-gnu-toolchain
#   -D LOONGARCH_SYSROOT=/path/to/loongarch64/sysroot
#
# Prerequisites:
#   sudo apt install gcc-loongarch64-linux-gnu g++-loongarch64-linux-gnu qemu-user
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

# Cross-compiler.
# Set LOONGARCH_TOOLCHAIN_PATH to override the default location.
if (NOT DEFINED LOONGARCH_TOOLCHAIN_PATH)
    if (DEFINED ENV{LOONGARCH_TOOLCHAIN_PATH})
        set(LOONGARCH_TOOLCHAIN_PATH "$ENV{LOONGARCH_TOOLCHAIN_PATH}")
    else ()
        set(LOONGARCH_TOOLCHAIN_PATH "/tmp/loongarch")
    endif ()
endif ()
set(LOONGARCH_TOOLCHAIN_PATH "${LOONGARCH_TOOLCHAIN_PATH}" CACHE PATH "LoongArch GNU toolchain root")
set(ENV{LOONGARCH_TOOLCHAIN_PATH} "${LOONGARCH_TOOLCHAIN_PATH}")

set(CMAKE_C_COMPILER "${LOONGARCH_TOOLCHAIN_PATH}/bin/loongarch64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${LOONGARCH_TOOLCHAIN_PATH}/bin/loongarch64-unknown-linux-gnu-g++")

# Sysroot for the target system.
if (NOT DEFINED LOONGARCH_SYSROOT)
    if (DEFINED ENV{LOONGARCH_SYSROOT})
        set(LOONGARCH_SYSROOT "$ENV{LOONGARCH_SYSROOT}")
    else ()
        set(LOONGARCH_SYSROOT "${LOONGARCH_TOOLCHAIN_PATH}/sysroot")
    endif ()
endif ()
set(LOONGARCH_SYSROOT "${LOONGARCH_SYSROOT}" CACHE PATH "LoongArch sysroot for GNU cross-compilation")
set(ENV{LOONGARCH_SYSROOT} "${LOONGARCH_SYSROOT}")
set(CMAKE_SYSROOT "${LOONGARCH_SYSROOT}")

# Forward custom variables to nested try_compile() projects.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES LOONGARCH_TOOLCHAIN_PATH LOONGARCH_SYSROOT)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Enable LASX (256-bit SIMD) by default.
# The base ABI is lp64d (64-bit pointers, double-precision floats in FP registers).
set(CMAKE_C_FLAGS_INIT "-mlasx")
set(CMAKE_CXX_FLAGS_INIT "-mlasx")

# QEMU user-mode emulation for running tests.
# -L sysroot provides the dynamic linker path.
# -cpu max enables all supported extensions.
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-loongarch64;-L;${LOONGARCH_SYSROOT};-cpu;max")

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH "${LOONGARCH_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
