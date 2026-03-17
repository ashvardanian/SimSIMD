# RISC-V 64 GNU toolchain for NumKong.
#
# Usage:
#   cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake
#   cmake --build build_riscv
#
# Optional inputs:
#   -D RISCV_TOOLCHAIN_PATH=/path/to/riscv-gnu-toolchain
#   -D RISCV_SYSROOT=/path/to/riscv64/sysroot
#   -D RISCV_MARCH=rv64gcv_zvfh_zvfbfwma_zvbb
#   -D RISCV_MABI=lp64d
#
# Prerequisites:
#   Download the toolchain: https://github.com/riscv-collab/riscv-gnu-toolchain/releases
#   sudo apt install qemu-user
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR
#
# To customize the architecture, set RISCV_MARCH before configuring:
#   cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-gnu.cmake -D RISCV_MARCH=rv64gcv

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross-compiler from the riscv-collab toolchain.
# Set RISCV_TOOLCHAIN_PATH to override the default location.
if (NOT DEFINED RISCV_TOOLCHAIN_PATH)
    if (DEFINED ENV{RISCV_TOOLCHAIN_PATH})
        set(RISCV_TOOLCHAIN_PATH "$ENV{RISCV_TOOLCHAIN_PATH}")
    else ()
        set(RISCV_TOOLCHAIN_PATH "/tmp/riscv")
    endif ()
endif ()
set(RISCV_TOOLCHAIN_PATH "${RISCV_TOOLCHAIN_PATH}" CACHE PATH "RISC-V GNU toolchain root")
set(ENV{RISCV_TOOLCHAIN_PATH} "${RISCV_TOOLCHAIN_PATH}")

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-g++")

# Sysroot for the target system. By default it is derived from RISCV_TOOLCHAIN_PATH.
if (NOT DEFINED RISCV_SYSROOT)
    if (DEFINED ENV{RISCV_SYSROOT})
        set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}")
    else ()
        set(RISCV_SYSROOT "${RISCV_TOOLCHAIN_PATH}/sysroot")
    endif ()
endif ()
set(RISCV_SYSROOT "${RISCV_SYSROOT}" CACHE PATH "RISC-V sysroot for GNU cross-compilation")
set(ENV{RISCV_SYSROOT} "${RISCV_SYSROOT}")
set(CMAKE_SYSROOT "${RISCV_SYSROOT}")

# Target architecture with Vector extension (RVV 1.0)
# Default: Enables all extensions supported by NumKong
#
# Base architecture components:
#   rv64gc  : RV64 + G (IMAFD) + C (compressed)
#   v       : Vector extension (RVV 1.0)
#   lp64d   : 64-bit pointers, double-precision floats in FP registers
#
# Common profiles:
#   rv64gcv                     : Base RVV 1.0 only
#   rv64gcv_zvfh                : RVV + fp16 (SiFive-class)
#   rv64gcv_zvfh_zvfbfwma       : RVV + fp16 + bf16
#   rv64gcv_zvfh_zvfbfwma_zvbb  : RVV + fp16 + bf16 + bit-manip (full NumKong support)
#
if (NOT DEFINED RISCV_MARCH)
    if (DEFINED ENV{RISCV_MARCH})
        set(RISCV_MARCH "$ENV{RISCV_MARCH}")
    else ()
        set(RISCV_MARCH "rv64gcv_zvfh_zvfbfwma_zvbb")
    endif ()
endif ()
set(RISCV_MARCH "${RISCV_MARCH}" CACHE STRING "RISC-V ISA string for GNU cross-compilation")
set(ENV{RISCV_MARCH} "${RISCV_MARCH}")

if (NOT DEFINED RISCV_MABI)
    if (DEFINED ENV{RISCV_MABI})
        set(RISCV_MABI "$ENV{RISCV_MABI}")
    else ()
        set(RISCV_MABI "lp64d")
    endif ()
endif ()
set(RISCV_MABI "${RISCV_MABI}" CACHE STRING "RISC-V ABI for GNU cross-compilation")
set(ENV{RISCV_MABI} "${RISCV_MABI}")

# Nested try_compile() projects re-enter this toolchain file, so forward the
# custom cross-compilation settings explicitly.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_PATH RISCV_SYSROOT RISCV_MARCH RISCV_MABI)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")

# QEMU user-mode emulation for running tests.
# -L sysroot provides the dynamic linker path.
# -cpu max enables all supported extensions, including zvfh and zvfbfwma when available.
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-riscv64;-L;${RISCV_SYSROOT};-cpu;max")

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH "${RISCV_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
