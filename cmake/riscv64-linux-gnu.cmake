# RISC-V 64-bit Linux cross-compilation toolchain for NumKong
#
# Usage:
#   cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/riscv64-linux-gnu.cmake
#   cmake --build build_riscv
#
# Prerequisites:
#   Download toolchain: https://github.com/riscv-collab/riscv-gnu-toolchain/releases
#   sudo apt install qemu-user
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR
#
# To customize the architecture, set RISCV_MARCH before configuring:
#   cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/riscv64-linux-gnu.cmake -D RISCV_MARCH=rv64gcv

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross-compiler from riscv-collab toolchain
# Set RISCV_TOOLCHAIN_PATH to override default location
if(NOT DEFINED RISCV_TOOLCHAIN_PATH)
    set(RISCV_TOOLCHAIN_PATH "/tmp/riscv")
endif()

set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-g++)

# Sysroot for the target system
set(CMAKE_SYSROOT ${RISCV_TOOLCHAIN_PATH}/sysroot)

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
if(NOT DEFINED RISCV_MARCH)
    set(RISCV_MARCH "rv64gcv_zvfh_zvfbfwma_zvbb")
endif()

if(NOT DEFINED RISCV_MABI)
    set(RISCV_MABI "lp64d")
endif()

set(CMAKE_C_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")
set(CMAKE_CXX_FLAGS_INIT "-march=${RISCV_MARCH} -mabi=${RISCV_MABI}")

# QEMU user-mode emulation for running tests
# -L sysroot provides the dynamic linker path
# -cpu max enables all supported extensions (including zvfh, zvfbfwma if available)
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-riscv64;-L;${RISCV_TOOLCHAIN_PATH}/sysroot;-cpu;max")

# Search paths for libraries and headers (target system only)
set(CMAKE_FIND_ROOT_PATH ${RISCV_TOOLCHAIN_PATH}/sysroot)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
