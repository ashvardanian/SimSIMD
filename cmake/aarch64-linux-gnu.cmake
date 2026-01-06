# AArch64 Linux cross-compilation toolchain for NumKong
#
# Usage:
#   cmake -B build_arm64 -D CMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake
#   cmake --build build_arm64
#
# Prerequisites:
#   sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR
#
# To customize the architecture, set AARCH64_MARCH before configuring:
#   cmake -B build_arm64 -D CMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake -D AARCH64_MARCH=armv8.2-a+sve

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler from standard Ubuntu/Debian packages
# Set AARCH64_TOOLCHAIN_PREFIX to override (e.g., for custom toolchain path)
if(NOT DEFINED AARCH64_TOOLCHAIN_PREFIX)
    set(AARCH64_TOOLCHAIN_PREFIX "aarch64-linux-gnu-")
endif()

set(CMAKE_C_COMPILER ${AARCH64_TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${AARCH64_TOOLCHAIN_PREFIX}g++)

# Target architecture with modern Arm extensions
# Default: Enables all extensions supported by NumKong
#
# Architecture hierarchy:
#   armv8-a      : Base AArch64 (NEON mandatory)
#   armv8.2-a    : +fp16 (half-precision), +dotprod available
#   armv8.4-a    : +bf16 (bfloat16) available
#   armv8.6-a    : +i8mm (int8 matrix multiply) available
#   armv9-a      : SVE2 mandatory, +sme available
#   armv9.2-a    : +sme2 available
#
# NumKong Arm targets and required extensions:
#   NK_TARGET_NEON      : Base NEON (automatic with aarch64)
#   NK_TARGET_NEONHALF  : +fp16
#   NK_TARGET_NEONBFDOT : +bf16
#   NK_TARGET_NEONFHM   : +fp16fml (FMLAL/FMLSL widening ops)
#   NK_TARGET_NEONSDOT  : +dotprod
#   NK_TARGET_SVE       : +sve
#   NK_TARGET_SVEHALF   : +sve +fp16
#   NK_TARGET_SVEBFDOT  : +sve +bf16
#   NK_TARGET_SVE2      : +sve2
#   NK_TARGET_SME       : +sme
#   NK_TARGET_SME2      : +sme2
#
# Common CPU profiles:
#   armv8.2-a+fp16+dotprod                       : Apple M1, Ampere Altra
#   armv8.4-a+sve+fp16+bf16+i8mm                 : AWS Graviton 3, Neoverse V1
#   armv9-a+sve2+fp16+bf16+i8mm                  : Neoverse V2, Cortex-X3
#   armv9.2-a+sve2+sme2+fp16+bf16+i8mm+fp16fml   : Future SME2-capable CPUs
#
if(NOT DEFINED AARCH64_MARCH)
    set(AARCH64_MARCH "armv9.2-a+sve2+sme2+fp16+bf16+i8mm+dotprod+fp16fml")
endif()

set(CMAKE_C_FLAGS_INIT "-march=${AARCH64_MARCH}")
set(CMAKE_CXX_FLAGS_INIT "-march=${AARCH64_MARCH}")

# QEMU user-mode emulation for running tests
# -cpu max enables all supported features
# For specific CPU emulation: -cpu neoverse-v1, -cpu neoverse-v2, etc.
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-aarch64;-cpu;max")

# Search paths for libraries and headers (target system only)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
