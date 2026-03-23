# Power ppc64le GNU toolchain for NumKong.
#
# Usage:
#   cmake -B build_ppc -D CMAKE_TOOLCHAIN_FILE=cmake/toolchain-ppc64le-gnu.cmake
#   cmake --build build_ppc
#
# Optional inputs:
#   -D PPC_TOOLCHAIN_PATH=/path/to/powerpc64le-linux-gnu
#   -D PPC_SYSROOT=/path/to/ppc64le/sysroot
#   -D PPC_MCPU=power10
#
# Prerequisites:
#   sudo apt install gcc-powerpc64le-linux-gnu g++-powerpc64le-linux-gnu libc6-dev-ppc64el-cross qemu-user
#
# Testing with QEMU:
#   Tests will automatically run under QEMU via CMAKE_CROSSCOMPILING_EMULATOR

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

# Cross-compiler from the system package (gcc-powerpc64le-linux-gnu).
set(CMAKE_C_COMPILER powerpc64le-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER powerpc64le-linux-gnu-g++)

# Sysroot for the target system.
if (NOT DEFINED PPC_SYSROOT)
    if (DEFINED ENV{PPC_SYSROOT})
        set(PPC_SYSROOT "$ENV{PPC_SYSROOT}")
    else ()
        # Default: derive from the cross-compiler
        execute_process(
            COMMAND powerpc64le-linux-gnu-gcc -print-sysroot OUTPUT_VARIABLE PPC_SYSROOT
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        )
        if (NOT PPC_SYSROOT OR PPC_SYSROOT STREQUAL "")
            set(PPC_SYSROOT "/usr/powerpc64le-linux-gnu")
        endif ()
    endif ()
endif ()
set(PPC_SYSROOT "${PPC_SYSROOT}" CACHE PATH "ppc64le sysroot for GNU cross-compilation")
set(CMAKE_SYSROOT "${PPC_SYSROOT}")

# Target CPU microarchitecture.
# power10: VSX + MMA + native bf16 conversions.
# power8:  VSX + vec_popcnt (minimum for NumKong powervsx backend).
if (NOT DEFINED PPC_MCPU)
    if (DEFINED ENV{PPC_MCPU})
        set(PPC_MCPU "$ENV{PPC_MCPU}")
    else ()
        set(PPC_MCPU "power10")
    endif ()
endif ()
set(PPC_MCPU "${PPC_MCPU}" CACHE STRING "Power CPU target for cross-compilation")

# Forward settings to nested try_compile() invocations.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES PPC_SYSROOT PPC_MCPU)

# Toolchain validation only needs to prove the project compiles.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_C_FLAGS_INIT "-mcpu=${PPC_MCPU} -mvsx")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=${PPC_MCPU} -mvsx")

# QEMU user-mode emulation for running tests.
# -L sysroot provides the dynamic linker path.
# -cpu max enables all supported extensions.
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-ppc64le;-L;${PPC_SYSROOT};-cpu;max")

# Search paths for libraries and headers (target system only).
set(CMAKE_FIND_ROOT_PATH "${PPC_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
