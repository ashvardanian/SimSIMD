# LLVM Clang toolchain for RISC-V 64-bit Linux builds.
#
# Required inputs:
#   -D RISCV_SYSROOT=/path/to/riscv64/sysroot
# Optional inputs:
#   -D LLVM_ROOT=/path/to/llvm
#   -D RISCV_TARGET=riscv64-unknown-linux-gnu
#   -D RISCV_MARCH=rv64gcv_zvfh_zvfbfwma_zvbb
#   -D RISCV_MABI=lp64d

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if (NOT DEFINED RISCV_SYSROOT)
    if (DEFINED ENV{RISCV_SYSROOT})
        set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}")
    else ()
        message(FATAL_ERROR "RISCV_SYSROOT must point to a riscv64 Linux sysroot for LLVM cross-compilation.")
    endif ()
endif ()

if (NOT DEFINED RISCV_TARGET)
    set(RISCV_TARGET "riscv64-unknown-linux-gnu")
endif ()

if (NOT DEFINED RISCV_MARCH)
    set(RISCV_MARCH "rv64gcv_zvfh_zvfbfwma_zvbb")
endif ()

if (NOT DEFINED RISCV_MABI)
    set(RISCV_MABI "lp64d")
endif ()

if (NOT DEFINED LLVM_ROOT)
    if (EXISTS "/opt/homebrew/opt/llvm/bin/clang")
        set(LLVM_ROOT "/opt/homebrew/opt/llvm")
    elseif (EXISTS "/usr/local/opt/llvm/bin/clang")
        set(LLVM_ROOT "/usr/local/opt/llvm")
    endif ()
endif ()

if (DEFINED LLVM_ROOT)
    set(_NK_LLVM_BIN "${LLVM_ROOT}/bin")
    set(CMAKE_C_COMPILER "${_NK_LLVM_BIN}/clang")
    set(CMAKE_CXX_COMPILER "${_NK_LLVM_BIN}/clang++")
    set(CMAKE_AR "${_NK_LLVM_BIN}/llvm-ar")
    set(CMAKE_RANLIB "${_NK_LLVM_BIN}/llvm-ranlib")
    if (EXISTS "${_NK_LLVM_BIN}/ld.lld")
        set(CMAKE_LINKER "${_NK_LLVM_BIN}/ld.lld")
    endif ()
else ()
    find_program(CMAKE_C_COMPILER clang REQUIRED)
    find_program(CMAKE_CXX_COMPILER clang++ REQUIRED)
    find_program(CMAKE_AR llvm-ar REQUIRED)
    find_program(CMAKE_RANLIB llvm-ranlib REQUIRED)
    find_program(CMAKE_LINKER ld.lld)
endif ()

set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
set(CMAKE_C_COMPILER_TARGET "${RISCV_TARGET}")
set(CMAKE_CXX_COMPILER_TARGET "${RISCV_TARGET}")

set(_NK_RISCV_FLAGS "--target=${RISCV_TARGET} --sysroot=${RISCV_SYSROOT} -march=${RISCV_MARCH} -mabi=${RISCV_MABI}")
set(CMAKE_C_FLAGS_INIT "${_NK_RISCV_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_NK_RISCV_FLAGS}")

find_program(_NK_QEMU_RISCV64 qemu-riscv64)
if (_NK_QEMU_RISCV64)
    set(CMAKE_CROSSCOMPILING_EMULATOR "${_NK_QEMU_RISCV64};-L;${RISCV_SYSROOT};-cpu;max")
endif ()

set(CMAKE_FIND_ROOT_PATH "${RISCV_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
