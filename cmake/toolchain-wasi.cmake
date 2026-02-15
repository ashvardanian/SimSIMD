# WASI toolchain for NumKong (standalone WASM runtimes: Wasmer, Wasmtime)
# Usage: cmake -B build-wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake -DNK_BUILD_TEST=ON

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

# Locate WASI SDK
if(NOT DEFINED WASI_SDK_PATH)
    if(DEFINED ENV{WASI_SDK_PATH})
        set(WASI_SDK_PATH "$ENV{WASI_SDK_PATH}")
    elseif(EXISTS "$ENV{HOME}/wasi-sdk")
        set(WASI_SDK_PATH "$ENV{HOME}/wasi-sdk")
    else()
        message(FATAL_ERROR
            "WASI_SDK_PATH not set and ~/wasi-sdk not found.\n"
            "Download from: https://github.com/WebAssembly/wasi-sdk/releases\n"
            "Then set: export WASI_SDK_PATH=~/wasi-sdk")
    endif()
endif()

# Set compilers
set(CMAKE_C_COMPILER "${WASI_SDK_PATH}/bin/clang")
set(CMAKE_CXX_COMPILER "${WASI_SDK_PATH}/bin/clang++")
set(CMAKE_AR "${WASI_SDK_PATH}/bin/llvm-ar")
set(CMAKE_RANLIB "${WASI_SDK_PATH}/bin/llvm-ranlib")
set(CMAKE_SYSROOT "${WASI_SDK_PATH}/share/wasi-sysroot")
set(CMAKE_FIND_ROOT_PATH "${WASI_SDK_PATH}")

# WASM SIMD flags (same as Emscripten for consistency)
set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=wasm32-wasi")
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=wasm32-wasi")

# Optimization flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Linker flags for WASI
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,--allow-undefined \
     -Wl,--export=main \
     -Wl,--export=_start")

# Don't look for programs in build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Verify WASI SDK version
if(EXISTS "${WASI_SDK_PATH}/VERSION")
    file(READ "${WASI_SDK_PATH}/VERSION" WASI_SDK_VERSION)
    string(STRIP "${WASI_SDK_VERSION}" WASI_SDK_VERSION)
    message(STATUS "NumKong WASI: WASI-SDK ${WASI_SDK_VERSION}")
else()
    message(STATUS "NumKong WASI: WASI-SDK version unknown")
endif()

message(STATUS "NumKong WASI: SIMD128 and Relaxed SIMD enabled")
message(STATUS "NumKong WASI: Toolchain at ${WASI_SDK_PATH}")
