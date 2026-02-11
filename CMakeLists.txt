# CMake is the preferred way to compile the C shared libraries, tests, and benchmarks.
# Use like:
#
#   cmake -B build_release -D CMAKE_BUILD_TYPE=Release -D NK_BUILD_TEST=ON
#   cmake --build build_release --config Release
#   build_release/nk_test
#
# CMake options:
#
#   -D CMAKE_BUILD_TYPE=<Release|Debug|RelWithDebInfo>
#   -D NK_BUILD_TEST=<ON|OFF>           (default: OFF)
#   -D NK_BUILD_BENCH=<ON|OFF>          (default: OFF)
#   -D NK_COMPARE_TO_BLAS=<ON|OFF>      (default: OFF)
#   -D NK_COMPARE_TO_MKL=<ON|OFF>       (default: OFF)
#   -D NK_BUILD_SHARED=<ON|OFF>         (default: OFF)
#
# Different SIMD ISA extensions reqiore different compiler versions:
#
#   - Basic functionality:  GCC 9+, Clang 10+, MSVC 2019+
#   - Float16 support:      GCC 12+, Clang 16+
#   - Intel AMX/Arm SME:    GCC 14+, Clang 18+
#
# To run on macOS with Homebrew LLVM:
#
#     brew install llvm openblas ccache
#     cmake -B build_release -D CMAKE_BUILD_TYPE=Release \
#           -D NK_BUILD_TEST=ON \
#           -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
#           -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
#           -D CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES="$(brew --prefix openblas)/include"
#     cmake --build build_release --config Release
#
# To cross-compile for ARM64:
#
#     cmake -B build_arm64 -D CMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake \
#           -D NK_BUILD_TEST=ON
#
# To cross-compile for RISC-V:
#
#     cmake -B build_riscv -D CMAKE_TOOLCHAIN_FILE=cmake/riscv64-linux-gnu.cmake \
#           -D NK_BUILD_TEST=ON
#
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Leverage `ccache` for incremental compilation
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    message(STATUS "IMPORTANT: Set environment variable CCACHE_SLOPPINESS=pch_defines,time_macros")
endif()

# 
project(
    numkong
    VERSION 6.5.12
    LANGUAGES C CXX
    DESCRIPTION "Portable mixed-precision BLAS-like vector math library for x86 and ARM"
    HOMEPAGE_URL "https://github.com/ashvardanian/numkong"
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED YES)
# Enable GNU extensions for WASM (required for EM_ASM runtime detection)
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(CMAKE_C_EXTENSIONS YES)
else()
    set(CMAKE_C_EXTENSIONS NO)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Determine if NumKong is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(NK_IS_MAIN_PROJECT OFF)
set(NK_IS_WASI_PROJECT OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NK_IS_MAIN_PROJECT ON)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "WASI")
    set(NK_IS_WASI_PROJECT ON)
endif()

option(NK_BUILD_SHARED "Compile a dynamic library" ${NK_IS_MAIN_PROJECT})
option(NK_BUILD_TEST "Compile precision tests with ULP-based error analysis" OFF)
option(NK_BUILD_BENCH "Compile micro-benchmarks for current ISA" OFF)

# BLAS/MKL options: ON (require), OFF (disable), AUTO (detect if available)
set(NK_COMPARE_TO_BLAS "AUTO" CACHE STRING "Include BLAS (OpenBLAS/Accelerate) in tests and benchmarks")
set(NK_COMPARE_TO_MKL "AUTO" CACHE STRING "Include Intel MKL in tests and benchmarks")
set_property(CACHE NK_COMPARE_TO_BLAS PROPERTY STRINGS "AUTO" "ON" "OFF")
set_property(CACHE NK_COMPARE_TO_MKL PROPERTY STRINGS "AUTO" "ON" "OFF")

# Default to Release build type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# MSVC does not set `CMAKE_SYSTEM_PROCESSOR` correctly
# Use `CMAKE_C_COMPILER_ARCHITECTURE_ID` instead
if(MSVC)
    if(CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "x64")
        set(CMAKE_SYSTEM_PROCESSOR "AMD64")
    elseif(CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "X86")
        set(CMAKE_SYSTEM_PROCESSOR "X86")
    elseif(CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "ARM64")
        set(CMAKE_SYSTEM_PROCESSOR "ARM64")
    else()
        message(WARNING "Unknown CMAKE_C_COMPILER_ARCHITECTURE_ID=${CMAKE_C_COMPILER_ARCHITECTURE_ID}")
    endif()
endif()

# Detect target architecture from CMAKE_SYSTEM_PROCESSOR (safe for cross-compilation)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64")
    set(NK_IS_64BIT_X86_ TRUE)
    message(STATUS "NumKong Platform: x86_64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(NK_IS_64BIT_ARM_ TRUE)
    message(STATUS "NumKong Platform: ARM64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(NK_IS_64BIT_RISCV_ TRUE)
    message(STATUS "NumKong Platform: RISC-V 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Global compiler flags for debug and release
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
check_cxx_source_compiles("int main(){return 0;}" NK_HAS_ASAN)
unset(CMAKE_REQUIRED_FLAGS)

if(NK_HAS_ASAN)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
    set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_C_FLAGS_DEBUG "-g")
    message(STATUS "AddressSanitizer not available, Debug builds without ASAN")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Compiler-specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    if(NOT APPLE AND NOT CMAKE_CROSSCOMPILING)
        add_compile_options(-march=native)
    endif()

    add_compile_options(-pedantic -ferror-limit=1 -Wno-typedef-redefinition)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(NOT CMAKE_CROSSCOMPILING)
        add_compile_options(-march=native)
    endif()
    add_compile_options(-pedantic -fmax-errors=1 -Wno-tautological-constant-compare)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-w -ferror-limit=1)
endif()

# Define the header-only library
file(GLOB NK_SOURCES include/numkong/*.h)
add_library(numkong INTERFACE)
target_sources(numkong INTERFACE ${NK_SOURCES})
target_include_directories(numkong INTERFACE "${PROJECT_SOURCE_DIR}/include")

# Enable Clang C modules for incremental compilation
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(numkong INTERFACE
        $<$<COMPILE_LANGUAGE:C>:-fmodules>
        $<$<COMPILE_LANGUAGE:C>:-fimplicit-module-maps>
        $<$<COMPILE_LANGUAGE:C>:-fmodules-cache-path=${CMAKE_BINARY_DIR}/module-cache>
    )

    # Install module map
    install(FILES ${CMAKE_SOURCE_DIR}/include/module.modulemap
        DESTINATION include)

    message(STATUS "Clang C modules enabled: cache at ${CMAKE_BINARY_DIR}/module-cache")
endif()


# Find OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_LIBRARIES}")
else()
    message(WARNING "OpenMP not found. Parallel execution will be disabled.")
endif()


# BLAS/MKL detection (shared by tests and benchmarks)
# Options can be: ON (require), OFF (disable), AUTO (detect if available)
set(NK_BLAS_FOUND FALSE)
set(NK_MKL_FOUND FALSE)

# Skip BLAS/MKL/Accelerate detection when cross-compiling (host libraries won't work)
if(CMAKE_CROSSCOMPILING)
    if(NK_COMPARE_TO_MKL STREQUAL "AUTO")
        message(STATUS "Cross-compiling: skipping MKL detection (host library)")
        set(NK_COMPARE_TO_MKL "OFF")
    endif()
    if(NK_COMPARE_TO_BLAS STREQUAL "AUTO")
        message(STATUS "Cross-compiling: skipping BLAS detection (host library)")
        set(NK_COMPARE_TO_BLAS "OFF")
    endif()
endif()

# Only attempt detection if building tests or benchmarks
if(NK_BUILD_TEST OR NK_BUILD_BENCH)

    # MKL detection
    if(NOT NK_COMPARE_TO_MKL STREQUAL "OFF")
        # Use environment variable MKLROOT or standard MKL paths
        if(DEFINED ENV{MKLROOT})
            set(MKL_ROOT $ENV{MKLROOT})
        else()
            # Try common installation paths
            if(EXISTS "/opt/intel/oneapi/mkl/latest")
                set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
            elseif(EXISTS "/opt/intel/mkl")
                set(MKL_ROOT "/opt/intel/mkl")
            endif()
        endif()

        if(MKL_ROOT)
            # Verify the libraries actually exist (prefer shared libraries for simpler linking)
            if(EXISTS "${MKL_ROOT}/lib/libmkl_intel_lp64.so")
                message(STATUS "MKL found at: ${MKL_ROOT}")
                set(NK_MKL_FOUND TRUE)
                set(NK_MKL_INCLUDE_DIRS ${MKL_ROOT}/include)
                set(NK_MKL_LIBRARIES
                    ${MKL_ROOT}/lib/libmkl_intel_lp64.so
                    ${MKL_ROOT}/lib/libmkl_sequential.so
                    ${MKL_ROOT}/lib/libmkl_core.so
                    pthread m dl)
            elseif(EXISTS "${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a")
                # Fallback to static libraries
                message(STATUS "MKL found at: ${MKL_ROOT} (static)")
                set(NK_MKL_FOUND TRUE)
                set(NK_MKL_INCLUDE_DIRS ${MKL_ROOT}/include)
                set(NK_MKL_LIBRARIES
                    -Wl,--start-group
                    ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a
                    ${MKL_ROOT}/lib/intel64/libmkl_sequential.a
                    ${MKL_ROOT}/lib/intel64/libmkl_core.a
                    -Wl,--end-group
                    pthread m dl)
            elseif(NK_COMPARE_TO_MKL STREQUAL "ON")
                message(FATAL_ERROR "MKL libraries not found at ${MKL_ROOT}/lib/")
            else()
                message(STATUS "MKL directory found but libraries missing, skipping MKL")
            endif()
        elseif(NK_COMPARE_TO_MKL STREQUAL "ON")
            message(FATAL_ERROR "MKL not found. Set MKLROOT environment variable or install Intel oneAPI.")
        else()
            message(STATUS "MKL not found (auto-detection), skipping MKL benchmarks")
        endif()
    endif()

    # BLAS detection (skip if MKL found in AUTO mode, since MKL provides BLAS)
    # We differentiate between:
    # - NK_MKL_FOUND: Intel MKL
    # - NK_ACCELERATE_FOUND: Apple Accelerate framework
    # - NK_BLAS_FOUND: Generic CBLAS (OpenBLAS, etc.)
    if(NOT NK_COMPARE_TO_BLAS STREQUAL "OFF")
        if(NK_MKL_FOUND AND NOT NK_COMPARE_TO_BLAS STREQUAL "ON")
            message(STATUS "BLAS: Using MKL (skipping separate BLAS detection)")
        elseif(APPLE)
            # Apple Accelerate is always available on macOS
            message(STATUS "Using Apple Accelerate framework")
            set(NK_ACCELERATE_FOUND TRUE)
            set(NK_ACCELERATE_LIBRARIES "-framework Accelerate")
            set(NK_ACCELERATE_INCLUDE_DIRS "")
        else()
            # Try to find BLAS without REQUIRED flag for auto-detection
            find_package(BLAS QUIET)
            if(BLAS_FOUND)
                message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
                set(NK_BLAS_FOUND TRUE)
                set(NK_BLAS_LIBRARIES ${BLAS_LIBRARIES})
                set(NK_BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIRS})
            elseif(NK_COMPARE_TO_BLAS STREQUAL "ON")
                message(FATAL_ERROR "BLAS not found. Install OpenBLAS or set NK_COMPARE_TO_BLAS=OFF")
            else()
                message(STATUS "BLAS not found (auto-detection), skipping BLAS benchmarks")
            endif()
        endif()
    endif()

    # Summary
    if(NK_BLAS_FOUND OR NK_MKL_FOUND OR NK_ACCELERATE_FOUND)
        message(STATUS "Comparison libraries: BLAS=${NK_BLAS_FOUND}, MKL=${NK_MKL_FOUND}, Accelerate=${NK_ACCELERATE_FOUND}")
    endif()

endif()


# Build benchmarks
if(NK_BUILD_BENCH)
    # Fetch external dependencies
    include(FetchContent)

    # Suppress building tests of Google Benchmark
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_INSTALL OFF)
    set(BENCHMARK_ENABLE_DOXYGEN OFF)
    set(BENCHMARK_INSTALL_DOCS OFF)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    set(BENCHMARK_USE_BUNDLED_GTEST ON)

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
    )
    FetchContent_MakeAvailable(benchmark)

    # Remove the Google Benchmark's "built in debug warning"
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_definitions(benchmark PRIVATE NDEBUG)
    endif()

    find_package(Threads REQUIRED)
    add_executable(nk_bench
        bench/bench.cpp
        bench/bench_dot.cpp
        bench/bench_spatial.cpp
        bench/bench_set.cpp
        bench/bench_curved.cpp
        bench/bench_probability.cpp
        bench/bench_each.cpp
        bench/bench_trigonometry.cpp
        bench/bench_geospatial.cpp
        bench/bench_mesh.cpp
        bench/bench_sparse.cpp
        bench/bench_cast.cpp
        bench/bench_reduce.cpp
        # ISA-family cross/batch files for parallel compilation
        bench/bench_cross_serial.cpp
        bench/bench_cross_x86.cpp
        bench/bench_cross_amx.cpp
        bench/bench_cross_arm.cpp
        bench/bench_cross_sme.cpp
        bench/bench_cross_blas.cpp
    )
    target_link_libraries(nk_bench numkong Threads::Threads benchmark)

    if(OpenMP_FOUND)
        target_compile_options(nk_bench PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(nk_bench OpenMP::OpenMP_CXX)
    endif()

    if(NK_BLAS_FOUND)
        target_compile_definitions(nk_bench PRIVATE NK_COMPARE_TO_BLAS=1)
        target_include_directories(nk_bench PRIVATE ${NK_BLAS_INCLUDE_DIRS})
        target_link_libraries(nk_bench ${NK_BLAS_LIBRARIES})
    endif()

    if(NK_ACCELERATE_FOUND)
        target_compile_definitions(nk_bench PRIVATE NK_COMPARE_TO_ACCELERATE=1 ACCELERATE_NEW_LAPACK)
        target_include_directories(nk_bench PRIVATE ${NK_ACCELERATE_INCLUDE_DIRS})
        target_link_libraries(nk_bench ${NK_ACCELERATE_LIBRARIES})
    endif()

    if(NK_MKL_FOUND)
        target_compile_definitions(nk_bench PRIVATE NK_COMPARE_TO_MKL=1)
        target_include_directories(nk_bench PRIVATE ${NK_MKL_INCLUDE_DIRS})
        target_link_libraries(nk_bench ${NK_MKL_LIBRARIES})
    endif()

endif()


# Dtype-specific dispatch sources (for parallel compilation)
set(NK_DTYPE_SOURCES
    c/dispatch_f64.c c/dispatch_f32.c c/dispatch_f16.c c/dispatch_bf16.c
    c/dispatch_i8.c c/dispatch_u8.c c/dispatch_i4.c c/dispatch_u4.c
    c/dispatch_e4m3.c c/dispatch_e5m2.c c/dispatch_e2m3.c c/dispatch_e3m2.c
    c/dispatch_u1.c
    c/dispatch_f64c.c c/dispatch_f32c.c c/dispatch_f16c.c c/dispatch_bf16c.c
    c/dispatch_i16.c c/dispatch_u16.c c/dispatch_i32.c c/dispatch_u32.c
    c/dispatch_i64.c c/dispatch_u64.c
    c/dispatch_cast.c
)

# Build tests
if(NK_BUILD_TEST)
    # C++ precision test (ULP-based error analysis)
    add_executable(nk_test
        test/test.cpp
        test/test_dot.cpp
        test/test_spatial.cpp
        test/test_set.cpp
        test/test_curved.cpp
        test/test_probability.cpp
        test/test_each.cpp
        test/test_trigonometry.cpp
        test/test_geospatial.cpp
        test/test_mesh.cpp
        test/test_cast.cpp
        test/test_reduce.cpp
        test/test_sparse.cpp
        # ISA-family cross/batch files for parallel compilation
        test/test_cross_serial.cpp
        test/test_cross_x86.cpp
        test/test_cross_amx.cpp
        test/test_cross_arm.cpp
        test/test_cross_sme.cpp
        test/test_cross_blas.cpp
        test/test_tensor.cpp
    )
    target_link_libraries(nk_test numkong m)

    # WASI-specific configuration
    if(NK_IS_WASI_PROJECT)
        target_compile_definitions(nk_test PRIVATE
            NK_WASI_BUILD=1
            NK_DYNAMIC_DISPATCH=1)
        set_target_properties(nk_test PROPERTIES OUTPUT_NAME "test.wasm")
        target_link_options(nk_test PRIVATE
            -Wl,--export=main
            -Wl,--export=_start)
        message(STATUS "NumKong: Building tests for WASI (output: test.wasm)")
    endif()

    # Native-only dependencies (not available in WASI)
    if(OpenMP_FOUND AND NOT NK_IS_WASI_PROJECT)
        target_compile_options(nk_test PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(nk_test OpenMP::OpenMP_CXX)
        target_compile_definitions(nk_test PRIVATE NK_TEST_USE_OPENMP=1)
    endif()

    if(NK_BLAS_FOUND AND NOT NK_IS_WASI_PROJECT)
        target_compile_definitions(nk_test PRIVATE NK_COMPARE_TO_BLAS=1)
        target_include_directories(nk_test PRIVATE ${NK_BLAS_INCLUDE_DIRS})
        target_link_libraries(nk_test ${NK_BLAS_LIBRARIES})
    endif()

    if(NK_ACCELERATE_FOUND AND NOT NK_IS_WASI_PROJECT)
        target_compile_definitions(nk_test PRIVATE NK_COMPARE_TO_ACCELERATE=1 ACCELERATE_NEW_LAPACK)
        target_include_directories(nk_test PRIVATE ${NK_ACCELERATE_INCLUDE_DIRS})
        target_link_libraries(nk_test ${NK_ACCELERATE_LIBRARIES})
    endif()

    if(NK_MKL_FOUND AND NOT NK_IS_WASI_PROJECT)
        target_compile_definitions(nk_test PRIVATE NK_COMPARE_TO_MKL=1)
        target_include_directories(nk_test PRIVATE ${NK_MKL_INCLUDE_DIRS})
        target_link_libraries(nk_test ${NK_MKL_LIBRARIES})
    endif()

    add_test(NAME nk_test COMMAND nk_test)
    enable_testing()
endif()

# Shared library (or Emscripten JS module)
if(NK_BUILD_SHARED)
    set(NK_SOURCES ${NK_SOURCES} c/numkong.c ${NK_DTYPE_SOURCES})

    # Emscripten requires building as executable to generate JS bindings
    if(EMSCRIPTEN)
        # Create a minimal main() wrapper for Emscripten
        file(WRITE "${CMAKE_BINARY_DIR}/emscripten_stub.c"
            "// Stub main for Emscripten to generate JS bindings\nint main() { return 0; }\n")

        add_executable(nk_shared ${NK_SOURCES} "${CMAKE_BINARY_DIR}/emscripten_stub.c")
        target_include_directories(nk_shared PUBLIC "${PROJECT_SOURCE_DIR}/include")
        set_target_properties(nk_shared PROPERTIES
            OUTPUT_NAME "numkong"
            SUFFIX ".js" # Generates numkong.js + numkong.wasm
        )

        target_link_options(nk_shared PRIVATE
            "SHELL:-s EXPORT_ES6=1" # Generate ES6 module
            "SHELL:-s MODULARIZE=1" # Wrap in module factory
            "SHELL:-s EXPORT_NAME=NumKongModule" # Export name
            "SHELL:-s EXPORTED_FUNCTIONS=_malloc,_free,_nk_dot_f32,_nk_dot_f64,_nk_dot_f16,_nk_dot_bf16,_nk_dot_i8,_nk_dot_u8,_nk_angular_f32,_nk_angular_f64,_nk_angular_f16,_nk_angular_bf16,_nk_angular_i8,_nk_angular_u8,_nk_sqeuclidean_f32,_nk_sqeuclidean_f64,_nk_sqeuclidean_f16,_nk_sqeuclidean_bf16,_nk_sqeuclidean_i8,_nk_sqeuclidean_u8,_nk_euclidean_f32,_nk_euclidean_f64,_nk_euclidean_f16,_nk_euclidean_bf16,_nk_euclidean_i8,_nk_euclidean_u8,_nk_hamming_u1,_nk_hamming_u8,_nk_jaccard_u1,_nk_jaccard_u16,_nk_kld_f32,_nk_kld_f64,_nk_jsd_f32,_nk_jsd_f64,_nk_capabilities" # Direct exports (35 functions)
            "SHELL:-s EXPORTED_RUNTIME_METHODS=wasmMemory" # Export memory object
            "SHELL:-s ALLOW_MEMORY_GROWTH=1" # Dynamic memory
            "SHELL:-s ENVIRONMENT=web,node" # Both browser and Node.js
            "SHELL:-s SINGLE_FILE=0" # Separate .wasm file (recommended)
            "SHELL:-s EXPORT_ALL=0" # Security - don't export everything
            "SHELL:-s STANDALONE_WASM=0" # Keep JS wrapper (required for browser)
            "SHELL:-s INITIAL_MEMORY=16MB" # Initial heap size
            "SHELL:-s STACK_SIZE=5MB" # Stack size for recursion
        )
    else()
        add_library(nk_shared SHARED ${NK_SOURCES})
        target_include_directories(nk_shared PUBLIC "${PROJECT_SOURCE_DIR}/include")
        set_target_properties(nk_shared PROPERTIES OUTPUT_NAME numkong)
    endif()

    install(
        TARGETS nk_shared
        ARCHIVE
        BUNDLE
        FRAMEWORK
        LIBRARY
        OBJECTS
        PRIVATE_HEADER
        PUBLIC_HEADER
        RESOURCE
        RUNTIME
    )
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY c/ DESTINATION share/doc/${PROJECT_NAME}/src)
