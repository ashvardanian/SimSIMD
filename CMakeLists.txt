# NumKong library CMakeLists.txt
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(
    numkong
    VERSION 6.5.12
    LANGUAGES C CXX
    DESCRIPTION "Portable mixed-precision BLAS-like vector math library for x86 and ARM"
    HOMEPAGE_URL "https://github.com/ashvardanian/numkong"
)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED YES)
set(CMAKE_C_EXTENSIONS NO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Determine if NumKong is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(NK_IS_MAIN_PROJECT OFF)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NK_IS_MAIN_PROJECT ON)
endif ()

option(NK_BUILD_SHARED "Compile a dynamic library" ${NK_IS_MAIN_PROJECT})
option(NK_BUILD_TESTS "Small compilation tests compile-time and run-time dispatch" OFF)
option(NK_BUILD_BENCHMARKS "Compile micro-benchmarks for current ISA" OFF)
option(NK_BUILD_BENCHMARKS_WITH_CBLAS "Include OpenBLAS in micro-kernel benchmarks" OFF)
option(NK_BUILD_BENCHMARKS_WITH_MKL "Include Intel MKL in micro-kernel benchmarks" OFF)

# Default to Release build type if not set
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# MSVC does not set `CMAKE_SYSTEM_PROCESSOR` correctly
# Use `CMAKE_C_COMPILER_ARCHITECTURE_ID` instead
if (MSVC)
    if (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "x64")
        set(CMAKE_SYSTEM_PROCESSOR "AMD64")
    elseif (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "X86")
        set(CMAKE_SYSTEM_PROCESSOR "X86")
    elseif (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "ARM64")
        set(CMAKE_SYSTEM_PROCESSOR "ARM64")
    else ()
        message(WARNING "Unknown CMAKE_C_COMPILER_ARCHITECTURE_ID=${CMAKE_C_COMPILER_ARCHITECTURE_ID}")
    endif ()
endif ()

# Detect target architecture from CMAKE_SYSTEM_PROCESSOR (safe for cross-compilation)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64")
    set(NK_IS_64BIT_X86_ TRUE)
    message(STATUS "NumKong Platform: x86_64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(NK_IS_64BIT_ARM_ TRUE)
    message(STATUS "NumKong Platform: ARM64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# Global compiler flags for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Compiler-specific flags
if (CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    if (NOT APPLE)
        add_compile_options(-march=native)
    endif ()

    add_compile_options(-pedantic -ferror-limit=1)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-march=native -pedantic -fmax-errors=1 -Wno-tautological-constant-compare)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(-w -ferror-limit=1)
endif ()

# Define the header-only library
file(GLOB NK_SOURCES include/numkong/*.h)
add_library(numkong INTERFACE)
target_sources(numkong INTERFACE ${NK_SOURCES})
target_include_directories(numkong INTERFACE "${PROJECT_SOURCE_DIR}/include")


# Find OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_LIBRARIES}")
else ()
    message(WARNING "OpenMP not found. Parallel execution will be disabled.")
endif ()


# Build benchmarks if required
if (NK_BUILD_BENCHMARKS)
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
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_definitions(benchmark PRIVATE NDEBUG)
    endif ()

    find_package(Threads REQUIRED)
    add_executable(nk_bench scripts/bench.cxx)
    target_link_libraries(nk_bench numkong Threads::Threads benchmark)

    if (OpenMP_FOUND)
        target_compile_options(nk_bench PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(nk_bench OpenMP::OpenMP_CXX)
    endif ()

    if (NK_BUILD_BENCHMARKS_WITH_CBLAS)
        find_package(BLAS REQUIRED)
        if (BLAS_FOUND)
            message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
            include_directories(${BLAS_INCLUDE_DIRS})
            target_include_directories(nk_bench PRIVATE ${BLAS_INCLUDE_DIRS})
            target_link_libraries(nk_bench ${BLAS_LIBRARIES})
            target_compile_definitions(nk_bench PRIVATE NK_BUILD_BENCHMARKS_WITH_CBLAS=1)
        else ()
            message(FATAL_ERROR "BLAS not found")
        endif ()
    endif ()

    if (NK_BUILD_BENCHMARKS_WITH_MKL)
        # Use environment variable MKLROOT or standard MKL paths
        if (DEFINED ENV{MKLROOT})
            set(MKL_ROOT $ENV{MKLROOT})
        else ()
            # Try common installation paths
            if (EXISTS "/opt/intel/oneapi/mkl/latest")
                set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
            elseif (EXISTS "/opt/intel/mkl")
                set(MKL_ROOT "/opt/intel/mkl")
            endif ()
        endif ()

        if (MKL_ROOT)
            message(STATUS "MKL found at: ${MKL_ROOT}")
            target_include_directories(nk_bench PRIVATE ${MKL_ROOT}/include)
            # Link against sequential MKL (single-threaded) for fair comparison
            target_link_libraries(nk_bench
                ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a
                ${MKL_ROOT}/lib/intel64/libmkl_sequential.a
                ${MKL_ROOT}/lib/intel64/libmkl_core.a
                pthread m dl)
            target_compile_definitions(nk_bench PRIVATE NK_BUILD_BENCHMARKS_WITH_MKL=1)
        else ()
            message(FATAL_ERROR "MKL not found. Set MKLROOT environment variable or install Intel oneAPI.")
        endif ()
    endif ()

endif ()

if (NK_BUILD_TESTS)
    # Fetch Boost for high-precision reference computations
    include(FetchContent)
    FetchContent_Declare(
        boost_multiprecision
        GIT_REPOSITORY https://github.com/boostorg/multiprecision.git
        GIT_TAG boost-1.89.0
    )
    FetchContent_Declare(
        boost_config
        GIT_REPOSITORY https://github.com/boostorg/config.git
        GIT_TAG boost-1.89.0
    )
    FetchContent_MakeAvailable(boost_multiprecision boost_config)

    # C++ precision test (ULP-based error analysis)
    add_executable(nk_test scripts/test.cpp c/numkong.c)
    target_compile_definitions(nk_test PRIVATE NK_DYNAMIC_DISPATCH=1)
    target_include_directories(nk_test PRIVATE
        ${boost_multiprecision_SOURCE_DIR}/include
        ${boost_config_SOURCE_DIR}/include
    )
    target_link_libraries(nk_test numkong m)

    if (OpenMP_FOUND)
        target_compile_options(nk_test PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(nk_test OpenMP::OpenMP_CXX)
        target_compile_definitions(nk_test PRIVATE NK_TEST_USE_OPENMP=1)
    endif ()

    add_test(NAME nk_test COMMAND nk_test)
    enable_testing()
endif ()

if (NK_BUILD_SHARED)
    set(NK_SOURCES ${NK_SOURCES} c/numkong.c)
    add_library(nk_shared SHARED ${NK_SOURCES})
    target_include_directories(nk_shared PUBLIC "${PROJECT_SOURCE_DIR}/include")
    set_target_properties(nk_shared PROPERTIES OUTPUT_NAME numkong)

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
endif ()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY c/ DESTINATION share/doc/${PROJECT_NAME}/src)
