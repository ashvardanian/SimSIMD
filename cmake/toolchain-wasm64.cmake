# WASM64/Emscripten Memory64 toolchain for NumKong
# Usage: cmake -B build-wasm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm64.cmake

set(CMAKE_SYSTEM_NAME Emscripten)
set(CMAKE_SYSTEM_PROCESSOR wasm64)

# Verify Emscripten SDK
if(NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR
        "EMSDK environment variable not set.\n"
        "Install Emscripten: https://emscripten.org/docs/getting_started/downloads.html\n"
        "Then run: source $EMSDK/emsdk_env.sh")
endif()

# Set compilers
set(EMSCRIPTEN_ROOT "$ENV{EMSDK}/upstream/emscripten")
set(CMAKE_C_COMPILER "${EMSCRIPTEN_ROOT}/emcc")
set(CMAKE_CXX_COMPILER "${EMSCRIPTEN_ROOT}/em++")
set(CMAKE_AR "${EMSCRIPTEN_ROOT}/emar")
set(CMAKE_RANLIB "${EMSCRIPTEN_ROOT}/emranlib")

# Required WASM SIMD flags + Memory64
set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS} -sMEMORY64=1")
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS} -sMEMORY64=1")

# Enable GNU extensions for EM_ASM support (required for runtime detection)
set(CMAKE_C_EXTENSIONS ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_EXTENSIONS ON CACHE BOOL "" FORCE)

# Optimization
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -s ASSERTIONS=2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -s ASSERTIONS=2")

# Linker flags for Node.js execution (larger memory limits for 64-bit)
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-s MEMORY64=1 \
     -s ALLOW_MEMORY_GROWTH=1 \
     -s IMPORTED_MEMORY=1 \
     -s INITIAL_MEMORY=256MB \
     -s MAXIMUM_MEMORY=16GB \
     -s STACK_SIZE=5MB \
     -s EXPORTED_FUNCTIONS='[\"_main\"]' \
     -s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\"]'")

# Verify Emscripten version (need 3.1.35+ for memory64)
execute_process(
    COMMAND ${CMAKE_C_COMPILER} --version
    OUTPUT_VARIABLE EMCC_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" EMCC_VERSION "${EMCC_VERSION_OUTPUT}")

if(EMCC_VERSION VERSION_LESS "3.1.35")
    message(WARNING "Emscripten ${EMCC_VERSION} < 3.1.35. Upgrade recommended for Memory64 support.")
endif()

message(STATUS "NumKong WASM64: Emscripten ${EMCC_VERSION}")
message(STATUS "NumKong WASM64: Relaxed SIMD + Memory64 enabled")
