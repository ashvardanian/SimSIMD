# WASM/Emscripten toolchain for NumKong.
# Usage: cmake -B build-wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm.cmake

set(CMAKE_SYSTEM_NAME Emscripten)
set(CMAKE_SYSTEM_PROCESSOR wasm32)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Verify the Emscripten SDK.
if (NOT DEFINED ENV{EMSDK})
    message(
        FATAL_ERROR
            "EMSDK environment variable not set.\n"
            "Install Emscripten: https://emscripten.org/docs/getting_started/downloads.html\n"
            "Then run: source $EMSDK/emsdk_env.sh"
    )
endif ()

# Set compilers.
set(EMSCRIPTEN_ROOT "$ENV{EMSDK}/upstream/emscripten")
file(TO_CMAKE_PATH "${EMSCRIPTEN_ROOT}" EMSCRIPTEN_ROOT)

if (CMAKE_HOST_WIN32)
    set(EMSCRIPTEN_TOOL_SUFFIX ".bat")
else ()
    set(EMSCRIPTEN_TOOL_SUFFIX "")
endif ()

set(CMAKE_C_COMPILER "${EMSCRIPTEN_ROOT}/emcc${EMSCRIPTEN_TOOL_SUFFIX}")
set(CMAKE_CXX_COMPILER "${EMSCRIPTEN_ROOT}/em++${EMSCRIPTEN_TOOL_SUFFIX}")
set(CMAKE_AR "${EMSCRIPTEN_ROOT}/emar${EMSCRIPTEN_TOOL_SUFFIX}")
set(CMAKE_RANLIB "${EMSCRIPTEN_ROOT}/emranlib${EMSCRIPTEN_TOOL_SUFFIX}")

# Required WASM SIMD flags.
set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS}")

# Enable GNU extensions for EM_ASM support (required for runtime detection).
set(CMAKE_C_EXTENSIONS ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_EXTENSIONS ON CACHE BOOL "" FORCE)

# Optimization.
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -s ASSERTIONS=2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -s ASSERTIONS=2")

# Linker flags for Node.js execution.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-sALLOW_MEMORY_GROWTH=1 \
     -sIMPORTED_MEMORY=1 \
     -sINITIAL_MEMORY=64MB \
     -sMAXIMUM_MEMORY=2GB \
     -sSTACK_SIZE=5MB \
     -sEXPORTED_FUNCTIONS=['_main'] \
     -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
)

# Verify the Emscripten version (3.1.27+ recommended for relaxed SIMD).
execute_process(
    COMMAND ${CMAKE_C_COMPILER} --version OUTPUT_VARIABLE EMCC_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" EMCC_VERSION "${EMCC_VERSION_OUTPUT}")

if (EMCC_VERSION VERSION_LESS "3.1.27")
    message(WARNING "Emscripten ${EMCC_VERSION} < 3.1.27. Upgrade recommended for relaxed SIMD.")
endif ()

message(STATUS "NumKong WASM: Emscripten ${EMCC_VERSION}")
message(STATUS "NumKong WASM: Relaxed SIMD enabled")
