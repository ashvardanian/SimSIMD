# cmake/nk_x86_isa_probes.cmake — x86 ISA compiler-capability probes
#
# Detect which ISA extensions the host CPU can actually execute.  Results are
# used to override NK_TARGET_* on the nk_test and nk_bench targets so that
# ISA-specific kernels are only compiled when they can run on the build host.
# The nk_shared library keeps all ISAs enabled (its dispatch is runtime-selected
# via nk_capabilities() function pointers).
#
# Native builds (all compilers): probes use check_source_runs to compile with
# per-ISA flags *and* verify the host CPU can execute the resulting binary.
# Cross-compilation (GCC/Clang): probes use check_source_compiles (can't run
# host binaries), relying on runtime dispatch in the library.
#
# Contract:
#   Input:  NK_BUILD_TEST, NK_BUILD_BENCH (from parent scope)
#   Output: nk_x86_isa_defs_ list (consumed by nk_test/nk_bench targets)

if (NOT (NK_BUILD_TEST OR NK_BUILD_BENCH))
    return()
endif ()

include(CheckSourceRuns)
include(CheckSourceCompiles)

# Save and restore CMAKE_REQUIRED_FLAGS around probes.
# Force Release config for try_run to avoid ASAN DLL-not-found failures (0xc0000135).
set(nk_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
set(nk_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")

# Helper macro: use check_source_runs to verify both compiler support AND host
# CPU capability.  On MSVC the compiler always succeeds so runtime detection is
# essential; on GCC/Clang the per-ISA -m flags let the compiler emit the
# instructions, and running the probe confirms the host can execute them.
# When cross-compiling, fall back to check_source_compiles (can't run probes).
macro(nk_isa_probe_ var_ msvc_arch_ gcc_flags_ source_)
    if (CMAKE_CROSSCOMPILING)
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
        check_source_compiles(C "${source_}" ${var_})
    elseif (MSVC)
        set(CMAKE_REQUIRED_FLAGS "${msvc_arch_}")
        check_source_runs(C "${source_}" ${var_})
    else ()
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
        check_source_runs(C "${source_}" ${var_})
    endif ()
endmacro()

# Haswell — AVX2
nk_isa_probe_(nk_target_haswell_ "/arch:AVX2" "-mavx2 -mfma -mf16c" "
    #include <immintrin.h>
    int main(void) {
        volatile int one = 1;
        __m256i a = _mm256_set1_epi32(one);
        __m256i b = _mm256_add_epi32(a, a);
        return _mm256_extract_epi32(b, 0) == 2 ? 0 : 1;
    }
")

# Skylake — AVX-512F
nk_isa_probe_(nk_target_skylake_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl" "
    #include <immintrin.h>
    int main(void) {
        volatile int one = 1;
        __m512i a = _mm512_set1_epi32(one);
        __m512i b = _mm512_add_epi32(a, a);
        return (int)_mm512_reduce_add_epi32(b) == 32 ? 0 : 1;
    }
")

# Ice Lake — AVX-512VNNI
nk_isa_probe_(nk_target_icelake_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni" "
    #include <immintrin.h>
    int main(void) {
        volatile int one = 1;
        __m512i acc = _mm512_setzero_si512();
        __m512i a = _mm512_set1_epi8((char)one);
        __m512i b = _mm512_set1_epi8((char)one);
        acc = _mm512_dpbusd_epi32(acc, a, b);
        return (int)_mm512_reduce_add_epi32(acc) == 64 ? 0 : 1;
    }
")

# Genoa — AVX-512BF16
nk_isa_probe_(nk_target_genoa_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16" "
    #include <immintrin.h>
    int main(void) {
        volatile float one = 1.0f;
        __m512 f = _mm512_set1_ps(one);
        __m256bh a = _mm512_cvtneps_pbh(f);
        __m512bh wide = (__m512bh)_mm512_castsi512_ps(
            _mm512_inserti64x4(_mm512_setzero_si512(), (__m256i)a, 0));
        __m512 r = _mm512_dpbf16_ps(_mm512_setzero_ps(), wide, wide);
        return _mm512_reduce_add_ps(r) >= 0.0f ? 0 : 1;
    }
")

# Sapphire Rapids — AVX-512FP16
nk_isa_probe_(nk_target_sapphire_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512fp16" "
    #include <immintrin.h>
    int main(void) {
        volatile float one = 1.0f;
        __m512h a = _mm512_set1_ph((_Float16)one);
        __m512h b = _mm512_set1_ph((_Float16)(one + one));
        __m512h c = _mm512_fmadd_ph(a, b, a);
        return (int)_mm_extract_epi16(_mm256_castsi256_si128(_mm512_castsi512_si256((__m512i)c)), 0) != 0 ? 0 : 1;
    }
")

# Sapphire Rapids AMX — AMX-TILE + INT8
nk_isa_probe_(nk_target_sapphireamx_ "/arch:AVX512" "-mamx-tile -mamx-int8" "
    #include <immintrin.h>
    int main(void) {
        volatile int zero = 0;
        _tile_release();
        return zero;
    }
")

# Granite Rapids AMX — AMX-FP16
nk_isa_probe_(nk_target_graniteamx_ "/arch:AVX512" "-mamx-tile -mamx-fp16" "
    #include <immintrin.h>
    #include <amxfp16intrin.h>
    int main(void) {
        volatile int zero = 0;
        _tile_release();
        return zero;
    }
")

# Turin — AVX-512VP2INTERSECT
nk_isa_probe_(nk_target_turin_ "/arch:AVX512" "-mavx512f -mavx512vp2intersect" "
    #include <immintrin.h>
    int main(void) {
        volatile int val = 42;
        __m512i a = _mm512_set1_epi32(val);
        __m512i b = _mm512_set1_epi32(val);
        __mmask16 k0, k1;
        _mm512_2intersect_epi32(a, b, &k0, &k1);
        return k0 != 0 ? 0 : 1;
    }
")

# Alder Lake — AVX-VNNI (DPBUSD 256-bit, VEX-encoded)
nk_isa_probe_(nk_target_alder_ "/arch:AVX2" "-mavx2 -mavxvnni" "
    #include <immintrin.h>
    int main(void) {
        volatile int two = 2;
        __m256i acc = _mm256_setzero_si256();
        __m256i a = _mm256_set1_epi8((char)two);
        __m256i b = _mm256_set1_epi8((char)(two + 1));
        acc = _mm256_dpbusd_avx_epi32(acc, a, b);
        return _mm256_extract_epi32(acc, 0) == 24 ? 0 : 1;
    }
")

# Sierra Forest — AVXVNNIINT8 (DPBSSD 256-bit)
nk_isa_probe_(nk_target_sierra_ "/arch:AVX2" "-mavx2 -mavxvnniint8" "
    #include <immintrin.h>
    int main(void) {
        volatile int two = 2;
        __m256i acc = _mm256_setzero_si256();
        __m256i a = _mm256_set1_epi8((char)two);
        __m256i b = _mm256_set1_epi8((char)(two + 1));
        acc = _mm256_dpbssd_epi32(acc, a, b);
        return _mm256_extract_epi32(acc, 0) == 24 ? 0 : 1;
    }
")

set(CMAKE_REQUIRED_FLAGS "${nk_saved_required_flags_}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "${nk_saved_try_compile_config_}")

# Build the override list from probe results
set(nk_x86_isa_defs_ "")
foreach (isa_ IN ITEMS HASWELL SKYLAKE ICELAKE GENOA SAPPHIRE SAPPHIREAMX GRANITEAMX TURIN ALDER SIERRA)
    string(TOLOWER "${isa_}" isa_lower_)
    if (nk_target_${isa_lower_}_)
        list(APPEND nk_x86_isa_defs_ "NK_TARGET_${isa_}=1")
    else ()
        list(APPEND nk_x86_isa_defs_ "NK_TARGET_${isa_}=0")
    endif ()
endforeach ()

if (nk_x86_isa_defs_)
    message(STATUS "ISA compiler probes (for nk_test/nk_bench): ${nk_x86_isa_defs_}")
endif ()
