# cmake/nk_x86_isa_probes.cmake — x86 ISA runtime probes
#
# Compile+execute ISA-specific instructions at configure time to detect what the
# *host* CPU actually supports.  Results are used to override NK_TARGET_* on the
# nk_test and nk_bench targets.  The nk_shared library keeps all ISAs enabled
# (its dispatch is runtime-selected via nk_capabilities() function pointers).
#
# On MSVC, `types.h` auto-enables all ISAs based on _MSC_VER alone (since MSVC
# provides no fine-grained ISA preprocessor macros like __AVXVNNI__, __AVX512VNNI__,
# etc.).  The generic dispatch `#elif` chains always pick the highest ISA → SIGILL
# on CPUs that don't support it.  These probes detect what the host CPU actually
# supports and override NK_TARGET_* accordingly.
#
# On GCC/Clang the probes naturally fail (different intrinsic signatures / missing
# headers) so the override list stays empty and types.h auto-detection is used,
# which works correctly.
#
# Contract:
#   Input:  NK_BUILD_TEST, NK_BUILD_BENCH (from parent scope)
#   Output: nk_x86_isa_defs_ list (consumed by nk_test/nk_bench targets)

if (NOT (NK_BUILD_TEST OR NK_BUILD_BENCH))
    return()
endif ()

include(CheckSourceRuns)

# MSVC needs /arch flags to compile SIMD intrinsics; GCC/Clang use -march=native from the
# project flags.  Save and restore CMAKE_REQUIRED_FLAGS around each group of probes.
# Also force Release config for try_run to avoid ASAN DLL-not-found failures (0xc0000135).
set(nk_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
set(nk_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")

if (MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
endif ()

# Haswell — AVX2
check_source_runs(
    C "
    #include <immintrin.h>
    int main(void) {
        __m256i a = _mm256_set1_epi32(1);
        __m256i b = _mm256_add_epi32(a, a);
        return _mm256_extract_epi32(b, 0) == 2 ? 0 : 1;
    }
" nk_target_haswell_
)

if (MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
endif ()

# Skylake — AVX-512F
check_source_runs(
    C "
    #include <immintrin.h>
    int main(void) {
        __m512i a = _mm512_set1_epi32(1);
        __m512i b = _mm512_add_epi32(a, a);
        return (int)_mm512_reduce_add_epi32(b) == 32 ? 0 : 1;
    }
" nk_target_skylake_
)

# Ice Lake — AVX-512VNNI
check_source_runs(
    C
    "
    #include <immintrin.h>
    int main(void) {
        __m512i acc = _mm512_setzero_si512();
        __m512i a = _mm512_set1_epi8(1);
        __m512i b = _mm512_set1_epi8(1);
        acc = _mm512_dpbusd_epi32(acc, a, b);
        return (int)_mm512_reduce_add_epi32(acc) == 64 ? 0 : 1;
    }
"
    nk_target_icelake_
)

# Genoa — AVX-512BF16
check_source_runs(
    C
    "
    #include <immintrin.h>
    int main(void) {
        __m512 f = _mm512_set1_ps(1.0f);
        __m256bh a = _mm512_cvtneps_pbh(f);
        __m512bh wide = (__m512bh)_mm512_castsi512_ps(
            _mm512_inserti64x4(_mm512_setzero_si512(), (__m256i)a, 0));
        __m512 r = _mm512_dpbf16_ps(_mm512_setzero_ps(), wide, wide);
        (void)r;
        return 0;
    }
"
    nk_target_genoa_
)

# Sapphire Rapids — AVX-512FP16
check_source_runs(
    C
    "
    #include <immintrin.h>
    int main(void) {
        __m512h a = _mm512_set1_ph(1.0f);
        __m512h b = _mm512_set1_ph(2.0f);
        __m512h c = _mm512_fmadd_ph(a, b, a);
        (void)c;
        return 0;
    }
"
    nk_target_sapphire_
)

# Sapphire Rapids AMX — AMX-TILE + INT8
check_source_runs(
    C "
    #include <immintrin.h>
    int main(void) {
        _tile_release();
        return 0;
    }
" nk_target_sapphireamx_
)

# Granite Rapids AMX — AMX-FP16
check_source_runs(
    C "
    #include <immintrin.h>
    #include <amxfp16intrin.h>
    int main(void) {
        _tile_release();
        return 0;
    }
" nk_target_graniteamx_
)

# Turin — AVX-512VP2INTERSECT
check_source_runs(
    C
    "
    #include <immintrin.h>
    int main(void) {
        __m512i a = _mm512_set1_epi32(42);
        __m512i b = _mm512_set1_epi32(42);
        __mmask16 k0, k1;
        _mm512_2intersect_epi32(a, b, &k0, &k1);
        return k0 != 0 ? 0 : 1;
    }
"
    nk_target_turin_
)

if (MSVC)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
endif ()

# Alder Lake — AVX-VNNI (DPBUSD 256-bit, VEX-encoded)
check_source_runs(
    C
    "
    #include <immintrin.h>
    int main(void) {
        __m256i acc = _mm256_setzero_si256();
        __m256i a = _mm256_set1_epi8(2);
        __m256i b = _mm256_set1_epi8(3);
        acc = _mm256_dpbusd_avx_epi32(acc, a, b);
        return _mm256_extract_epi32(acc, 0) == 24 ? 0 : 1;
    }
"
    nk_target_alder_
)

# Sierra Forest — AVXVNNIINT8 (DPBSSD 256-bit)
check_source_runs(
    C
    "
    #include <immintrin.h>
    int main(void) {
        __m256i acc = _mm256_setzero_si256();
        __m256i a = _mm256_set1_epi8(2);
        __m256i b = _mm256_set1_epi8(3);
        acc = _mm256_dpbssd_epi32(acc, a, b);
        return _mm256_extract_epi32(acc, 0) == 24 ? 0 : 1;
    }
"
    nk_target_sierra_
)

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
    message(STATUS "ISA runtime probes (for nk_test/nk_bench): ${nk_x86_isa_defs_}")
endif ()
