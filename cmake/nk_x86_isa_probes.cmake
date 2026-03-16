# cmake/nk_x86_isa_probes.cmake — x86 ISA compiler-capability probes
#
# Detect which ISA extensions the *compiler* can target, regardless of the host
# CPU.  Results are used to override NK_TARGET_* on the nk_test and nk_bench
# targets.  The nk_shared library keeps all ISAs enabled (its dispatch is
# runtime-selected via nk_capabilities() function pointers).
#
# On MSVC, `types.h` auto-enables all ISAs based on _MSC_VER alone (since MSVC
# provides no fine-grained ISA preprocessor macros like __AVXVNNI__, __AVX512VNNI__,
# etc.).  The generic dispatch `#elif` chains always pick the highest ISA → SIGILL
# on CPUs that don't support it.  These probes use check_source_runs to detect
# what the host CPU actually supports.
#
# On GCC/Clang each probe uses check_source_compiles with per-ISA -m flags so
# that even an older host can build test/bench binaries targeting newer ISAs.
# Runtime dispatch in the library ensures only supported ISAs are exercised.
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

# Helper macro: on MSVC use check_source_runs (compiler always succeeds, need
# runtime host detection); on GCC/Clang use check_source_compiles with per-ISA
# -m flags (tests compiler capability, not host capability).
macro(nk_isa_probe_ var_ msvc_arch_ gcc_flags_ source_)
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "${msvc_arch_}")
        check_source_runs(C "${source_}" ${var_})
    else ()
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
        check_source_compiles(C "${source_}" ${var_})
    endif ()
endmacro()

# Haswell — AVX2
nk_isa_probe_(nk_target_haswell_ "/arch:AVX2" "-mavx2 -mfma -mf16c" "
    #include <immintrin.h>
    int main(void) {
        __m256i a = _mm256_set1_epi32(1);
        __m256i b = _mm256_add_epi32(a, a);
        return _mm256_extract_epi32(b, 0) == 2 ? 0 : 1;
    }
")

# Skylake — AVX-512F
nk_isa_probe_(nk_target_skylake_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl" "
    #include <immintrin.h>
    int main(void) {
        __m512i a = _mm512_set1_epi32(1);
        __m512i b = _mm512_add_epi32(a, a);
        return (int)_mm512_reduce_add_epi32(b) == 32 ? 0 : 1;
    }
")

# Ice Lake — AVX-512VNNI
nk_isa_probe_(nk_target_icelake_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni" "
    #include <immintrin.h>
    int main(void) {
        __m512i acc = _mm512_setzero_si512();
        __m512i a = _mm512_set1_epi8(1);
        __m512i b = _mm512_set1_epi8(1);
        acc = _mm512_dpbusd_epi32(acc, a, b);
        return (int)_mm512_reduce_add_epi32(acc) == 64 ? 0 : 1;
    }
")

# Genoa — AVX-512BF16
nk_isa_probe_(nk_target_genoa_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16" "
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
")

# Sapphire Rapids — AVX-512FP16
nk_isa_probe_(nk_target_sapphire_ "/arch:AVX512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512fp16" "
    #include <immintrin.h>
    int main(void) {
        __m512h a = _mm512_set1_ph(1.0f);
        __m512h b = _mm512_set1_ph(2.0f);
        __m512h c = _mm512_fmadd_ph(a, b, a);
        (void)c;
        return 0;
    }
")

# Sapphire Rapids AMX — AMX-TILE + INT8
nk_isa_probe_(nk_target_sapphireamx_ "/arch:AVX512" "-mamx-tile -mamx-int8" "
    #include <immintrin.h>
    int main(void) {
        _tile_release();
        return 0;
    }
")

# Granite Rapids AMX — AMX-FP16
nk_isa_probe_(nk_target_graniteamx_ "/arch:AVX512" "-mamx-tile -mamx-fp16" "
    #include <immintrin.h>
    #include <amxfp16intrin.h>
    int main(void) {
        _tile_release();
        return 0;
    }
")

# Turin — AVX-512VP2INTERSECT
nk_isa_probe_(nk_target_turin_ "/arch:AVX512" "-mavx512f -mavx512vp2intersect" "
    #include <immintrin.h>
    int main(void) {
        __m512i a = _mm512_set1_epi32(42);
        __m512i b = _mm512_set1_epi32(42);
        __mmask16 k0, k1;
        _mm512_2intersect_epi32(a, b, &k0, &k1);
        return k0 != 0 ? 0 : 1;
    }
")

# Alder Lake — AVX-VNNI (DPBUSD 256-bit, VEX-encoded)
nk_isa_probe_(nk_target_alder_ "/arch:AVX2" "-mavx2 -mavxvnni" "
    #include <immintrin.h>
    int main(void) {
        __m256i acc = _mm256_setzero_si256();
        __m256i a = _mm256_set1_epi8(2);
        __m256i b = _mm256_set1_epi8(3);
        acc = _mm256_dpbusd_avx_epi32(acc, a, b);
        return _mm256_extract_epi32(acc, 0) == 24 ? 0 : 1;
    }
")

# Sierra Forest — AVXVNNIINT8 (DPBSSD 256-bit)
nk_isa_probe_(nk_target_sierra_ "/arch:AVX2" "-mavx2 -mavxvnniint8" "
    #include <immintrin.h>
    int main(void) {
        __m256i acc = _mm256_setzero_si256();
        __m256i a = _mm256_set1_epi8(2);
        __m256i b = _mm256_set1_epi8(3);
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
