# cmake/nk_arm_isa_probes.cmake — Arm ISA compiler-capability probes
#
# Detect which ISA extensions the host CPU can actually execute.  Results are
# used to override NK_TARGET_* on the nk_test and nk_bench targets so that
# ISA-specific kernels are only compiled when they can run on the build host.
# The nk_shared library keeps all ISAs enabled (its dispatch is runtime-selected
# via nk_capabilities() function pointers).
#
# NEON variants auto-detect via __ARM_NEON in types.h and need no probes.
# SVE/SME require explicit target attributes because no compiler predefines
# the feature macros without -march/-mcpu flags.  The probes use function-level
# __attribute__((target(...))) — the same mechanism the library kernels use —
# so they work with the default -march.
#
# SME probes use __arm_locally_streaming so the test function enters streaming
# mode internally and can be called from a normal (non-streaming) main().
#
# Native builds: probes use check_source_runs to compile with per-ISA target
# attributes *and* verify the host CPU can execute the resulting binary.
# Cross-compilation: probes use check_source_compiles (can't run host binaries),
# relying on runtime dispatch in the library.
#
# Contract:
#   Input:  NK_BUILD_TEST, NK_BUILD_BENCH (from parent scope)
#   Output: nk_arm_isa_defs_ list (consumed by nk_test/nk_bench targets)

if (NOT (NK_BUILD_TEST OR NK_BUILD_BENCH))
    return()
endif ()

include(CheckSourceRuns)
include(CheckSourceCompiles)

# Save and restore CMAKE_REQUIRED_FLAGS around probes.
set(nk_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
set(nk_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")

# Helper macro: use check_source_runs to verify both compiler support AND host
# CPU capability.  On Arm the function-level target attribute lets the compiler
# emit the instructions without global -march flags, and running the probe
# confirms the host can execute them.
# When cross-compiling, fall back to check_source_compiles (can't run probes).
macro (nk_arm_isa_probe_ var_ flags_ source_)
    set(CMAKE_REQUIRED_FLAGS "${flags_}")
    if (CMAKE_CROSSCOMPILING)
        check_source_compiles(C "${source_}" ${var_})
    else ()
        check_source_runs(C "${source_}" ${var_})
    endif ()
endmacro ()

# SVE — Scalable Vector Extension
nk_arm_isa_probe_(
    nk_target_sve_ "" "
    #include <arm_sve.h>
    __attribute__((target(\"sve\")))
    int test_sve(void) {
        svfloat32_t z = svdup_f32(1.0f);
        return (int)svaddv_f32(svptrue_b32(), z);
    }
    int main(void) { return test_sve() > 0 ? 0 : 1; }
"
)

# SVE F16 — SVE with half-precision support (same gate as base SVE)
nk_arm_isa_probe_(
    nk_target_svehalf_ "" "
    #include <arm_sve.h>
    __attribute__((target(\"sve\")))
    int test_svehalf(void) {
        svfloat16_t z = svdup_f16((__fp16)1.0f);
        return (int)svaddv_f16(svptrue_b16(), z);
    }
    int main(void) { return test_svehalf() > 0 ? 0 : 1; }
"
)

# SVE BF16 — SVE with BFloat16 dot-product (FEAT_BF16)
nk_arm_isa_probe_(
    nk_target_svebfdot_ "" "
    #include <arm_sve.h>
    __attribute__((target(\"sve,bf16\")))
    int test_svebfdot(void) {
        svfloat32_t acc = svdup_f32(0.0f);
        svbfloat16_t a = svdup_bf16(0.0f);
        acc = svbfdot_f32(acc, a, a);
        return (int)svaddv_f32(svptrue_b32(), acc) == 0 ? 0 : 1;
    }
    int main(void) { return test_svebfdot(); }
"
)

# SVE I8 — SVE with signed-dot (FEAT_SVEDot)
nk_arm_isa_probe_(
    nk_target_svesdot_ "" "
    #include <arm_sve.h>
    __attribute__((target(\"sve,dotprod\")))
    int test_svesdot(void) {
        svint32_t acc = svdup_s32(0);
        svint8_t a = svdup_s8(1);
        svint8_t b = svdup_s8(1);
        acc = svdot_s32(acc, a, b);
        return (int)svaddv_s32(svptrue_b32(), acc) >= 0 ? 0 : 1;
    }
    int main(void) { return test_svesdot(); }
"
)

# SVE2
nk_arm_isa_probe_(
    nk_target_sve2_ "" "
    #include <arm_sve.h>
    __attribute__((target(\"sve2\")))
    int test_sve2(void) {
        svint32_t a = svdup_s32(2);
        svint32_t b = svdup_s32(3);
        svint32_t c = svmul_s32_z(svptrue_b32(), a, b);
        return (int)svaddv_s32(svptrue_b32(), c) > 0 ? 0 : 1;
    }
    int main(void) { return test_sve2(); }
"
)

# SVE2P1
nk_arm_isa_probe_(
    nk_target_sve2p1_ "" "
    #include <arm_sve.h>
    __attribute__((target(\"sve2p1\")))
    int test_sve2p1(void) {
        svfloat32_t a = svdup_f32(1.0f);
        return (int)svaddv_f32(svptrue_b32(), a) > 0 ? 0 : 1;
    }
    int main(void) { return test_sve2p1(); }
"
)

# SME — Scalable Matrix Extension
nk_arm_isa_probe_(
    nk_target_sme_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_sme(void) {
        svfloat32_t a = svdup_f32(1.0f);
        svbool_t p = svptrue_b32();
        svmopa_za32_f32_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_sme(); }
"
)

# SME2
nk_arm_isa_probe_(
    nk_target_sme2_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme2\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_sme2(void) {
        svfloat32_t a = svdup_f32(1.0f);
        svbool_t p = svptrue_b32();
        svmopa_za32_f32_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_sme2(); }
"
)

# SME2P1
nk_arm_isa_probe_(
    nk_target_sme2p1_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme2p1\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_sme2p1(void) {
        svfloat32_t a = svdup_f32(1.0f);
        svbool_t p = svptrue_b32();
        svmopa_za32_f32_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_sme2p1(); }
"
)

# SME F64 — FEAT_SME_F64F64
nk_arm_isa_probe_(
    nk_target_smef64_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme,sme-f64f64\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_smef64(void) {
        svfloat64_t a = svdup_f64(1.0);
        svbool_t p = svptrue_b64();
        svmopa_za64_f64_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_smef64(); }
"
)

# SME F16 — FEAT_SME_F16F16
nk_arm_isa_probe_(
    nk_target_smehalf_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_smehalf(void) {
        svfloat16_t a = svdup_f16((__fp16)1.0f);
        svbool_t p = svptrue_b16();
        svmopa_za32_f16_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_smehalf(); }
"
)

# SME BF16 — SME with BFloat16 outer product
nk_arm_isa_probe_(
    nk_target_smebf16_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_smebf16(void) {
        svbfloat16_t a = svdup_bf16(0.0f);
        svbool_t p = svptrue_b16();
        svmopa_za32_bf16_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_smebf16(); }
"
)

# SME BI32 — SME boolean/integer 32-bit outer product
nk_arm_isa_probe_(
    nk_target_smebi32_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme2\")))
    __arm_new(\"za\") __arm_locally_streaming
    int test_smebi32(void) {
        svuint32_t a = svdup_u32(1);
        svbool_t p = svptrue_b32();
        svbmopa_za32_u32_m(0, p, p, a, a);
        return 0;
    }
    int main(void) { return test_smebi32(); }
"
)

# SME LUT2 — FEAT_SME_LUTv2
nk_arm_isa_probe_(
    nk_target_smelut2_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme2\")))
    __arm_new(\"zt0\") __arm_locally_streaming
    int test_smelut2(void) {
        svuint8_t idx = svdup_u8(0);
        svuint8_t r = svluti2_lane_zt_u8(0, idx, 0);
        return (int)svaddv_u8(svptrue_b8(), r) == 0 ? 0 : 1;
    }
    int main(void) { return test_smelut2(); }
"
)

# SME FA64 — FEAT_SME_FA64 (full SVE2 in streaming mode)
# No hardware currently supports this; hardcoded to 0 in types.h.
# Probe kept as a placeholder for future silicon.
nk_arm_isa_probe_(
    nk_target_smefa64_ "" "
    #include <arm_sme.h>
    __attribute__((target(\"sme-fa64\")))
    __arm_locally_streaming
    int test_smefa64(void) {
        svfloat32_t a = svdup_f32(1.0f);
        return (int)svaddv_f32(svptrue_b32(), a) > 0 ? 0 : 1;
    }
    int main(void) { return test_smefa64(); }
"
)

set(CMAKE_REQUIRED_FLAGS "${nk_saved_required_flags_}")
set(CMAKE_TRY_COMPILE_CONFIGURATION "${nk_saved_try_compile_config_}")

# Build the override list from probe results.
# NEON variants are omitted — they auto-detect via __ARM_NEON in types.h.
set(nk_arm_isa_defs_ "")
foreach (
    isa_ IN ITEMS
    SVE SVEHALF SVEBFDOT SVESDOT SVE2 SVE2P1
    SME SME2 SME2P1 SMEF64 SMEHALF SMEBF16 SMEBI32 SMELUT2 SMEFA64
)
    string(TOLOWER "${isa_}" isa_lower_)
    if (nk_target_${isa_lower_}_)
        list(APPEND nk_arm_isa_defs_ "NK_TARGET_${isa_}=1")
    else ()
        list(APPEND nk_arm_isa_defs_ "NK_TARGET_${isa_}=0")
    endif ()
endforeach ()

if (nk_arm_isa_defs_)
    message(STATUS "Arm ISA compiler probes (for nk_test/nk_bench): ${nk_arm_isa_defs_}")
endif ()
