/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE BF16.
 *  @file include/numkong/spatial/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_svebfdot_instructions ARM SVE+BF16 Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_bf16                  LD1H (Z.H, P/Z, [Xn])           4-6cy       2/cy
 *      svld1_u16                   LD1H (Z.H, P/Z, [Xn])           4-6cy       2/cy
 *      svbfdot_f32                 BFDOT (Z.S, Z.H, Z.H)           4cy         2/cy
 *      svmla_f32_x                 FMLA (Z.S, P/M, Z.S, Z.S)       4cy         2/cy
 *      svsub_f32_x                 FSUB (Z.S, P/M, Z.S, Z.S)       3cy         2/cy
 *      svaddv_f32                  FADDV (S, P, Z.S)               6cy         1/cy
 *      svunpklo_u32                UUNPKLO (Z.S, Z.H)              2cy         2/cy
 *      svunpkhi_u32                UUNPKHI (Z.S, Z.H)              2cy         2/cy
 *      svlsl_n_u32_x               LSL (Z.S, P/M, Z.S, #imm)       2cy         2/cy
 *      svwhilelt_b16               WHILELT (P.H, Xn, Xm)           2cy         1/cy
 *      svwhilelt_b32               WHILELT (P.S, Xn, Xm)           2cy         1/cy
 *      svcnth                      CNTH (Xd)                       1cy         2/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  The BFDOT instruction fuses two BF16 multiplications with FP32 accumulation, providing
 *  efficient BF16 dot products without explicit conversion overhead.
 */
#ifndef NK_SPATIAL_SVEBFDOT_H
#define NK_SPATIAL_SVEBFDOT_H

#if defined(__cplusplus)
extern "C" {
#endif

#if NK_TARGET_ARM_
#if NK_TARGET_SVEBFDOT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+bf16")
#endif

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

NK_PUBLIC void nk_sqeuclidean_bf16_svebfdot(nk_bf16_t const *a_enum, nk_bf16_t const *b_enum, nk_size_t n,
                                            nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t d2_low_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t d2_high_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    nk_u16_t const *a = (nk_u16_t const *)(a_enum);
    nk_u16_t const *b = (nk_u16_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svuint16_t a_vec = svld1_u16(pg_vec, a + i);
        svuint16_t b_vec = svld1_u16(pg_vec, b + i);

        // There is no `bf16` subtraction in SVE, so we need to convert to `u32` and shift.
        svbool_t pg_low_vec = svwhilelt_b32((unsigned int)(i), (unsigned int)n);
        svbool_t pg_high_vec = svwhilelt_b32((unsigned int)(i + svcnth() / 2), (unsigned int)n);
        svfloat32_t a_low_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_low_vec, svunpklo_u32(a_vec), 16));
        svfloat32_t a_high_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_high_vec, svunpkhi_u32(a_vec), 16));
        svfloat32_t b_low_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_low_vec, svunpklo_u32(b_vec), 16));
        svfloat32_t b_high_vec = svreinterpret_f32_u32(svlsl_n_u32_x(pg_high_vec, svunpkhi_u32(b_vec), 16));

        svfloat32_t a_minus_b_low_vec = svsub_f32_x(pg_low_vec, a_low_vec, b_low_vec);
        svfloat32_t a_minus_b_high_vec = svsub_f32_x(pg_high_vec, a_high_vec, b_high_vec);
        d2_low_vec = svmla_f32_x(pg_vec, d2_low_vec, a_minus_b_low_vec, a_minus_b_low_vec);
        d2_high_vec = svmla_f32_x(pg_vec, d2_high_vec, a_minus_b_high_vec, a_minus_b_high_vec);
        i += svcnth();
    } while (i < n);
    nk_f32_t d2 = svaddv_f32(svptrue_b32(), d2_low_vec) + svaddv_f32(svptrue_b32(), d2_high_vec);
    *result = d2;
}
NK_PUBLIC void nk_euclidean_bf16_svebfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_bf16_svebfdot(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_bf16_svebfdot(nk_bf16_t const *a_enum, nk_bf16_t const *b_enum, nk_size_t n,
                                        nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    nk_bf16_for_arm_simd_t const *a = (nk_bf16_for_arm_simd_t const *)(a_enum);
    nk_bf16_for_arm_simd_t const *b = (nk_bf16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svbfloat16_t a_vec = svld1_bf16(pg_vec, a + i);
        svbfloat16_t b_vec = svld1_bf16(pg_vec, b + i);
        ab_vec = svbfdot_f32(ab_vec, a_vec, b_vec);
        a2_vec = svbfdot_f32(a2_vec, a_vec, a_vec);
        b2_vec = svbfdot_f32(b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    nk_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    nk_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    nk_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVEBFDOT
#endif // NK_TARGET_ARM_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SPATIAL_SVEBFDOT_H
