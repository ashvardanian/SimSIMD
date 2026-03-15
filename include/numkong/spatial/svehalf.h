/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE FP16.
 *  @file include/numkong/spatial/sve.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_svehalf_instructions ARM SVE+FP16 Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      svld1_f16                   LD1H (Z.H, P/Z, [Xn])           4-6cy       2/cy
 *      svsub_f16_x                 FSUB (Z.H, P/M, Z.H, Z.H)       3cy         2/cy
 *      svmla_f16_x                 FMLA (Z.H, P/M, Z.H, Z.H)       4cy         2/cy
 *      svaddv_f16                  FADDV (H, P, Z.H)               6cy         1/cy
 *      svdupq_n_f16                DUP (Z.H, #imm)                 1cy         2/cy
 *      svwhilelt_b16               WHILELT (P.H, Xn, Xm)           2cy         1/cy
 *      svptrue_b16                 PTRUE (P.H, pattern)            1cy         2/cy
 *      svcnth                      CNTH (Xd)                       1cy         2/cy
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  FP16 spatial operations trade precision for throughput, processing twice as many elements
 *  per cycle. This is particularly effective for embedding similarity in ML applications.
 */
#ifndef NK_SPATIAL_SVEHALF_H
#define NK_SPATIAL_SVEHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVEHALF

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#endif

NK_PUBLIC void nk_sqeuclidean_f16_svehalf(nk_f16_t const *a_enum, nk_f16_t const *b_enum, nk_size_t n,
                                          nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t d2_f32x = svdup_n_f32(0.0f);
    nk_f16_for_arm_simd_t const *a = (nk_f16_for_arm_simd_t const *)(a_enum);
    nk_f16_for_arm_simd_t const *b = (nk_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t predicate_f32x = svwhilelt_b32_u64(i, n);
        svfloat16_t a_f16x = svld1_f16(svwhilelt_b16_u64(i, n), a + i);
        svfloat16_t b_f16x = svld1_f16(svwhilelt_b16_u64(i, n), b + i);
        svfloat32_t a_f32x = svcvt_f32_f16_x(predicate_f32x, a_f16x);
        svfloat32_t b_f32x = svcvt_f32_f16_x(predicate_f32x, b_f16x);
        svfloat32_t diff_f32x = svsub_f32_x(predicate_f32x, a_f32x, b_f32x);
        d2_f32x = svmla_f32_x(predicate_f32x, d2_f32x, diff_f32x, diff_f32x);
        i += svcntw();
    } while (i < n);
    *result = svaddv_f32(svptrue_b32(), d2_f32x);
}

NK_PUBLIC void nk_euclidean_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_svehalf(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f16_svehalf(nk_f16_t const *a_enum, nk_f16_t const *b_enum, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t ab_f32x = svdup_n_f32(0.0f);
    svfloat32_t a2_f32x = svdup_n_f32(0.0f);
    svfloat32_t b2_f32x = svdup_n_f32(0.0f);
    nk_f16_for_arm_simd_t const *a = (nk_f16_for_arm_simd_t const *)(a_enum);
    nk_f16_for_arm_simd_t const *b = (nk_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t predicate_f32x = svwhilelt_b32_u64(i, n);
        svfloat16_t a_f16x = svld1_f16(svwhilelt_b16_u64(i, n), a + i);
        svfloat16_t b_f16x = svld1_f16(svwhilelt_b16_u64(i, n), b + i);
        svfloat32_t a_f32x = svcvt_f32_f16_x(predicate_f32x, a_f16x);
        svfloat32_t b_f32x = svcvt_f32_f16_x(predicate_f32x, b_f16x);
        ab_f32x = svmla_f32_x(predicate_f32x, ab_f32x, a_f32x, b_f32x);
        a2_f32x = svmla_f32_x(predicate_f32x, a2_f32x, a_f32x, a_f32x);
        b2_f32x = svmla_f32_x(predicate_f32x, b2_f32x, b_f32x, b_f32x);
        i += svcntw();
    } while (i < n);

    nk_f32_t ab_f32 = svaddv_f32(svptrue_b32(), ab_f32x);
    nk_f32_t a2_f32 = svaddv_f32(svptrue_b32(), a2_f32x);
    nk_f32_t b2_f32 = svaddv_f32(svptrue_b32(), b2_f32x);
    *result = nk_angular_normalize_f32_neon_(ab_f32, a2_f32, b2_f32);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVEHALF
#endif // NK_TARGET_ARM_
#endif // NK_SPATIAL_SVEHALF_H
