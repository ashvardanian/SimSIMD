/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/spatial/sve.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SVEHALF_H
#define NK_SPATIAL_SVEHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVEHALF
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_f16_svehalf(nk_f16_t const *a_enum, nk_f16_t const *b_enum, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat16_t d2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    nk_f16_for_arm_simd_t const *a = (nk_f16_for_arm_simd_t const *)(a_enum);
    nk_f16_for_arm_simd_t const *b = (nk_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        svfloat16_t a_minus_b_vec = svsub_f16_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f16_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcnth();
    } while (i < n);
    nk_f16_for_arm_simd_t d2_f16 = svaddv_f16(svptrue_b16(), d2_vec);
    *result = d2_f16;
}

NK_PUBLIC void nk_l2_f16_svehalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f16_svehalf(a, b, n, result);
    *result = nk_sqrt_f32_neon_(*result);
}

NK_PUBLIC void nk_angular_f16_svehalf(nk_f16_t const *a_enum, nk_f16_t const *b_enum, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat16_t ab_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t a2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    svfloat16_t b2_vec = svdupq_n_f16(0, 0, 0, 0, 0, 0, 0, 0);
    nk_f16_for_arm_simd_t const *a = (nk_f16_for_arm_simd_t const *)(a_enum);
    nk_f16_for_arm_simd_t const *b = (nk_f16_for_arm_simd_t const *)(b_enum);
    do {
        svbool_t pg_vec = svwhilelt_b16((unsigned int)i, (unsigned int)n);
        svfloat16_t a_vec = svld1_f16(pg_vec, a + i);
        svfloat16_t b_vec = svld1_f16(pg_vec, b + i);
        ab_vec = svmla_f16_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f16_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f16_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcnth();
    } while (i < n);

    nk_f16_for_arm_simd_t ab = svaddv_f16(svptrue_b16(), ab_vec);
    nk_f16_for_arm_simd_t a2 = svaddv_f16(svptrue_b16(), a2_vec);
    nk_f16_for_arm_simd_t b2 = svaddv_f16(svptrue_b16(), b2_vec);
    *result = nk_angular_normalize_f32_neon_(ab, a2, b2);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SVEHALF
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_SVEHALF_H