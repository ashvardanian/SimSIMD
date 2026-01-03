/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Arm SVE-capable CPUs.
 *  @file include/numkong/spatial/sve.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SVE_H
#define NK_SPATIAL_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t d2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        svfloat32_t a_minus_b_vec = svsub_f32_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f32_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntw();
    } while (i < n);
    nk_f32_t d2 = svaddv_f32(svptrue_b32(), d2_vec);
    *result = d2;
}

NK_PUBLIC void nk_l2_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f32_sve(a, b, n, result);
    *result = nk_f32_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f32_sve(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svfloat32_t ab_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t a2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    svfloat32_t b2_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);
    do {
        svbool_t pg_vec = svwhilelt_b32((unsigned int)i, (unsigned int)n);
        svfloat32_t a_vec = svld1_f32(pg_vec, a + i);
        svfloat32_t b_vec = svld1_f32(pg_vec, b + i);
        ab_vec = svmla_f32_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f32_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f32_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntw();
    } while (i < n);

    nk_f32_t ab = svaddv_f32(svptrue_b32(), ab_vec);
    nk_f32_t a2 = svaddv_f32(svptrue_b32(), a2_vec);
    nk_f32_t b2 = svaddv_f32(svptrue_b32(), b2_vec);
    *result = (nk_f32_t)nk_angular_normalize_f64_neon_(ab, a2, b2);
}

NK_PUBLIC void nk_l2sq_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t i = 0;
    svfloat64_t d2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        svfloat64_t a_minus_b_vec = svsub_f64_x(pg_vec, a_vec, b_vec);
        d2_vec = svmla_f64_x(pg_vec, d2_vec, a_minus_b_vec, a_minus_b_vec);
        i += svcntd();
    } while (i < n);
    nk_f64_t d2 = svaddv_f64(svptrue_b32(), d2_vec);
    *result = d2;
}

NK_PUBLIC void nk_l2_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_l2sq_f64_sve(a, b, n, result);
    *result = nk_f64_sqrt_neon(*result);
}

NK_PUBLIC void nk_angular_f64_sve(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_size_t i = 0;
    svfloat64_t ab_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t a2_vec = svdupq_n_f64(0.0, 0.0);
    svfloat64_t b2_vec = svdupq_n_f64(0.0, 0.0);
    do {
        svbool_t pg_vec = svwhilelt_b64((unsigned int)i, (unsigned int)n);
        svfloat64_t a_vec = svld1_f64(pg_vec, a + i);
        svfloat64_t b_vec = svld1_f64(pg_vec, b + i);
        ab_vec = svmla_f64_x(pg_vec, ab_vec, a_vec, b_vec);
        a2_vec = svmla_f64_x(pg_vec, a2_vec, a_vec, a_vec);
        b2_vec = svmla_f64_x(pg_vec, b2_vec, b_vec, b_vec);
        i += svcntd();
    } while (i < n);

    nk_f64_t ab = svaddv_f64(svptrue_b32(), ab_vec);
    nk_f64_t a2 = svaddv_f64(svptrue_b32(), a2_vec);
    nk_f64_t b2 = svaddv_f64(svptrue_b32(), b2_vec);
    *result = nk_angular_normalize_f64_neon_(ab, a2, b2);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_SPATIAL_SVE_H