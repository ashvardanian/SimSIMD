/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/spatial/skylake.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SKYLAKE_H
#define NK_SPATIAL_SKYLAKE_H

#if _NK_TARGET_X86
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f32_skylake(a, b, n, result);
    *result = _nk_sqrt_f32_haswell(*result);
}
NK_PUBLIC void nk_l2sq_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 d2_vec = _mm512_setzero();
    __m512 a_vec, b_vec;

nk_l2sq_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
    d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    if (n) goto nk_l2sq_f32_skylake_cycle;

    *result = _nk_reduce_add_f32x16_skylake(d2_vec);
}

NK_INTERNAL nk_f64_t _nk_angular_normalize_f64_skylake(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0) return 1;

    // We want to avoid the `nk_f32_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The maximum relative error for this approximation is less than 2^-14, which is 6x lower than
    // for single-precision floats in the `_nk_angular_normalize_f64_haswell` implementation.
    // Mysteriously, MSVC has no `_mm_rsqrt14_pd` intrinsic, but has its masked variants,
    // so let's use `_mm_maskz_rsqrt14_pd(0xFF, ...)` instead.
    __m128d squares = _mm_set_pd(a2, b2);
    __m128d rsqrts = _mm_maskz_rsqrt14_pd(0xFF, squares);

    // Let's implement a single Newton-Raphson iteration to refine the result.
    // This is how it affects downstream applications for 1536-dimensional vectors:
    //
    //      DType     Baseline Error       Old NumKong Error    New NumKong Error
    //      bfloat16  1.89e-08 ± 1.59e-08  3.07e-07 ± 3.09e-07  3.53e-09 ± 2.70e-09
    //      float16   1.67e-02 ± 1.44e-02  2.68e-05 ± 1.95e-05  2.02e-05 ± 1.39e-05
    //      float32   2.21e-08 ± 1.65e-08  3.47e-07 ± 3.49e-07  3.77e-09 ± 2.84e-09
    //      float64   0.00e+00 ± 0.00e+00  3.80e-07 ± 4.50e-07  1.35e-11 ± 1.85e-11
    //      int8      0.00e+00 ± 0.00e+00  4.60e-04 ± 3.36e-04  4.20e-04 ± 4.88e-04
    //
    // https://en.wikipedia.org/wiki/Newton%27s_method
    rsqrts = _mm_add_pd( //
        _mm_mul_pd(_mm_set1_pd(1.5), rsqrts),
        _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(squares, _mm_set1_pd(-0.5)), rsqrts), _mm_mul_pd(rsqrts, rsqrts)));

    nk_f64_t a2_reciprocal = _mm_cvtsd_f64(_mm_unpackhi_pd(rsqrts, rsqrts));
    nk_f64_t b2_reciprocal = _mm_cvtsd_f64(rsqrts);
    nk_f64_t result = 1 - ab * a2_reciprocal * b2_reciprocal;
    return result > 0 ? result : 0;
}

NK_PUBLIC void nk_angular_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_product_f32x16 = _mm512_setzero();
    __m512 a_norm_sq_f32x16 = _mm512_setzero();
    __m512 b_norm_sq_f32x16 = _mm512_setzero();
    __m512 a_f32x16, b_f32x16;

nk_angular_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    dot_product_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_product_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_f32_skylake_cycle;

    nk_f64_t dot_product_f64 = _nk_reduce_add_f32x16_skylake(dot_product_f32x16);
    nk_f64_t a_norm_sq_f64 = _nk_reduce_add_f32x16_skylake(a_norm_sq_f32x16);
    nk_f64_t b_norm_sq_f64 = _nk_reduce_add_f32x16_skylake(b_norm_sq_f32x16);
    *result = _nk_angular_normalize_f64_skylake(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

NK_PUBLIC void nk_l2_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_l2sq_f64_skylake(a, b, n, result);
    *result = _nk_sqrt_f64_haswell(*result);
}
NK_PUBLIC void nk_l2sq_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d distance_sq_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

nk_l2sq_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d diff_f64x8 = _mm512_sub_pd(a_f64x8, b_f64x8);
    distance_sq_f64x8 = _mm512_fmadd_pd(diff_f64x8, diff_f64x8, distance_sq_f64x8);
    if (n) goto nk_l2sq_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(distance_sq_f64x8);
}

NK_PUBLIC void nk_angular_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d dot_product_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d a_f64x8, b_f64x8;

nk_angular_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    dot_product_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, dot_product_f64x8);
    a_norm_sq_f64x8 = _mm512_fmadd_pd(a_f64x8, a_f64x8, a_norm_sq_f64x8);
    b_norm_sq_f64x8 = _mm512_fmadd_pd(b_f64x8, b_f64x8, b_norm_sq_f64x8);
    if (n) goto nk_angular_f64_skylake_cycle;

    nk_f64_t dot_product_f64 = _mm512_reduce_add_pd(dot_product_f64x8);
    nk_f64_t a_norm_sq_f64 = _mm512_reduce_add_pd(a_norm_sq_f64x8);
    nk_f64_t b_norm_sq_f64 = _mm512_reduce_add_pd(b_norm_sq_f64x8);
    *result = _nk_angular_normalize_f64_skylake(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
}

typedef nk_dot_f64x8_state_skylake_t nk_angular_f64x8_state_skylake_t;
NK_INTERNAL void nk_angular_f64x8_init_skylake(nk_angular_f64x8_state_skylake_t *state) {
    nk_dot_f64x8_init_skylake(state);
}
NK_INTERNAL void nk_angular_f64x8_update_skylake(nk_angular_f64x8_state_skylake_t *state, nk_b512_vec_t a,
                                                 nk_b512_vec_t b) {
    nk_dot_f64x8_update_skylake(state, a, b);
}
NK_INTERNAL void nk_angular_f64x8_finalize_skylake(nk_angular_f64x8_state_skylake_t const *state_a,
                                                   nk_angular_f64x8_state_skylake_t const *state_b,
                                                   nk_angular_f64x8_state_skylake_t const *state_c,
                                                   nk_angular_f64x8_state_skylake_t const *state_d, nk_f64_t query_norm,
                                                   nk_f64_t target_norm_a, nk_f64_t target_norm_b,
                                                   nk_f64_t target_norm_c, nk_f64_t target_norm_d, nk_f64_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f64_t dots[4];
    nk_dot_f64x8_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel angular computation (use 4 F64 values = 256-bit)
    __m256d dots_f64x4 = _mm256_loadu_pd(dots);
    __m256d query_norm_f64x4 = _mm256_set1_pd(query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m256d products_f64x4 = _mm256_mul_pd(query_norm_f64x4, target_norms_f64x4);

    // Compute sqrt and normalize: 1 - dot / sqrt(product)
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);

    // Store results
    _mm256_storeu_pd(results, angular_f64x4);
}

typedef nk_dot_f64x8_state_skylake_t nk_l2_f64x8_state_skylake_t;
NK_INTERNAL void nk_l2_f64x8_init_skylake(nk_l2_f64x8_state_skylake_t *state) { nk_dot_f64x8_init_skylake(state); }
NK_INTERNAL void nk_l2_f64x8_update_skylake(nk_l2_f64x8_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_f64x8_update_skylake(state, a, b);
}
NK_INTERNAL void nk_l2_f64x8_finalize_skylake(nk_l2_f64x8_state_skylake_t const *state_a,
                                              nk_l2_f64x8_state_skylake_t const *state_b,
                                              nk_l2_f64x8_state_skylake_t const *state_c,
                                              nk_l2_f64x8_state_skylake_t const *state_d, nk_f64_t query_norm,
                                              nk_f64_t target_norm_a, nk_f64_t target_norm_b, nk_f64_t target_norm_c,
                                              nk_f64_t target_norm_d, nk_f64_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f64_t dots[4];
    nk_dot_f64x8_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_loadu_pd(dots);
    __m256d query_norm_f64x4 = _mm256_set1_pd(query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m256d query_sq_f64x4 = _mm256_mul_pd(query_norm_f64x4, query_norm_f64x4);
    __m256d target_sq_f64x4 = _mm256_mul_pd(target_norms_f64x4, target_norms_f64x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sq_f64x4, target_sq_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);

    // Clamp negative to zero, then sqrt
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);

    // Store results
    _mm256_storeu_pd(results, dist_f64x4);
}

typedef nk_dot_f32x16_state_skylake_t nk_angular_f32x16_state_skylake_t;
NK_INTERNAL void nk_angular_f32x16_init_skylake(nk_angular_f32x16_state_skylake_t *state) {
    nk_dot_f32x16_init_skylake(state);
}
NK_INTERNAL void nk_angular_f32x16_update_skylake(nk_angular_f32x16_state_skylake_t *state, nk_b512_vec_t a,
                                                  nk_b512_vec_t b) {
    nk_dot_f32x16_update_skylake(state, a, b);
}
NK_INTERNAL void nk_angular_f32x16_finalize_skylake(nk_angular_f32x16_state_skylake_t const *state_a,
                                                    nk_angular_f32x16_state_skylake_t const *state_b,
                                                    nk_angular_f32x16_state_skylake_t const *state_c,
                                                    nk_angular_f32x16_state_skylake_t const *state_d,
                                                    nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                    nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f32x16_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for higher precision angular computation
    __m256d dots_f64x4 = _mm256_set_pd((nk_f64_t)dots[3], (nk_f64_t)dots[2], //
                                       (nk_f64_t)dots[1], (nk_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((nk_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((nk_f64_t)target_norm_d, (nk_f64_t)target_norm_c,
                                               (nk_f64_t)target_norm_b, (nk_f64_t)target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m256d products_f64x4 = _mm256_mul_pd(query_norm_f64x4, target_norms_f64x4);

    // Compute sqrt and normalize: 1 - dot / sqrt(product)
    __m256d sqrt_products_f64x4 = _mm256_sqrt_pd(products_f64x4);
    __m256d normalized_f64x4 = _mm256_div_pd(dots_f64x4, sqrt_products_f64x4);
    __m256d ones_f64x4 = _mm256_set1_pd(1.0);
    __m256d angular_f64x4 = _mm256_sub_pd(ones_f64x4, normalized_f64x4);

    // Store results (convert f64 → f32)
    nk_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, angular_f64x4);
    results[0] = (nk_f32_t)u.f64s[0];
    results[1] = (nk_f32_t)u.f64s[1];
    results[2] = (nk_f32_t)u.f64s[2];
    results[3] = (nk_f32_t)u.f64s[3];
}

typedef nk_dot_f32x16_state_skylake_t nk_l2_f32x16_state_skylake_t;
NK_INTERNAL void nk_l2_f32x16_init_skylake(nk_l2_f32x16_state_skylake_t *state) { nk_dot_f32x16_init_skylake(state); }
NK_INTERNAL void nk_l2_f32x16_update_skylake(nk_l2_f32x16_state_skylake_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_f32x16_update_skylake(state, a, b);
}
NK_INTERNAL void nk_l2_f32x16_finalize_skylake(nk_l2_f32x16_state_skylake_t const *state_a,
                                               nk_l2_f32x16_state_skylake_t const *state_b,
                                               nk_l2_f32x16_state_skylake_t const *state_c,
                                               nk_l2_f32x16_state_skylake_t const *state_d, nk_f32_t query_norm,
                                               nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                               nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f32x16_finalize_skylake(state_a, state_b, state_c, state_d, dots);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_set_pd((nk_f64_t)dots[3], (nk_f64_t)dots[2], //
                                       (nk_f64_t)dots[1], (nk_f64_t)dots[0]);
    __m256d query_norm_f64x4 = _mm256_set1_pd((nk_f64_t)query_norm);
    __m256d target_norms_f64x4 = _mm256_set_pd((nk_f64_t)target_norm_d, (nk_f64_t)target_norm_c,
                                               (nk_f64_t)target_norm_b, (nk_f64_t)target_norm_a);

    // Compute squared norms in parallel
    __m256d query_sq_f64x4 = _mm256_mul_pd(query_norm_f64x4, query_norm_f64x4);
    __m256d target_sq_f64x4 = _mm256_mul_pd(target_norms_f64x4, target_norms_f64x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m256d two_f64x4 = _mm256_set1_pd(2.0);
    __m256d sum_sq_f64x4 = _mm256_add_pd(query_sq_f64x4, target_sq_f64x4);
    __m256d dist_sq_f64x4 = _mm256_fnmadd_pd(two_f64x4, dots_f64x4, sum_sq_f64x4);

    // Clamp negative to zero, then sqrt
    __m256d zeros_f64x4 = _mm256_setzero_pd();
    __m256d clamped_f64x4 = _mm256_max_pd(dist_sq_f64x4, zeros_f64x4);
    __m256d dist_f64x4 = _mm256_sqrt_pd(clamped_f64x4);

    // Store results (convert f64 → f32)
    nk_b256_vec_t u;
    _mm256_storeu_pd(u.f64s, dist_f64x4);
    results[0] = (nk_f32_t)u.f64s[0];
    results[1] = (nk_f32_t)u.f64s[1];
    results[2] = (nk_f32_t)u.f64s[2];
    results[3] = (nk_f32_t)u.f64s[3];
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // _NK_TARGET_X86

#endif // NK_SPATIAL_SKYLAKE_H