/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/spatial/skylake.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SKYLAKE_H
#define NK_SPATIAL_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/skylake.h" // nk_reduce_add_f32x16_skylake_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2sq_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    // Upcast to f64 for higher precision accumulation
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m256 a_f32x8, b_f32x8;

nk_l2sq_f32_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, a);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps(a);
        b_f32x8 = _mm256_loadu_ps(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    __m512d diff_f64x8 = _mm512_sub_pd(a_f64x8, b_f64x8);
    sum_f64x8 = _mm512_fmadd_pd(diff_f64x8, diff_f64x8, sum_f64x8);
    if (n) goto nk_l2sq_f32_skylake_cycle;

    *result = (nk_f32_t)_mm512_reduce_add_pd(sum_f64x8);
}

NK_PUBLIC void nk_l2_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f32_skylake(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}

NK_INTERNAL nk_f64_t nk_angular_normalize_f64_skylake_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {

    // If both vectors have magnitude 0, the distance is 0.
    if (a2 == 0 && b2 == 0) return 0;
    // If any one of the vectors is 0, the square root of the product is 0,
    // the division is illformed, and the result is 1.
    else if (ab == 0) return 1;

    // We want to avoid the `nk_f32_approximate_inverse_square_root` due to high latency:
    // https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
    // The maximum relative error for this approximation is less than 2^-14, which is 6x lower than
    // for single-precision floats in the `nk_angular_normalize_f64_haswell_` implementation.
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
    // Upcast to f64 for higher precision accumulation
    __m512d dot_f64x8 = _mm512_setzero_pd();
    __m512d a_norm_sq_f64x8 = _mm512_setzero_pd();
    __m512d b_norm_sq_f64x8 = _mm512_setzero_pd();
    __m256 a_f32x8, b_f32x8;

nk_angular_f32_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x8 = _mm256_maskz_loadu_ps(mask, a);
        b_f32x8 = _mm256_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_loadu_ps(a);
        b_f32x8 = _mm256_loadu_ps(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d a_f64x8 = _mm512_cvtps_pd(a_f32x8);
    __m512d b_f64x8 = _mm512_cvtps_pd(b_f32x8);
    dot_f64x8 = _mm512_fmadd_pd(a_f64x8, b_f64x8, dot_f64x8);
    a_norm_sq_f64x8 = _mm512_fmadd_pd(a_f64x8, a_f64x8, a_norm_sq_f64x8);
    b_norm_sq_f64x8 = _mm512_fmadd_pd(b_f64x8, b_f64x8, b_norm_sq_f64x8);
    if (n) goto nk_angular_f32_skylake_cycle;

    nk_f64_t dot_f64 = _mm512_reduce_add_pd(dot_f64x8);
    nk_f64_t a_norm_sq_f64 = _mm512_reduce_add_pd(a_norm_sq_f64x8);
    nk_f64_t b_norm_sq_f64 = _mm512_reduce_add_pd(b_norm_sq_f64x8);
    *result = (nk_f32_t)nk_angular_normalize_f64_skylake_(dot_f64, a_norm_sq_f64, b_norm_sq_f64);
}

NK_PUBLIC void nk_l2sq_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Neumaier summation for improved numerical stability
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();
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
    __m512d x_f64x8 = _mm512_mul_pd(diff_f64x8, diff_f64x8); // x = diff², always >= 0
    // Neumaier TwoSum: t = sum + x
    __m512d t_f64x8 = _mm512_add_pd(sum_f64x8, x_f64x8);
    // Compare |sum| vs x (x is already non-negative, skip abs)
    __m512d abs_sum_f64x8 = _mm512_abs_pd(sum_f64x8);
    __mmask8 sum_ge_x = _mm512_cmp_pd_mask(abs_sum_f64x8, x_f64x8, _CMP_GE_OQ);
    // z = t - larger: recovers approximately the smaller value
    // When |sum| >= x: z = t - sum; else z = t - x
    __m512d z_f64x8 = _mm512_sub_pd(t_f64x8, x_f64x8);
    z_f64x8 = _mm512_mask_sub_pd(z_f64x8, sum_ge_x, t_f64x8, sum_f64x8);
    // error = smaller - z: the rounding error from the addition
    // When |sum| >= x: error = x - z; else error = sum - z
    __m512d error_f64x8 = _mm512_sub_pd(sum_f64x8, z_f64x8);
    error_f64x8 = _mm512_mask_sub_pd(error_f64x8, sum_ge_x, x_f64x8, z_f64x8);
    compensation_f64x8 = _mm512_add_pd(compensation_f64x8, error_f64x8);
    sum_f64x8 = t_f64x8;
    if (n) goto nk_l2sq_f64_skylake_cycle;

    *result = _mm512_reduce_add_pd(_mm512_add_pd(sum_f64x8, compensation_f64x8));
}

NK_PUBLIC void nk_l2_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_l2sq_f64_skylake(a, b, n, result);
    *result = nk_sqrt_f64_haswell_(*result);
}

NK_PUBLIC void nk_angular_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    // Dot2 (Ogita-Rump-Oishi 2005) for cross-product a·b only - it may have cancellation.
    // Self-products ||a||² and ||b||² use simple FMA - all terms are non-negative, no cancellation.
    __m512d dot_sum_f64x8 = _mm512_setzero_pd();
    __m512d dot_compensation_f64x8 = _mm512_setzero_pd();
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
    // TwoProd for cross-product: product = a * b, error = fma(a, b, -product)
    __m512d x_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    __m512d product_error_f64x8 = _mm512_fmsub_pd(a_f64x8, b_f64x8, x_f64x8);
    // Neumaier TwoSum: t = sum + x, with masked error recovery
    __m512d t_f64x8 = _mm512_add_pd(dot_sum_f64x8, x_f64x8);
    __m512d abs_sum_f64x8 = _mm512_abs_pd(dot_sum_f64x8);
    __m512d abs_x_f64x8 = _mm512_abs_pd(x_f64x8);
    __mmask8 sum_ge_x = _mm512_cmp_pd_mask(abs_sum_f64x8, abs_x_f64x8, _CMP_GE_OQ);
    // z = t - larger, error = smaller - z
    __m512d z_f64x8 = _mm512_sub_pd(t_f64x8, x_f64x8);
    z_f64x8 = _mm512_mask_sub_pd(z_f64x8, sum_ge_x, t_f64x8, dot_sum_f64x8);
    __m512d sum_error_f64x8 = _mm512_sub_pd(dot_sum_f64x8, z_f64x8);
    sum_error_f64x8 = _mm512_mask_sub_pd(sum_error_f64x8, sum_ge_x, x_f64x8, z_f64x8);
    dot_sum_f64x8 = t_f64x8;
    dot_compensation_f64x8 = _mm512_add_pd(dot_compensation_f64x8, _mm512_add_pd(sum_error_f64x8, product_error_f64x8));
    // Simple FMA for self-products (no cancellation possible)
    a_norm_sq_f64x8 = _mm512_fmadd_pd(a_f64x8, a_f64x8, a_norm_sq_f64x8);
    b_norm_sq_f64x8 = _mm512_fmadd_pd(b_f64x8, b_f64x8, b_norm_sq_f64x8);
    if (n) goto nk_angular_f64_skylake_cycle;

    nk_f64_t dot_product_f64 = _mm512_reduce_add_pd(_mm512_add_pd(dot_sum_f64x8, dot_compensation_f64x8));
    nk_f64_t a_norm_sq_f64 = _mm512_reduce_add_pd(a_norm_sq_f64x8);
    nk_f64_t b_norm_sq_f64 = _mm512_reduce_add_pd(b_norm_sq_f64x8);
    *result = nk_angular_normalize_f64_skylake_(dot_product_f64, a_norm_sq_f64, b_norm_sq_f64);
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
    nk_b256_vec_t dots_vec;
    nk_dot_f64x8_finalize_skylake(state_a, state_b, state_c, state_d, &dots_vec);
    nk_f64_t dots[4] = {dots_vec.f64s[0], dots_vec.f64s[1], dots_vec.f64s[2], dots_vec.f64s[3]};

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
    nk_b256_vec_t dots_vec;
    nk_dot_f64x8_finalize_skylake(state_a, state_b, state_c, state_d, &dots_vec);
    nk_f64_t dots[4] = {dots_vec.f64s[0], dots_vec.f64s[1], dots_vec.f64s[2], dots_vec.f64s[3]};

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

typedef nk_dot_f32x8_state_skylake_t nk_angular_f32x8_state_skylake_t;

NK_INTERNAL void nk_angular_f32x8_init_skylake(nk_angular_f32x8_state_skylake_t *state) {
    nk_dot_f32x8_init_skylake(state);
}

NK_INTERNAL void nk_angular_f32x8_update_skylake(nk_angular_f32x8_state_skylake_t *state, nk_b256_vec_t a,
                                                 nk_b256_vec_t b) {
    nk_dot_f32x8_update_skylake(state, a, b);
}

NK_INTERNAL void nk_angular_f32x8_finalize_skylake(nk_angular_f32x8_state_skylake_t const *state_a,
                                                   nk_angular_f32x8_state_skylake_t const *state_b,
                                                   nk_angular_f32x8_state_skylake_t const *state_c,
                                                   nk_angular_f32x8_state_skylake_t const *state_d,
                                                   nk_f32_t query_norm, nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                   nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_b128_vec_t dots_vec;
    nk_dot_f32x8_finalize_skylake(state_a, state_b, state_c, state_d, &dots_vec);

    // Build 256-bit F64 vectors for higher precision angular computation
    __m256d dots_f64x4 = _mm256_cvtps_pd(dots_vec.xmm_ps);
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

    // Convert f64x4 → f32x4 and store
    __m128 angular_f32x4 = _mm256_cvtpd_ps(angular_f64x4);
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_f32x8_state_skylake_t nk_l2_f32x8_state_skylake_t;

NK_INTERNAL void nk_l2_f32x8_init_skylake(nk_l2_f32x8_state_skylake_t *state) { nk_dot_f32x8_init_skylake(state); }

NK_INTERNAL void nk_l2_f32x8_update_skylake(nk_l2_f32x8_state_skylake_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_f32x8_update_skylake(state, a, b);
}

NK_INTERNAL void nk_l2_f32x8_finalize_skylake(nk_l2_f32x8_state_skylake_t const *state_a,
                                              nk_l2_f32x8_state_skylake_t const *state_b,
                                              nk_l2_f32x8_state_skylake_t const *state_c,
                                              nk_l2_f32x8_state_skylake_t const *state_d, nk_f32_t query_norm,
                                              nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                              nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_b128_vec_t dots_vec;
    nk_dot_f32x8_finalize_skylake(state_a, state_b, state_c, state_d, &dots_vec);

    // Build 256-bit F64 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m256d dots_f64x4 = _mm256_cvtps_pd(dots_vec.xmm_ps);
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

    // Convert f64x4 → f32x4 and store
    __m128 dist_f32x4 = _mm256_cvtpd_ps(dist_f64x4);
    _mm_storeu_ps(results, dist_f32x4);
}

NK_PUBLIC void nk_l2sq_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m256i a_f16x16, b_f16x16;

nk_l2sq_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_l2sq_f16_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_l2_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f16_skylake(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}

NK_PUBLIC void nk_angular_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m256i a_f16x16, b_f16x16;

nk_angular_f16_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_f16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_f16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x16 = _mm256_loadu_si256((__m256i const *)a);
        b_f16x16 = _mm256_loadu_si256((__m256i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = _mm512_cvtph_ps(a_f16x16);
    __m512 b_f32x16 = _mm512_cvtph_ps(b_f16x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_f16_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_l2sq_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e4m3x16, b_e4m3x16;

nk_l2sq_e4m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_l2sq_e4m3_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_l2_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_e4m3_skylake(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}

NK_PUBLIC void nk_angular_e4m3_skylake(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e4m3x16, b_e4m3x16;

nk_angular_e4m3_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e4m3x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e4m3x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x16 = _mm_loadu_si128((__m128i const *)a);
        b_e4m3x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e4m3x16_to_f32x16_skylake_(a_e4m3x16);
    __m512 b_f32x16 = nk_e4m3x16_to_f32x16_skylake_(b_e4m3x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e4m3_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

NK_PUBLIC void nk_l2sq_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    __m128i a_e5m2x16, b_e5m2x16;

nk_l2sq_e5m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    __m512 b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    __m512 diff_f32x16 = _mm512_sub_ps(a_f32x16, b_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_f32x16, diff_f32x16, sum_f32x16);
    if (n) goto nk_l2sq_e5m2_skylake_cycle;

    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_l2_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_e5m2_skylake(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}

NK_PUBLIC void nk_angular_e5m2_skylake(nk_e5m2_t const *a, nk_e5m2_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m128i a_e5m2x16, b_e5m2x16;

nk_angular_e5m2_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        a_e5m2x16 = _mm_maskz_loadu_epi8(mask, a);
        b_e5m2x16 = _mm_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e5m2x16 = _mm_loadu_si128((__m128i const *)a);
        b_e5m2x16 = _mm_loadu_si128((__m128i const *)b);
        a += 16, b += 16, n -= 16;
    }
    __m512 a_f32x16 = nk_e5m2x16_to_f32x16_skylake_(a_e5m2x16);
    __m512 b_f32x16 = nk_e5m2x16_to_f32x16_skylake_(b_e5m2x16);
    dot_f32x16 = _mm512_fmadd_ps(a_f32x16, b_f32x16, dot_f32x16);
    a_norm_sq_f32x16 = _mm512_fmadd_ps(a_f32x16, a_f32x16, a_norm_sq_f32x16);
    b_norm_sq_f32x16 = _mm512_fmadd_ps(b_f32x16, b_f32x16, b_norm_sq_f32x16);
    if (n) goto nk_angular_e5m2_skylake_cycle;

    nk_f32_t dot_f32 = nk_reduce_add_f32x16_skylake_(dot_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_f32, a_norm_sq_f32, b_norm_sq_f32);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SKYLAKE_H