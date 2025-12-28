/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Sierra Forest CPUs.
 *  @file include/numkong/spatial/sierra.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SIERRA_H
#define NK_SPATIAL_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "avx2vnni")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,avx2vnni"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_angular_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_lddqu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_lddqu_si256((__m256i const *)(b + i));
        dot_product_i32x8 = _mm256_dpbssds_epi32(dot_product_i32x8, a_i8x32, b_i8x32);
        a_norm_sq_i32x8 = _mm256_dpbssds_epi32(a_norm_sq_i32x8, a_i8x32, a_i8x32);
        b_norm_sq_i32x8 = _mm256_dpbssds_epi32(b_norm_sq_i32x8, b_i8x32, b_i8x32);
    }

    // Further reduce to a single sum for each vector
    nk_i32_t dot_product_i32 = nk_reduce_add_i32x8_haswell_(dot_product_i32x8);
    nk_i32_t a_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(a_norm_sq_i32x8);
    nk_i32_t b_norm_sq_i32 = nk_reduce_add_i32x8_haswell_(b_norm_sq_i32x8);

    // Take care of the tail:
    for (; i < n; ++i) {
        nk_i32_t a_element_i32 = a[i], b_element_i32 = b[i];
        dot_product_i32 += a_element_i32 * b_element_i32;
        a_norm_sq_i32 += a_element_i32 * a_element_i32;
        b_norm_sq_i32 += b_element_i32 * b_element_i32;
    }

    *result = nk_angular_normalize_f32_haswell_(dot_product_i32, a_norm_sq_i32, b_norm_sq_i32);
}

typedef nk_dot_i8x32_state_sierra_t nk_angular_i8x32_state_sierra_t;
NK_INTERNAL void nk_angular_i8x32_init_sierra(nk_angular_i8x32_state_sierra_t *state) {
    nk_dot_i8x32_init_sierra(state);
}
NK_INTERNAL void nk_angular_i8x32_update_sierra(nk_angular_i8x32_state_sierra_t *state, nk_b256_vec_t a,
                                                nk_b256_vec_t b) {
    nk_dot_i8x32_update_sierra(state, a, b);
}
NK_INTERNAL void nk_angular_i8x32_finalize_sierra(nk_angular_i8x32_state_sierra_t const *state_a,
                                                  nk_angular_i8x32_state_sierra_t const *state_b,
                                                  nk_angular_i8x32_state_sierra_t const *state_c,
                                                  nk_angular_i8x32_state_sierra_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_i32_t dots_i32[4];
    nk_dot_i8x32_finalize_sierra(state_a, state_b, state_c, state_d, dots_i32);

    // Convert dots to f32 and build vectors for parallel processing
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_i32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // products = query_norm_sq * target_norms_sq
    __m128 products_f32x4 = _mm_mul_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);

    // rsqrt with Newton-Raphson refinement: x' = x * (1.5 - 0.5 * val * x * x)
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_halves_f32x4 = _mm_set1_ps(1.5f);
    __m128 rsqrt_sq_f32x4 = _mm_mul_ps(rsqrt_f32x4, rsqrt_f32x4);
    __m128 half_prod_f32x4 = _mm_mul_ps(half_f32x4, products_f32x4);
    __m128 muls_f32x4 = _mm_mul_ps(half_prod_f32x4, rsqrt_sq_f32x4);
    __m128 refinement_f32x4 = _mm_sub_ps(three_halves_f32x4, muls_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(rsqrt_f32x4, refinement_f32x4);

    // normalized = dots * rsqrt(products)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);

    // angular = 1 - normalized
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_i8x32_state_sierra_t nk_l2_i8x32_state_sierra_t;
NK_INTERNAL void nk_l2_i8x32_init_sierra(nk_l2_i8x32_state_sierra_t *state) { nk_dot_i8x32_init_sierra(state); }
NK_INTERNAL void nk_l2_i8x32_update_sierra(nk_l2_i8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    nk_dot_i8x32_update_sierra(state, a, b);
}
NK_INTERNAL void nk_l2_i8x32_finalize_sierra(nk_l2_i8x32_state_sierra_t const *state_a,
                                             nk_l2_i8x32_state_sierra_t const *state_b,
                                             nk_l2_i8x32_state_sierra_t const *state_c,
                                             nk_l2_i8x32_state_sierra_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_i32_t dots_i32[4];
    nk_dot_i8x32_finalize_sierra(state_a, state_b, state_c, state_d, dots_i32);

    // Convert dots to f32 and build vectors for parallel processing
    __m128 dots_f32x4 = _mm_cvtepi32_ps(_mm_loadu_si128((__m128i const *)dots_i32));
    __m128 query_norm_sq_f32x4 = _mm_set1_ps(query_norm * query_norm);
    __m128 target_norms_sq_f32x4 = _mm_set_ps(target_norm_d * target_norm_d, target_norm_c * target_norm_c,
                                              target_norm_b * target_norm_b, target_norm_a * target_norm_a);

    // L2 distance: sqrt(query_sq + target_sq - 2*dot)
    // dist_sq = query_norm_sq + target_norms_sq - 2*dots
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 two_dots_f32x4 = _mm_mul_ps(two_f32x4, dots_f32x4);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_norm_sq_f32x4, target_norms_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_sub_ps(sum_sq_f32x4, two_dots_f32x4);

    // Clamp negatives to zero and take sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    dist_sq_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(dist_sq_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SIERRA_H