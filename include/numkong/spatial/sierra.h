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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

#include "numkong/types.h"
#include "numkong/reduce/haswell.h" // nk_reduce_add_i32x8_haswell_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_angular_i8_sierra(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {

    __m256i dot_product_i32x8 = _mm256_setzero_si256();
    __m256i a_norm_sq_i32x8 = _mm256_setzero_si256();
    __m256i b_norm_sq_i32x8 = _mm256_setzero_si256();

    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = _mm256_loadu_si256((__m256i const *)(a + i));
        __m256i b_i8x32 = _mm256_loadu_si256((__m256i const *)(b + i));
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
                                                nk_b256_vec_t b, nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_dot_i8x32_update_sierra(state, a, b, depth_offset, active_dimensions);
}
NK_INTERNAL void nk_angular_i8x32_finalize_sierra(nk_angular_i8x32_state_sierra_t const *state_a,
                                                  nk_angular_i8x32_state_sierra_t const *state_b,
                                                  nk_angular_i8x32_state_sierra_t const *state_c,
                                                  nk_angular_i8x32_state_sierra_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d,
                                                  nk_size_t total_dimensions, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_i8x32_finalize_sierra(state_a, state_b, state_c, state_d, &dots_vec, total_dimensions);
    nk_angular_f32x4_finalize_haswell_(_mm_cvtepi32_ps(dots_vec.xmm), query_norm, target_norm_a, target_norm_b,
                                       target_norm_c, target_norm_d, results);
}

typedef nk_dot_i8x32_state_sierra_t nk_l2_i8x32_state_sierra_t;
NK_INTERNAL void nk_l2_i8x32_init_sierra(nk_l2_i8x32_state_sierra_t *state) { nk_dot_i8x32_init_sierra(state); }
NK_INTERNAL void nk_l2_i8x32_update_sierra(nk_l2_i8x32_state_sierra_t *state, nk_b256_vec_t a, nk_b256_vec_t b,
                                           nk_size_t depth_offset, nk_size_t active_dimensions) {
    nk_dot_i8x32_update_sierra(state, a, b, depth_offset, active_dimensions);
}
NK_INTERNAL void nk_l2_i8x32_finalize_sierra(nk_l2_i8x32_state_sierra_t const *state_a,
                                             nk_l2_i8x32_state_sierra_t const *state_b,
                                             nk_l2_i8x32_state_sierra_t const *state_c,
                                             nk_l2_i8x32_state_sierra_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_size_t total_dimensions, nk_f32_t *results) {
    nk_b128_vec_t dots_vec;
    nk_dot_i8x32_finalize_sierra(state_a, state_b, state_c, state_d, &dots_vec, total_dimensions);
    nk_l2_f32x4_finalize_haswell_(_mm_cvtepi32_ps(dots_vec.xmm), query_norm, target_norm_a, target_norm_b,
                                  target_norm_c, target_norm_d, results);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SIERRA_H
