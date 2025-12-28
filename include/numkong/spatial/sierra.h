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

typedef nk_dot_i8x64_state_sierra_t nk_angular_i8x64_state_sierra_t;
NK_INTERNAL void nk_angular_i8x64_init_sierra(nk_angular_i8x64_state_sierra_t *state) {
    nk_dot_i8x64_init_sierra(state);
}
NK_INTERNAL void nk_angular_i8x64_update_sierra(nk_angular_i8x64_state_sierra_t *state, nk_b512_vec_t a,
                                                nk_b512_vec_t b) {
    nk_dot_i8x64_update_sierra(state, a, b);
}
NK_INTERNAL void nk_angular_i8x64_finalize_sierra(nk_angular_i8x64_state_sierra_t const *state_a,
                                                  nk_angular_i8x64_state_sierra_t const *state_b,
                                                  nk_angular_i8x64_state_sierra_t const *state_c,
                                                  nk_angular_i8x64_state_sierra_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states
    nk_distance_t dot_product_a, dot_product_b, dot_product_c, dot_product_d;
    nk_dot_i8x64_finalize_sierra(state_a, &dot_product_a);
    nk_dot_i8x64_finalize_sierra(state_b, &dot_product_b);
    nk_dot_i8x64_finalize_sierra(state_c, &dot_product_c);
    nk_dot_i8x64_finalize_sierra(state_d, &dot_product_d);

    // Compute squared norms (loop-unrolled)
    nk_f32_t query_norm_sq = (nk_f32_t)query_norm * (nk_f32_t)query_norm;
    nk_f32_t target_norm_sq_a = (nk_f32_t)target_norm_a * (nk_f32_t)target_norm_a;
    nk_f32_t target_norm_sq_b = (nk_f32_t)target_norm_b * (nk_f32_t)target_norm_b;
    nk_f32_t target_norm_sq_c = (nk_f32_t)target_norm_c * (nk_f32_t)target_norm_c;
    nk_f32_t target_norm_sq_d = (nk_f32_t)target_norm_d * (nk_f32_t)target_norm_d;

    // Compute angular distances (loop-unrolled)
    results[0] = nk_angular_normalize_f32_haswell_((nk_f32_t)dot_product_a, query_norm_sq, target_norm_sq_a);
    results[1] = nk_angular_normalize_f32_haswell_((nk_f32_t)dot_product_b, query_norm_sq, target_norm_sq_b);
    results[2] = nk_angular_normalize_f32_haswell_((nk_f32_t)dot_product_c, query_norm_sq, target_norm_sq_c);
    results[3] = nk_angular_normalize_f32_haswell_((nk_f32_t)dot_product_d, query_norm_sq, target_norm_sq_d);
}

typedef nk_dot_i8x64_state_sierra_t nk_l2_i8x64_state_sierra_t;
NK_INTERNAL void nk_l2_i8x64_init_sierra(nk_l2_i8x64_state_sierra_t *state) { nk_dot_i8x64_init_sierra(state); }
NK_INTERNAL void nk_l2_i8x64_update_sierra(nk_l2_i8x64_state_sierra_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_i8x64_update_sierra(state, a, b);
}
NK_INTERNAL void nk_l2_i8x64_finalize_sierra(nk_l2_i8x64_state_sierra_t const *state_a,
                                             nk_l2_i8x64_state_sierra_t const *state_b,
                                             nk_l2_i8x64_state_sierra_t const *state_c,
                                             nk_l2_i8x64_state_sierra_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states
    nk_distance_t dot_product_a, dot_product_b, dot_product_c, dot_product_d;
    nk_dot_i8x64_finalize_sierra(state_a, &dot_product_a);
    nk_dot_i8x64_finalize_sierra(state_b, &dot_product_b);
    nk_dot_i8x64_finalize_sierra(state_c, &dot_product_c);
    nk_dot_i8x64_finalize_sierra(state_d, &dot_product_d);

    // Compute squared norms (loop-unrolled)
    nk_f32_t query_norm_sq = (nk_f32_t)query_norm * (nk_f32_t)query_norm;
    nk_f32_t target_norm_sq_a = (nk_f32_t)target_norm_a * (nk_f32_t)target_norm_a;
    nk_f32_t target_norm_sq_b = (nk_f32_t)target_norm_b * (nk_f32_t)target_norm_b;
    nk_f32_t target_norm_sq_c = (nk_f32_t)target_norm_c * (nk_f32_t)target_norm_c;
    nk_f32_t target_norm_sq_d = (nk_f32_t)target_norm_d * (nk_f32_t)target_norm_d;

    // Compute squared distances (loop-unrolled)
    nk_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - (nk_f32_t)2 * (nk_f32_t)dot_product_a;
    nk_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - (nk_f32_t)2 * (nk_f32_t)dot_product_b;
    nk_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - (nk_f32_t)2 * (nk_f32_t)dot_product_c;
    nk_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - (nk_f32_t)2 * (nk_f32_t)dot_product_d;

    // Use 4-way SSE sqrt (128-bit)
    __m128 dist_sq_vec = _mm_set_ps((float)(dist_sq_d < 0 ? 0 : dist_sq_d), (float)(dist_sq_c < 0 ? 0 : dist_sq_c),
                                    (float)(dist_sq_b < 0 ? 0 : dist_sq_b), (float)(dist_sq_a < 0 ? 0 : dist_sq_a));
    __m128 dist_vec = _mm_sqrt_ps(dist_sq_vec);

    // Store results using nk_b512_vec_t
    nk_b512_vec_t storage;
    _mm_storeu_ps(storage.f32s, dist_vec);
    results[0] = storage.f32s[0];
    results[1] = storage.f32s[1];
    results[2] = storage.f32s[2];
    results[3] = storage.f32s[3];
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SIERRA_H