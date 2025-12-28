/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/spatial/sapphire.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SAPPHIRE_H
#define NK_SPATIAL_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_l2_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_f16_sapphire(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}
NK_PUBLIC void nk_l2sq_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512h distance_sq_f16x32 = _mm512_setzero_ph();
    __m512i a_f16x32, b_f16x32;

nk_l2sq_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a);
        b_f16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    __m512h diff_f16x32 = _mm512_sub_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32));
    distance_sq_f16x32 = _mm512_fmadd_ph(diff_f16x32, diff_f16x32, distance_sq_f16x32);
    if (n) goto nk_l2sq_f16_sapphire_cycle;

    *result = _mm512_reduce_add_ph(distance_sq_f16x32);
}

NK_PUBLIC void nk_angular_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512h dot_product_f16x32 = _mm512_setzero_ph();
    __m512h a_norm_sq_f16x32 = _mm512_setzero_ph();
    __m512h b_norm_sq_f16x32 = _mm512_setzero_ph();
    __m512i a_f16x32, b_f16x32;

nk_angular_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_f16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_epi16(a);
        b_f16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    dot_product_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(b_f16x32),
                                         dot_product_f16x32);
    a_norm_sq_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(a_f16x32), _mm512_castsi512_ph(a_f16x32), a_norm_sq_f16x32);
    b_norm_sq_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(b_f16x32), _mm512_castsi512_ph(b_f16x32), b_norm_sq_f16x32);
    if (n) goto nk_angular_f16_sapphire_cycle;

    nk_f32_t dot_product_f32 = _mm512_reduce_add_ph(dot_product_f16x32);
    nk_f32_t a_norm_sq_f32 = _mm512_reduce_add_ph(a_norm_sq_f16x32);
    nk_f32_t b_norm_sq_f32 = _mm512_reduce_add_ph(b_norm_sq_f16x32);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

typedef nk_dot_f16x32_state_sapphire_t nk_angular_f16x32_state_sapphire_t;
NK_INTERNAL void nk_angular_f16x32_init_sapphire(nk_angular_f16x32_state_sapphire_t *state) {
    nk_dot_f16x32_init_sapphire(state);
}
NK_INTERNAL void nk_angular_f16x32_update_sapphire(nk_angular_f16x32_state_sapphire_t *state, nk_b512_vec_t a,
                                                   nk_b512_vec_t b) {
    nk_dot_f16x32_update_sapphire(state, a, b);
}
NK_INTERNAL void nk_angular_f16x32_finalize_sapphire(nk_angular_f16x32_state_sapphire_t const *state_a,
                                                     nk_angular_f16x32_state_sapphire_t const *state_b,
                                                     nk_angular_f16x32_state_sapphire_t const *state_c,
                                                     nk_angular_f16x32_state_sapphire_t const *state_d,
                                                     nk_f32_t query_norm, nk_f32_t target_norm_a,
                                                     nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                     nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f16x32_finalize_sapphire(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel processing
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute products = query_norm * target_norm for all 4
    __m128 products_f32x4 = _mm_mul_ps(query_norm_f32x4, target_norms_f32x4);

    // Vectorized rsqrt with Newton-Raphson refinement
    __m128 rsqrt_f32x4 = _mm_rsqrt_ps(products_f32x4);
    __m128 half_f32x4 = _mm_set1_ps(0.5f);
    __m128 three_f32x4 = _mm_set1_ps(3.0f);
    __m128 nr_f32x4 = _mm_mul_ps(products_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_mul_ps(nr_f32x4, rsqrt_f32x4);
    nr_f32x4 = _mm_sub_ps(three_f32x4, nr_f32x4);
    rsqrt_f32x4 = _mm_mul_ps(_mm_mul_ps(half_f32x4, rsqrt_f32x4), nr_f32x4);

    // Compute angular: 1 - dot * rsqrt(product)
    __m128 normalized_f32x4 = _mm_mul_ps(dots_f32x4, rsqrt_f32x4);
    __m128 ones_f32x4 = _mm_set1_ps(1.0f);
    __m128 angular_f32x4 = _mm_sub_ps(ones_f32x4, normalized_f32x4);

    // Store results
    _mm_storeu_ps(results, angular_f32x4);
}

typedef nk_dot_f16x32_state_sapphire_t nk_l2_f16x32_state_sapphire_t;
NK_INTERNAL void nk_l2_f16x32_init_sapphire(nk_l2_f16x32_state_sapphire_t *state) {
    nk_dot_f16x32_init_sapphire(state);
}
NK_INTERNAL void nk_l2_f16x32_update_sapphire(nk_l2_f16x32_state_sapphire_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_f16x32_update_sapphire(state, a, b);
}
NK_INTERNAL void nk_l2_f16x32_finalize_sapphire(nk_l2_f16x32_state_sapphire_t const *state_a,
                                                nk_l2_f16x32_state_sapphire_t const *state_b,
                                                nk_l2_f16x32_state_sapphire_t const *state_c,
                                                nk_l2_f16x32_state_sapphire_t const *state_d, nk_f32_t query_norm,
                                                nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                                nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_f16x32_finalize_sapphire(state_a, state_b, state_c, state_d, dots);

    // Build 128-bit F32 vectors for parallel L2 distance: sqrt(q² + t² - 2*dot)
    __m128 dots_f32x4 = _mm_loadu_ps(dots);
    __m128 query_norm_f32x4 = _mm_set1_ps(query_norm);
    __m128 target_norms_f32x4 = _mm_set_ps(target_norm_d, target_norm_c, target_norm_b, target_norm_a);

    // Compute squared norms in parallel
    __m128 query_sq_f32x4 = _mm_mul_ps(query_norm_f32x4, query_norm_f32x4);
    __m128 target_sq_f32x4 = _mm_mul_ps(target_norms_f32x4, target_norms_f32x4);

    // Compute distance squared: q² + t² - 2*dot using FMA
    __m128 two_f32x4 = _mm_set1_ps(2.0f);
    __m128 sum_sq_f32x4 = _mm_add_ps(query_sq_f32x4, target_sq_f32x4);
    __m128 dist_sq_f32x4 = _mm_fnmadd_ps(two_f32x4, dots_f32x4, sum_sq_f32x4);

    // Clamp negative to zero, then sqrt
    __m128 zeros_f32x4 = _mm_setzero_ps();
    __m128 clamped_f32x4 = _mm_max_ps(dist_sq_f32x4, zeros_f32x4);
    __m128 dist_f32x4 = _mm_sqrt_ps(clamped_f32x4);

    // Store results
    _mm_storeu_ps(results, dist_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SAPPHIRE_H