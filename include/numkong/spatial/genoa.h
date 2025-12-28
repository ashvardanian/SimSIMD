/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for AMD Genoa CPUs.
 *  @file include/numkong/spatial/genoa.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_GENOA_H
#define NK_SPATIAL_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/skylake.h" // nk_reduce_add_f32x16_skylake_

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL __m512i nk_substract_bf16x32_genoa_(__m512i a_i16, __m512i b_i16) {

    nk_b512_vec_t d_odd, d_even, d, a_f32_even, b_f32_even, d_f32_even, a_f32_odd, b_f32_odd, d_f32_odd, a, b;
    a.zmm = a_i16;
    b.zmm = b_i16;

    // There are several approaches to perform subtraction in `bf16`. The first one is:
    //
    //      Perform a couple of casts - each is a bitshift. To convert `bf16` to `f32`,
    //      expand it to 32-bit integers, then shift the bits by 16 to the left.
    //      Then subtract as floats, and shift back. During expansion, we will double the space,
    //      and should use separate registers for top and bottom halves.
    //      Some compilers don't have `_mm512_extracti32x8_epi32`, so we use `_mm512_extracti64x4_epi64`:
    //
    //          a_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a_i16)), 16));
    //          b_f32_bot.fvec = _mm512_castsi512_ps(_mm512_slli_epi32(
    //              _mm512_cvtepu16_epi32(_mm512_castsi512_si256(b_i16)), 16));
    //          a_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(a_i16, 1)), 16));
    //          b_f32_top.fvec =_mm512_castsi512_ps(
    //              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(b_i16, 1)), 16));
    //          d_f32_top.fvec = _mm512_sub_ps(a_f32_top.fvec, b_f32_top.fvec);
    //          d_f32_bot.fvec = _mm512_sub_ps(a_f32_bot.fvec, b_f32_bot.fvec);
    //          d.ivec = _mm512_castsi256_si512(_mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_bot.fvec), 16)));
    //          d.ivec = _mm512_inserti64x4(d.ivec, _mm512_cvtepi32_epi16(
    //              _mm512_srli_epi32(_mm512_castps_si512(d_f32_top.fvec), 16)), 1);
    //
    // Instead of using multple shifts and an insertion, we can achieve similar result with fewer expensive
    // calls to `_mm512_permutex2var_epi16`, or a cheap `_mm512_mask_shuffle_epi8` and blend:
    //
    a_f32_odd.zmm = _mm512_and_si512(a_i16, _mm512_set1_epi32(0xFFFF0000));
    a_f32_even.zmm = _mm512_slli_epi32(a_i16, 16);
    b_f32_odd.zmm = _mm512_and_si512(b_i16, _mm512_set1_epi32(0xFFFF0000));
    b_f32_even.zmm = _mm512_slli_epi32(b_i16, 16);

    d_f32_odd.zmm_ps = _mm512_sub_ps(a_f32_odd.zmm_ps, b_f32_odd.zmm_ps);
    d_f32_even.zmm_ps = _mm512_sub_ps(a_f32_even.zmm_ps, b_f32_even.zmm_ps);

    d_f32_even.zmm = _mm512_srli_epi32(d_f32_even.zmm, 16);
    d.zmm = _mm512_mask_blend_epi16(0x55555555, d_f32_odd.zmm, d_f32_even.zmm);

    return d.zmm;
}

NK_PUBLIC void nk_l2_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_l2sq_bf16_genoa(a, b, n, result);
    *result = nk_sqrt_f32_haswell_(*result);
}
NK_PUBLIC void nk_l2sq_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 distance_sq_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32, diff_bf16x32;

nk_l2sq_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a);
        b_bf16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    diff_bf16x32 = nk_substract_bf16x32_genoa_(a_bf16x32, b_bf16x32);
    distance_sq_f32x16 = _mm512_dpbf16_ps(distance_sq_f32x16, (__m512bh)(diff_bf16x32), (__m512bh)(diff_bf16x32));
    if (n) goto nk_l2sq_bf16_genoa_cycle;

    *result = nk_reduce_add_f32x16_skylake_(distance_sq_f32x16);
}

NK_PUBLIC void nk_angular_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 dot_product_f32x16 = _mm512_setzero_ps();
    __m512 a_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512 b_norm_sq_f32x16 = _mm512_setzero_ps();
    __m512i a_bf16x32, b_bf16x32;

nk_angular_bf16_genoa_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x32 = _mm512_maskz_loadu_epi16(mask, a);
        b_bf16x32 = _mm512_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x32 = _mm512_loadu_epi16(a);
        b_bf16x32 = _mm512_loadu_epi16(b);
        a += 32, b += 32, n -= 32;
    }
    dot_product_f32x16 = _mm512_dpbf16_ps(dot_product_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(b_bf16x32));
    a_norm_sq_f32x16 = _mm512_dpbf16_ps(a_norm_sq_f32x16, (__m512bh)(a_bf16x32), (__m512bh)(a_bf16x32));
    b_norm_sq_f32x16 = _mm512_dpbf16_ps(b_norm_sq_f32x16, (__m512bh)(b_bf16x32), (__m512bh)(b_bf16x32));
    if (n) goto nk_angular_bf16_genoa_cycle;

    nk_f32_t dot_product_f32 = nk_reduce_add_f32x16_skylake_(dot_product_f32x16);
    nk_f32_t a_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(a_norm_sq_f32x16);
    nk_f32_t b_norm_sq_f32 = nk_reduce_add_f32x16_skylake_(b_norm_sq_f32x16);
    *result = nk_angular_normalize_f32_haswell_(dot_product_f32, a_norm_sq_f32, b_norm_sq_f32);
}

typedef nk_dot_bf16x32_state_genoa_t nk_angular_bf16x32_state_genoa_t;
NK_INTERNAL void nk_angular_bf16x32_init_genoa(nk_angular_bf16x32_state_genoa_t *state) {
    nk_dot_bf16x32_init_genoa(state);
}
NK_INTERNAL void nk_angular_bf16x32_update_genoa(nk_angular_bf16x32_state_genoa_t *state, nk_b512_vec_t a,
                                                 nk_b512_vec_t b) {
    nk_dot_bf16x32_update_genoa(state, a, b);
}
NK_INTERNAL void nk_angular_bf16x32_finalize_genoa(nk_angular_bf16x32_state_genoa_t const *state_a,
                                                   nk_angular_bf16x32_state_genoa_t const *state_b,
                                                   nk_angular_bf16x32_state_genoa_t const *state_c,
                                                   nk_angular_bf16x32_state_genoa_t const *state_d, nk_f32_t query_norm,
                                                   nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                   nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_bf16x32_finalize_genoa(state_a, state_b, state_c, state_d, dots);

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

typedef nk_dot_bf16x32_state_genoa_t nk_l2_bf16x32_state_genoa_t;
NK_INTERNAL void nk_l2_bf16x32_init_genoa(nk_l2_bf16x32_state_genoa_t *state) { nk_dot_bf16x32_init_genoa(state); }
NK_INTERNAL void nk_l2_bf16x32_update_genoa(nk_l2_bf16x32_state_genoa_t *state, nk_b512_vec_t a, nk_b512_vec_t b) {
    nk_dot_bf16x32_update_genoa(state, a, b);
}
NK_INTERNAL void nk_l2_bf16x32_finalize_genoa(nk_l2_bf16x32_state_genoa_t const *state_a,
                                              nk_l2_bf16x32_state_genoa_t const *state_b,
                                              nk_l2_bf16x32_state_genoa_t const *state_c,
                                              nk_l2_bf16x32_state_genoa_t const *state_d, nk_f32_t query_norm,
                                              nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                              nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract all 4 dot products with single ILP-optimized call
    nk_f32_t dots[4];
    nk_dot_bf16x32_finalize_genoa(state_a, state_b, state_c, state_d, dots);

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
#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_GENOA_H