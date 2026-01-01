/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/spatial/sapphire.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  Sapphire Rapids adds native FP16 support via AVX-512 FP16 extension.
 *  For e4m3 L2 distance, we can leverage F16 for the subtraction step:
 *  - e4m3 differences fit in F16 (max |a-b| = 896 < 65504)
 *  - But squared differences overflow F16 (896² = 802816 > 65504)
 *  - So: subtract in F16, convert to F32, then square and accumulate
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

/*  Convert 32x E4M3 values to 32x F16 values.
 *  Uses optimized path with bias adjustment.
 *  Denormals (exp=0) are flushed to zero (DAZ behavior).
 *
 *  E4M3 format: S EEEE  MMM        (bias=7)
 *  F16 format:  S EEEEE MMMMMMMMMM (bias=15)
 */
NK_INTERNAL __m512h nk_e4m3x32_to_f16x32_sapphire_(__m256i e4m3_i8x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3_i8x32);
    // Sign: bit 7 -> bit 15
    __m512i sign_i16x32 = _mm512_and_si512(_mm512_slli_epi16(e4m3_i16x32, 8), _mm512_set1_epi16((short)0x8000));
    // Exp+mant (7 bits) shifted left 7, then add bias adjustment (8<<10 = 0x2000)
    __m512i exp_mant_7bit_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));
    __m512i exp_mant_biased_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(exp_mant_7bit_i16x32, 7),
                                                      _mm512_set1_epi16(0x2000));
    // DAZ: use TEST to check if exp bits (bits 6-3) are nonzero
    __mmask32 nonzero_exp_mask = _mm512_test_epi16_mask(e4m3_i16x32, _mm512_set1_epi16(0x78));
    __m512i exp_mant_daz_i16x32 = _mm512_maskz_mov_epi16(nonzero_exp_mask, exp_mant_biased_i16x32);
    return _mm512_castsi512_ph(_mm512_or_si512(sign_i16x32, exp_mant_daz_i16x32));
}

NK_PUBLIC void nk_l2sq_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                     nk_f32_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m512 sum_f32x16 = _mm512_setzero_ps();

nk_l2sq_e4m3_sapphire_cycle:
    if (count_scalars < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, count_scalars);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a_scalars);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b_scalars);
        count_scalars = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_epi8(a_scalars);
        b_e4m3x32 = _mm256_loadu_epi8(b_scalars);
        a_scalars += 32, b_scalars += 32, count_scalars -= 32;
    }

    // Convert e4m3 -> f16
    __m512h a_f16x32 = nk_e4m3x32_to_f16x32_sapphire_(a_e4m3x32);
    __m512h b_f16x32 = nk_e4m3x32_to_f16x32_sapphire_(b_e4m3x32);

    // Subtract in F16 - differences fit (max 896 < 65504)
    __m512h diff_f16x32 = _mm512_sub_ph(a_f16x32, b_f16x32);

    // Convert to F32 before squaring (896² = 802816 overflows F16!)
    __m512 diff_lo_f32x16 = _mm512_cvtph_ps(_mm512_castsi512_si256(_mm512_castph_si512(diff_f16x32)));
    __m512 diff_hi_f32x16 = _mm512_cvtph_ps(
        _mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castph_pd(diff_f16x32), 1)));

    // Square and accumulate in F32
    sum_f32x16 = _mm512_fmadd_ps(diff_lo_f32x16, diff_lo_f32x16, sum_f32x16);
    sum_f32x16 = _mm512_fmadd_ps(diff_hi_f32x16, diff_hi_f32x16, sum_f32x16);

    if (count_scalars) goto nk_l2sq_e4m3_sapphire_cycle;

    *result = _mm512_reduce_add_ps(sum_f32x16);
}

NK_PUBLIC void nk_l2_e4m3_sapphire(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    nk_l2sq_e4m3_sapphire(a_scalars, b_scalars, count_scalars, result);
    *result = nk_sqrt_f32_haswell_(*result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#endif // NK_SPATIAL_SAPPHIRE_H
