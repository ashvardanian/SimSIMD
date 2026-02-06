/**
 *  @brief Sapphire-accelerated Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/sapphire.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_SAPPHIRE_H
#define NK_PROBABILITY_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"
#include "numkong/spatial/haswell.h"  // `nk_f32_sqrt_haswell`
#include "numkong/spatial/sapphire.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m512h nk_log2_f16_sapphire_(__m512h x) {
    // Extract the exponent and mantissa
    __m512h one_f16x32 = _mm512_set1_ph((nk_f16_t)1);
    __m512h exponent_f16x32 = _mm512_getexp_ph(x);
    __m512h mantissa_f16x32 = _mm512_getmant_ph(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512h poly_f16x32 = _mm512_set1_ph((nk_f16_t)-3.4436006e-2f);
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)3.1821337e-1f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)-1.2315303f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)2.5988452f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)-3.3241990f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)3.1157899f));

    return _mm512_fmadd_ph(poly_f16x32, _mm512_sub_ph(mantissa_f16x32, one_f16x32), exponent_f16x32);
}

NK_PUBLIC void nk_kld_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512h sum_f16x32 = _mm512_setzero_ph();
    __m512h epsilon_f16x32 = _mm512_set1_ph((nk_f16_t)NK_F16_DIVISION_EPSILON);
    __m512h a_f16x32, b_f16x32;

nk_kld_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a)), epsilon_f16x32);
        b_f16x32 = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b)), epsilon_f16x32);
        n = 0;
    }
    else {
        a_f16x32 = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(a)), epsilon_f16x32);
        b_f16x32 = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(b)), epsilon_f16x32);
        a += 32, b += 32, n -= 32;
    }
    __m512h ratio_f16x32 = _mm512_div_ph(a_f16x32, b_f16x32);
    __m512h log_ratio_f16x32 = nk_log2_f16_sapphire_(ratio_f16x32);
    __m512h contribution_f16x32 = _mm512_mul_ph(a_f16x32, log_ratio_f16x32);
    sum_f16x32 = _mm512_add_ph(sum_f16x32, contribution_f16x32);
    if (n) goto nk_kld_f16_sapphire_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    *result = _mm512_reduce_add_ph(sum_f16x32) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512h sum_a_f16x32 = _mm512_setzero_ph();
    __m512h sum_b_f16x32 = _mm512_setzero_ph();
    __m512h epsilon_f16x32 = _mm512_set1_ph((nk_f16_t)NK_F16_DIVISION_EPSILON);
    __m512h a_f16x32, b_f16x32;

nk_jsd_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    __m512h mean_f16x32 = _mm512_mul_ph(_mm512_add_ph(a_f16x32, b_f16x32), _mm512_set1_ph((nk_f16_t)0.5f));
    __mmask32 nonzero_mask_a = _mm512_cmp_ph_mask(a_f16x32, epsilon_f16x32, _CMP_GE_OQ);
    __mmask32 nonzero_mask_b = _mm512_cmp_ph_mask(b_f16x32, epsilon_f16x32, _CMP_GE_OQ);
    __mmask32 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512h mean_with_epsilon_f16x32 = _mm512_add_ph(mean_f16x32, epsilon_f16x32);
    __m512h mean_recip_approx_f16x32 = _mm512_rcp_ph(mean_with_epsilon_f16x32);
    __m512h ratio_a_f16x32 = _mm512_mul_ph(_mm512_add_ph(a_f16x32, epsilon_f16x32), mean_recip_approx_f16x32);
    __m512h ratio_b_f16x32 = _mm512_mul_ph(_mm512_add_ph(b_f16x32, epsilon_f16x32), mean_recip_approx_f16x32);
    __m512h log_ratio_a_f16x32 = nk_log2_f16_sapphire_(ratio_a_f16x32);
    __m512h log_ratio_b_f16x32 = nk_log2_f16_sapphire_(ratio_b_f16x32);
    sum_a_f16x32 = _mm512_mask3_fmadd_ph(a_f16x32, log_ratio_a_f16x32, sum_a_f16x32, nonzero_mask);
    sum_b_f16x32 = _mm512_mask3_fmadd_ph(b_f16x32, log_ratio_b_f16x32, sum_b_f16x32, nonzero_mask);
    if (n) goto nk_jsd_f16_sapphire_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = _mm512_reduce_add_ph(_mm512_add_ph(sum_a_f16x32, sum_b_f16x32));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_haswell(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_
#endif // NK_PROBABILITY_SAPPHIRE_H
