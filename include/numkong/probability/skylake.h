/**
 *  @brief Skylake-accelerated Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/skylake.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_SKYLAKE_H
#define NK_PROBABILITY_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/spatial/skylake.h" // `nk_f32_sqrt_skylake`, `nk_f64_sqrt_skylake`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m512 nk_log2_f32_skylake_(__m512 x) {
    // Extract the exponent and mantissa
    __m512 one_f32x16 = _mm512_set1_ps(1.0f);
    __m512 exponent_f32x16 = _mm512_getexp_ps(x);
    __m512 mantissa_f32x16 = _mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512 poly_f32x16 = _mm512_set1_ps(-3.4436006e-2f);
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(3.1821337e-1f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(-1.2315303f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(2.5988452f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(-3.3241990f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(3.1157899f));

    return _mm512_add_ps(_mm512_mul_ps(poly_f32x16, _mm512_sub_ps(mantissa_f32x16, one_f32x16)), exponent_f32x16);
}

NK_PUBLIC void nk_kld_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m512 epsilon_f32x16 = _mm512_set1_ps(epsilon);
    __m512 a_f32x16, b_f32x16;

nk_kld_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, a), epsilon_f32x16);
        b_f32x16 = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, b), epsilon_f32x16);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_add_ps(_mm512_loadu_ps(a), epsilon_f32x16);
        b_f32x16 = _mm512_add_ps(_mm512_loadu_ps(b), epsilon_f32x16);
        a += 16, b += 16, n -= 16;
    }
    __m512 ratio_f32x16 = _mm512_div_ps(a_f32x16, b_f32x16);
    __m512 log_ratio_f32x16 = nk_log2_f32_skylake_(ratio_f32x16);
    __m512 contribution_f32x16 = _mm512_mul_ps(a_f32x16, log_ratio_f32x16);
    sum_f32x16 = _mm512_add_ps(sum_f32x16, contribution_f32x16);
    if (n) goto nk_kld_f32_skylake_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    *result = _mm512_reduce_add_ps(sum_f32x16) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_a_f32x16 = _mm512_setzero();
    __m512 sum_b_f32x16 = _mm512_setzero();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m512 epsilon_f32x16 = _mm512_set1_ps(epsilon);
    __m512 a_f32x16, b_f32x16;

nk_jsd_f32_skylake_cycle:
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
    __m512 mean_f32x16 = _mm512_mul_ps(_mm512_add_ps(a_f32x16, b_f32x16), _mm512_set1_ps(0.5f));
    __mmask16 nonzero_mask_a = _mm512_cmp_ps_mask(a_f32x16, epsilon_f32x16, _CMP_GE_OQ);
    __mmask16 nonzero_mask_b = _mm512_cmp_ps_mask(b_f32x16, epsilon_f32x16, _CMP_GE_OQ);
    __mmask16 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512 mean_with_epsilon_f32x16 = _mm512_add_ps(mean_f32x16, epsilon_f32x16);
    __m512 mean_recip_approx_f32x16 = _mm512_rcp14_ps(mean_with_epsilon_f32x16);
    __m512 ratio_a_f32x16 = _mm512_mul_ps(_mm512_add_ps(a_f32x16, epsilon_f32x16), mean_recip_approx_f32x16);
    __m512 ratio_b_f32x16 = _mm512_mul_ps(_mm512_add_ps(b_f32x16, epsilon_f32x16), mean_recip_approx_f32x16);
    __m512 log_ratio_a_f32x16 = nk_log2_f32_skylake_(ratio_a_f32x16);
    __m512 log_ratio_b_f32x16 = nk_log2_f32_skylake_(ratio_b_f32x16);
    sum_a_f32x16 = _mm512_mask3_fmadd_ps(a_f32x16, log_ratio_a_f32x16, sum_a_f32x16, nonzero_mask);
    sum_b_f32x16 = _mm512_mask3_fmadd_ps(b_f32x16, log_ratio_b_f32x16, sum_b_f32x16, nonzero_mask);
    if (n) goto nk_jsd_f32_skylake_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = _mm512_reduce_add_ps(_mm512_add_ps(sum_a_f32x16, sum_b_f32x16));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_haswell(sum) : 0;
}

NK_INTERNAL __m512d nk_log2_f64_skylake_(__m512d x) {
    // Extract the exponent and mantissa: x = 2^exp × m, m ∈ [1, 2)
    __m512d one_f64x8 = _mm512_set1_pd(1.0);
    __m512d two_f64x8 = _mm512_set1_pd(2.0);
    __m512d exponent_f64x8 = _mm512_getexp_pd(x);
    __m512d mantissa_f64x8 = _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute log2(m) using the s-series: s = (m-1)/(m+1), s ∈ [0, 1/3] for m ∈ [1, 2)
    // ln(m) = 2 × s × (1 + s²/3 + s⁴/5 + s⁶/7 + ...) converges fast since s² ≤ 1/9
    // log2(m) = ln(m) × log2(e)
    __m512d s_f64x8 = _mm512_div_pd(_mm512_sub_pd(mantissa_f64x8, one_f64x8), _mm512_add_pd(mantissa_f64x8, one_f64x8));
    __m512d s2_f64x8 = _mm512_mul_pd(s_f64x8, s_f64x8);

    // Polynomial P(s²) = 1 + s²/3 + s⁴/5 + ... using Horner's method
    // 14 terms (k=0..13) achieves ~1 ULP accuracy for f64
    __m512d poly_f64x8 = _mm512_set1_pd(1.0 / 27.0); // 1/(2*13+1)
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 25.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 23.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 21.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 19.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 17.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 15.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 13.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 11.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 9.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 7.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 5.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 3.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0));

    // ln(m) = 2 × s × P(s²), then log2(m) = ln(m) × log2(e)
    __m512d ln_m_f64x8 = _mm512_mul_pd(_mm512_mul_pd(two_f64x8, s_f64x8), poly_f64x8);
    __m512d log2e_f64x8 = _mm512_set1_pd(1.4426950408889634); // 1/ln(2)
    __m512d log2_m_f64x8 = _mm512_mul_pd(ln_m_f64x8, log2e_f64x8);

    // log2(x) = exponent + log2(m)
    return _mm512_add_pd(exponent_f64x8, log2_m_f64x8);
}

NK_PUBLIC void nk_kld_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_f64_t epsilon = NK_F64_DIVISION_EPSILON;
    __m512d epsilon_f64x8 = _mm512_set1_pd(epsilon);
    __m512d a_f64x8, b_f64x8;

nk_kld_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        a_f64x8 = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, a), epsilon_f64x8);
        b_f64x8 = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, b), epsilon_f64x8);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_add_pd(_mm512_loadu_pd(a), epsilon_f64x8);
        b_f64x8 = _mm512_add_pd(_mm512_loadu_pd(b), epsilon_f64x8);
        a += 8, b += 8, n -= 8;
    }
    __m512d ratio_f64x8 = _mm512_div_pd(a_f64x8, b_f64x8);
    __m512d log_ratio_f64x8 = nk_log2_f64_skylake_(ratio_f64x8);
    __m512d contribution_f64x8 = _mm512_mul_pd(a_f64x8, log_ratio_f64x8);
    sum_f64x8 = _mm512_add_pd(sum_f64x8, contribution_f64x8);
    if (n) goto nk_kld_f64_skylake_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    *result = _mm512_reduce_add_pd(sum_f64x8) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d sum_a_f64x8 = _mm512_setzero_pd();
    __m512d sum_b_f64x8 = _mm512_setzero_pd();
    nk_f64_t epsilon = NK_F64_DIVISION_EPSILON;
    __m512d epsilon_f64x8 = _mm512_set1_pd(epsilon);
    __m512d a_f64x8, b_f64x8;

nk_jsd_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d mean_f64x8 = _mm512_mul_pd(_mm512_add_pd(a_f64x8, b_f64x8), _mm512_set1_pd(0.5));
    __mmask8 nonzero_mask_a = _mm512_cmp_pd_mask(a_f64x8, epsilon_f64x8, _CMP_GE_OQ);
    __mmask8 nonzero_mask_b = _mm512_cmp_pd_mask(b_f64x8, epsilon_f64x8, _CMP_GE_OQ);
    __mmask8 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512d mean_with_epsilon_f64x8 = _mm512_add_pd(mean_f64x8, epsilon_f64x8);
    // Use full precision division (not rcp14 approximate which only has 14 bits)
    __m512d ratio_a_f64x8 = _mm512_div_pd(_mm512_add_pd(a_f64x8, epsilon_f64x8), mean_with_epsilon_f64x8);
    __m512d ratio_b_f64x8 = _mm512_div_pd(_mm512_add_pd(b_f64x8, epsilon_f64x8), mean_with_epsilon_f64x8);
    __m512d log_ratio_a_f64x8 = nk_log2_f64_skylake_(ratio_a_f64x8);
    __m512d log_ratio_b_f64x8 = nk_log2_f64_skylake_(ratio_b_f64x8);
    sum_a_f64x8 = _mm512_mask3_fmadd_pd(a_f64x8, log_ratio_a_f64x8, sum_a_f64x8, nonzero_mask);
    sum_b_f64x8 = _mm512_mask3_fmadd_pd(b_f64x8, log_ratio_b_f64x8, sum_b_f64x8, nonzero_mask);
    if (n) goto nk_jsd_f64_skylake_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    nk_f64_t sum = _mm512_reduce_add_pd(_mm512_add_pd(sum_a_f64x8, sum_b_f64x8));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_f64_sqrt_haswell(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_PROBABILITY_SKYLAKE_H
