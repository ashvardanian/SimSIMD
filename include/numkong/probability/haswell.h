/**
 *  @brief Haswell-accelerated Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/haswell.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_HASWELL_H
#define NK_PROBABILITY_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_f16x8_to_f32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m256 nk_log2_f32_haswell_(__m256 x) {
    // Extracting the exponent
    __m256i bits_i32x8 = _mm256_castps_si256(x);
    __m256i exponent_i32x8 = _mm256_srli_epi32(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x7F800000)), 23);
    exponent_i32x8 = _mm256_sub_epi32(exponent_i32x8, _mm256_set1_epi32(127)); // removing the bias
    __m256 exponent_f32x8 = _mm256_cvtepi32_ps(exponent_i32x8);

    // Extracting the mantissa
    __m256 mantissa_f32x8 = _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000)));

    // Constants for polynomial
    __m256 one_f32x8 = _mm256_set1_ps(1.0f);
    __m256 poly_f32x8 = _mm256_set1_ps(-3.4436006e-2f);

    // Compute the polynomial using Horner's method
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(3.1821337e-1f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(-1.2315303f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(2.5988452f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(-3.3241990f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(3.1157899f));

    // Final computation
    __m256 result_f32x8 = _mm256_add_ps(_mm256_mul_ps(poly_f32x8, _mm256_sub_ps(mantissa_f32x8, one_f32x8)),
                                        exponent_f32x8);
    return result_f32x8;
}

NK_PUBLIC void nk_kld_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m256 epsilon_f32x8 = _mm256_set1_ps(epsilon);
    __m256 a_f32x8, b_f32x8;

nk_kld_f16_haswell_cycle:
    if (n < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_f16x8_to_f32x8_haswell_(a, &a_vec, n);
        nk_partial_load_f16x8_to_f32x8_haswell_(b, &b_vec, n);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    a_f32x8 = _mm256_add_ps(a_f32x8, epsilon_f32x8);
    b_f32x8 = _mm256_add_ps(b_f32x8, epsilon_f32x8);
    __m256 ratio_f32x8 = _mm256_div_ps(a_f32x8, b_f32x8);
    __m256 log_ratio_f32x8 = nk_log2_f32_haswell_(ratio_f32x8);
    __m256 contribution_f32x8 = _mm256_mul_ps(a_f32x8, log_ratio_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_f32x8);
    if (n) goto nk_kld_f16_haswell_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    sum *= log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m256 epsilon_f32x8 = _mm256_set1_ps(epsilon);
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 a_f32x8, b_f32x8;

nk_jsd_f16_haswell_cycle:
    if (n < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_f16x8_to_f32x8_haswell_(a, &a_vec, n);
        nk_partial_load_f16x8_to_f32x8_haswell_(b, &b_vec, n);
        a_f32x8 = a_vec.ymm_ps;
        b_f32x8 = b_vec.ymm_ps;
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 mean_f32x8 = _mm256_mul_ps(_mm256_add_ps(a_f32x8, b_f32x8), _mm256_set1_ps(0.5f)); // M = (P + Q) / 2
    __m256 ratio_a_f32x8 = _mm256_div_ps(_mm256_add_ps(a_f32x8, epsilon_f32x8),
                                         _mm256_add_ps(mean_f32x8, epsilon_f32x8));
    __m256 ratio_b_f32x8 = _mm256_div_ps(_mm256_add_ps(b_f32x8, epsilon_f32x8),
                                         _mm256_add_ps(mean_f32x8, epsilon_f32x8));
    __m256 log_ratio_a_f32x8 = nk_log2_f32_haswell_(ratio_a_f32x8);
    __m256 log_ratio_b_f32x8 = nk_log2_f32_haswell_(ratio_b_f32x8);
    __m256 contribution_a_f32x8 = _mm256_mul_ps(a_f32x8, log_ratio_a_f32x8);
    __m256 contribution_b_f32x8 = _mm256_mul_ps(b_f32x8, log_ratio_b_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_a_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_b_f32x8);
    if (n) goto nk_jsd_f16_haswell_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_haswell(sum) : 0;
}

NK_PUBLIC void nk_kld_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_kld_f64_serial(a, b, n, result);
}

NK_PUBLIC void nk_jsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_jsd_f64_serial(a, b, n, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_PROBABILITY_HASWELL_H
