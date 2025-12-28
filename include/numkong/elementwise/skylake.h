/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/elementwise/skylake.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_SKYLAKE_H
#define NK_ELEMENTWISE_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sum_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d a_vec, b_vec, sum_vec;
    __mmask8 mask = 0xFF;
nk_sum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    sum_vec = _mm512_add_pd(a_vec, b_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto nk_sum_f64_skylake_cycle;
}

NK_PUBLIC void nk_scale_f64_skylake(nk_f64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_scale_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        a += 8, n -= 8;
    }
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_scale_f64_skylake_cycle;
}

NK_PUBLIC void nk_wsum_f64_skylake(                    //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f64_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f64_t zero = 0;
        if (beta_val == 0) { nk_scale_f64_skylake(a, n, alpha, &zero, result); }
        else { nk_scale_f64_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, b_f64x8, a_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_wsum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    a_scaled_f64x8 = _mm512_mul_pd(a_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(b_f64x8, beta_f64x8, a_scaled_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_wsum_f64_skylake_cycle;
}

NK_PUBLIC void nk_sum_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

nk_sum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto nk_sum_f32_skylake_cycle;
}

NK_PUBLIC void nk_scale_f32_skylake(nk_f32_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;

nk_scale_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        a += 16, n -= 16;
    }
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_scale_f32_skylake_cycle;
}

NK_PUBLIC void nk_wsum_f32_skylake(                    //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f32_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f32_skylake(a, n, alpha, &zero, result); }
        else { nk_scale_f32_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_wsum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_wsum_f32_skylake_cycle;
}

NK_PUBLIC void nk_sum_bf16_skylake(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_bf16_t *result) {
    __m256i a_bf16_vec, b_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
nk_sum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16_vec = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        b_bf16_vec = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_vec = nk_bf16x16_to_f32x16_skylake_(a_bf16_vec);
    b_vec = nk_bf16x16_to_f32x16_skylake_(b_bf16_vec);
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    sum_bf16_vec = nk_f32x16_to_bf16x16_skylake_(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto nk_sum_bf16_skylake_cycle;
}

NK_PUBLIC void nk_scale_bf16_skylake(nk_bf16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, result_bf16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_scale_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        a += 16, n -= 16;
    }
    a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_bf16x16 = nk_f32x16_to_bf16x16_skylake_(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_scale_bf16_skylake_cycle;
}

NK_PUBLIC void nk_wsum_bf16_skylake(                     //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_bf16_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_bf16_skylake(a, n, alpha, &zero, result); }
        else { nk_scale_bf16_skylake(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, b_bf16x16, result_bf16x16;
    __m512 a_f32x16, b_f32x16, a_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_wsum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        b_bf16x16 = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
    a_scaled_f32x16 = _mm512_mul_ps(a_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(b_f32x16, beta_f32x16, a_scaled_f32x16);
    result_bf16x16 = nk_f32x16_to_bf16x16_skylake_(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_wsum_bf16_skylake_cycle;
}

NK_PUBLIC void nk_fma_f64_skylake(                                        //
    nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_f64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_fma_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        c_f64x8 = _mm512_maskz_loadu_pd(mask, c);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        c_f64x8 = _mm512_loadu_pd(c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    _mm512_mask_storeu_pd(result, mask, result_f64x8);
    result += 8;
    if (n) goto nk_fma_f64_skylake_cycle;
}

NK_PUBLIC void nk_fma_f32_skylake(                                        //
    nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f32_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_fma_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        c_f32x16 = _mm512_maskz_loadu_ps(mask, c);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        c_f32x16 = _mm512_loadu_ps(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    _mm512_mask_storeu_ps(result, mask, result_f32x16);
    result += 16;
    if (n) goto nk_fma_f32_skylake_cycle;
}

NK_PUBLIC void nk_fma_bf16_skylake(                                          //
    nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_bf16_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m256i a_bf16x16, b_bf16x16, c_bf16x16, result_bf16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
nk_fma_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_bf16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_bf16x16 = _mm256_loadu_epi16(a);
        b_bf16x16 = _mm256_loadu_epi16(b);
        c_bf16x16 = _mm256_loadu_epi16(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = nk_bf16x16_to_f32x16_skylake_(a_bf16x16);
    b_f32x16 = nk_bf16x16_to_f32x16_skylake_(b_bf16x16);
    c_f32x16 = nk_bf16x16_to_f32x16_skylake_(c_bf16x16);
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_bf16x16 = nk_f32x16_to_bf16x16_skylake_(result_f32x16);
    _mm256_mask_storeu_epi16(result, mask, result_bf16x16);
    result += 16;
    if (n) goto nk_fma_bf16_skylake_cycle;
}

NK_PUBLIC void nk_scale_i8_skylake(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_i8x16, result_i8x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-128);
    __m512i max_i32x16 = _mm512_set1_epi32(127);

nk_scale_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_i8x16 = _mm_lddqu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i8x16 = _mm512_cvtepi32_epi8(result_i32x16);
    _mm_mask_storeu_epi8(result, mask, result_i8x16);
    result += 16;
    if (n) goto nk_scale_i8_skylake_cycle;
}

NK_PUBLIC void nk_fma_i8_skylake(                                      //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_i8x16, b_i8x16, c_i8x16, result_i8x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-128);
    __m512i max_i32x16 = _mm512_set1_epi32(127);

nk_fma_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8x16 = _mm_maskz_loadu_epi8(mask, a);
        b_i8x16 = _mm_maskz_loadu_epi8(mask, b);
        c_i8x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_i8x16 = _mm_lddqu_si128((__m128i *)a);
        b_i8x16 = _mm_lddqu_si128((__m128i *)b);
        c_i8x16 = _mm_lddqu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_i8x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c_i8x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i8x16 = _mm512_cvtepi32_epi8(result_i32x16);
    _mm_mask_storeu_epi8(result, mask, result_i8x16);
    result += 16;
    if (n) goto nk_fma_i8_skylake_cycle;
}

NK_PUBLIC void nk_scale_u8_skylake(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                   nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_u8x16, result_u8x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(255);

nk_scale_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8x16 = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8x16 = _mm_lddqu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u8x16 = _mm512_cvtepi32_epi8(result_u32x16);
    _mm_mask_storeu_epi8(result, mask, result_u8x16);
    result += 16;
    if (n) goto nk_scale_u8_skylake_cycle;
}

NK_PUBLIC void nk_fma_u8_skylake(                                      //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {
    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_val);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_val);
    __m128i a_u8x16, b_u8x16, c_u8x16, result_u8x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(255);

nk_fma_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8x16 = _mm_maskz_loadu_epi8(mask, a);
        b_u8x16 = _mm_maskz_loadu_epi8(mask, b);
        c_u8x16 = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8x16 = _mm_lddqu_si128((__m128i *)a);
        b_u8x16 = _mm_lddqu_si128((__m128i *)b);
        c_u8x16 = _mm_lddqu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_u8x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c_u8x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u8x16 = _mm512_cvtepi32_epi8(result_u32x16);
    _mm_mask_storeu_epi8(result, mask, result_u8x16);
    result += 16;
    if (n) goto nk_fma_u8_skylake_cycle;
}

NK_PUBLIC void nk_scale_i16_skylake(nk_i16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_i16x16, result_i16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-32768);
    __m512i max_i32x16 = _mm512_set1_epi32(32767);

nk_scale_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_i16x16 = _mm256_lddqu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i16x16 = _mm512_cvtepi32_epi16(result_i32x16);
    _mm256_mask_storeu_epi16(result, mask, result_i16x16);
    result += 16;
    if (n) goto nk_scale_i16_skylake_cycle;
}

NK_PUBLIC void nk_fma_i16_skylake(                                        //
    nk_i16_t const *a, nk_i16_t const *b, nk_i16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_i16x16, b_i16x16, c_i16x16, result_i16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_i32x16;
    __m512i min_i32x16 = _mm512_set1_epi32(-32768);
    __m512i max_i32x16 = _mm512_set1_epi32(32767);

nk_fma_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_i16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_i16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_i16x16 = _mm256_lddqu_si256((__m256i *)a);
        b_i16x16 = _mm256_lddqu_si256((__m256i *)b);
        c_i16x16 = _mm256_lddqu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(b_i16x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c_i16x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_i32x16 = _mm512_cvtps_epi32(result_f32x16);
    result_i32x16 = _mm512_max_epi32(result_i32x16, min_i32x16);
    result_i32x16 = _mm512_min_epi32(result_i32x16, max_i32x16);
    result_i16x16 = _mm512_cvtepi32_epi16(result_i32x16);
    _mm256_mask_storeu_epi16(result, mask, result_i16x16);
    result += 16;
    if (n) goto nk_fma_i16_skylake_cycle;
}

NK_PUBLIC void nk_scale_u16_skylake(nk_u16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_u16x16, result_u16x16;
    __m512 a_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(65535);

nk_scale_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16x16 = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_u16x16 = _mm256_lddqu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16x16));
    result_f32x16 = _mm512_fmadd_ps(a_f32x16, alpha_f32x16, beta_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u16x16 = _mm512_cvtepi32_epi16(result_u32x16);
    _mm256_mask_storeu_epi16(result, mask, result_u16x16);
    result += 16;
    if (n) goto nk_scale_u16_skylake_cycle;
}

NK_PUBLIC void nk_fma_u16_skylake(                                        //
    nk_u16_t const *a, nk_u16_t const *b, nk_u16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u16_t *result) {
    nk_f32_t alpha_f32 = *alpha;
    nk_f32_t beta_f32 = *beta;
    __m512 alpha_f32x16 = _mm512_set1_ps(alpha_f32);
    __m512 beta_f32x16 = _mm512_set1_ps(beta_f32);
    __m256i a_u16x16, b_u16x16, c_u16x16, result_u16x16;
    __m512 a_f32x16, b_f32x16, c_f32x16, ab_f32x16, ab_scaled_f32x16, result_f32x16;
    __mmask16 mask = 0xFFFF;
    __m512i result_u32x16;
    __m512i min_u32x16 = _mm512_set1_epi32(0);
    __m512i max_u32x16 = _mm512_set1_epi32(65535);

nk_fma_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16x16 = _mm256_maskz_loadu_epi16(mask, a);
        b_u16x16 = _mm256_maskz_loadu_epi16(mask, b);
        c_u16x16 = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_u16x16 = _mm256_lddqu_si256((__m256i *)a);
        b_u16x16 = _mm256_lddqu_si256((__m256i *)b);
        c_u16x16 = _mm256_lddqu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16x16));
    b_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(b_u16x16));
    c_f32x16 = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(c_u16x16));
    ab_f32x16 = _mm512_mul_ps(a_f32x16, b_f32x16);
    ab_scaled_f32x16 = _mm512_mul_ps(ab_f32x16, alpha_f32x16);
    result_f32x16 = _mm512_fmadd_ps(c_f32x16, beta_f32x16, ab_scaled_f32x16);
    result_u32x16 = _mm512_cvtps_epu32(result_f32x16);
    result_u32x16 = _mm512_max_epu32(result_u32x16, min_u32x16);
    result_u32x16 = _mm512_min_epu32(result_u32x16, max_u32x16);
    result_u16x16 = _mm512_cvtepi32_epi16(result_u32x16);
    _mm256_mask_storeu_epi16(result, mask, result_u16x16);
    result += 16;
    if (n) goto nk_fma_u16_skylake_cycle;
}

NK_PUBLIC void nk_scale_i32_skylake(nk_i32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_i32x8, result_i32x8;
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(-2147483648.0);
    __m512d max_f64x8 = _mm512_set1_pd(2147483647.0);

nk_scale_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32x8 = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_i32x8 = _mm256_lddqu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi32_pd(a_i32x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_i32x8 = _mm512_cvttpd_epi32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_i32x8);
    result += 8;
    if (n) goto nk_scale_i32_skylake_cycle;
}

NK_PUBLIC void nk_fma_i32_skylake(                                        //
    nk_i32_t const *a, nk_i32_t const *b, nk_i32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_i32x8, b_i32x8, c_i32x8, result_i32x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(-2147483648.0);
    __m512d max_f64x8 = _mm512_set1_pd(2147483647.0);

nk_fma_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32x8 = _mm256_maskz_loadu_epi32(mask, a);
        b_i32x8 = _mm256_maskz_loadu_epi32(mask, b);
        c_i32x8 = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_i32x8 = _mm256_lddqu_si256((__m256i *)a);
        b_i32x8 = _mm256_lddqu_si256((__m256i *)b);
        c_i32x8 = _mm256_lddqu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi32_pd(a_i32x8);
    b_f64x8 = _mm512_cvtepi32_pd(b_i32x8);
    c_f64x8 = _mm512_cvtepi32_pd(c_i32x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_i32x8 = _mm512_cvttpd_epi32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_i32x8);
    result += 8;
    if (n) goto nk_fma_i32_skylake_cycle;
}

NK_PUBLIC void nk_scale_u32_skylake(nk_u32_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_u32x8, result_u32x8;
    __m512d a_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(0.0);
    __m512d max_f64x8 = _mm512_set1_pd(4294967295.0);

nk_scale_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32x8 = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_u32x8 = _mm256_lddqu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu32_pd(a_u32x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_u32x8 = _mm512_cvttpd_epu32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_u32x8);
    result += 8;
    if (n) goto nk_scale_u32_skylake_cycle;
}

NK_PUBLIC void nk_fma_u32_skylake(                                        //
    nk_u32_t const *a, nk_u32_t const *b, nk_u32_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u32_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m256i a_u32x8, b_u32x8, c_u32x8, result_u32x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
    __m512d min_f64x8 = _mm512_set1_pd(0.0);
    __m512d max_f64x8 = _mm512_set1_pd(4294967295.0);

nk_fma_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32x8 = _mm256_maskz_loadu_epi32(mask, a);
        b_u32x8 = _mm256_maskz_loadu_epi32(mask, b);
        c_u32x8 = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_u32x8 = _mm256_lddqu_si256((__m256i *)a);
        b_u32x8 = _mm256_lddqu_si256((__m256i *)b);
        c_u32x8 = _mm256_lddqu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu32_pd(a_u32x8);
    b_f64x8 = _mm512_cvtepu32_pd(b_u32x8);
    c_f64x8 = _mm512_cvtepu32_pd(c_u32x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_f64x8 = _mm512_max_pd(result_f64x8, min_f64x8);
    result_f64x8 = _mm512_min_pd(result_f64x8, max_f64x8);
    result_u32x8 = _mm512_cvttpd_epu32(result_f64x8);
    _mm256_mask_storeu_epi32(result, mask, result_u32x8);
    result += 8;
    if (n) goto nk_fma_u32_skylake_cycle;
}

NK_PUBLIC void nk_scale_i64_skylake(nk_i64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_i64x8;
    __m512d a_f64x8, result_f64x8;
    __m512i result_i64x8;
    __mmask8 mask = 0xFF;

nk_scale_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64x8 = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_i64x8 = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi64_pd(a_i64x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_i64x8 = _mm512_cvtpd_epi64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_i64x8);
    result += 8;
    if (n) goto nk_scale_i64_skylake_cycle;
}

NK_PUBLIC void nk_fma_i64_skylake(                                        //
    nk_i64_t const *a, nk_i64_t const *b, nk_i64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_i64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_i64x8, b_i64x8, c_i64x8, result_i64x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_fma_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64x8 = _mm512_maskz_loadu_epi64(mask, a);
        b_i64x8 = _mm512_maskz_loadu_epi64(mask, b);
        c_i64x8 = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_i64x8 = _mm512_loadu_si512((__m512i *)a);
        b_i64x8 = _mm512_loadu_si512((__m512i *)b);
        c_i64x8 = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepi64_pd(a_i64x8);
    b_f64x8 = _mm512_cvtepi64_pd(b_i64x8);
    c_f64x8 = _mm512_cvtepi64_pd(c_i64x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_i64x8 = _mm512_cvtpd_epi64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_i64x8);
    result += 8;
    if (n) goto nk_fma_i64_skylake_cycle;
}

NK_PUBLIC void nk_scale_u64_skylake(nk_u64_t const *a, nk_size_t n, nk_f64_t const *alpha, nk_f64_t const *beta,
                                    nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_u64x8;
    __m512d a_f64x8, result_f64x8;
    __m512i result_u64x8;
    __mmask8 mask = 0xFF;

nk_scale_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64x8 = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_u64x8 = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu64_pd(a_u64x8);
    result_f64x8 = _mm512_fmadd_pd(a_f64x8, alpha_f64x8, beta_f64x8);
    result_u64x8 = _mm512_cvtpd_epu64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_u64x8);
    result += 8;
    if (n) goto nk_scale_u64_skylake_cycle;
}

NK_PUBLIC void nk_fma_u64_skylake(                                        //
    nk_u64_t const *a, nk_u64_t const *b, nk_u64_t const *c, nk_size_t n, //
    nk_f64_t const *alpha, nk_f64_t const *beta, nk_u64_t *result) {
    nk_f64_t alpha_val = *alpha;
    nk_f64_t beta_val = *beta;
    __m512d alpha_f64x8 = _mm512_set1_pd(alpha_val);
    __m512d beta_f64x8 = _mm512_set1_pd(beta_val);
    __m512i a_u64x8, b_u64x8, c_u64x8, result_u64x8;
    __m512d a_f64x8, b_f64x8, c_f64x8, ab_f64x8, ab_scaled_f64x8, result_f64x8;
    __mmask8 mask = 0xFF;
nk_fma_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64x8 = _mm512_maskz_loadu_epi64(mask, a);
        b_u64x8 = _mm512_maskz_loadu_epi64(mask, b);
        c_u64x8 = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_u64x8 = _mm512_loadu_si512((__m512i *)a);
        b_u64x8 = _mm512_loadu_si512((__m512i *)b);
        c_u64x8 = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_f64x8 = _mm512_cvtepu64_pd(a_u64x8);
    b_f64x8 = _mm512_cvtepu64_pd(b_u64x8);
    c_f64x8 = _mm512_cvtepu64_pd(c_u64x8);
    ab_f64x8 = _mm512_mul_pd(a_f64x8, b_f64x8);
    ab_scaled_f64x8 = _mm512_mul_pd(ab_f64x8, alpha_f64x8);
    result_f64x8 = _mm512_fmadd_pd(c_f64x8, beta_f64x8, ab_scaled_f64x8);
    result_u64x8 = _mm512_cvtpd_epu64(result_f64x8);
    _mm512_mask_storeu_epi64(result, mask, result_u64x8);
    result += 8;
    if (n) goto nk_fma_u64_skylake_cycle;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_ELEMENTWISE_SKYLAKE_H