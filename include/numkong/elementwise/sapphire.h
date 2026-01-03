/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/elementwise/sapphire.h
 *  @sa include/numkong/elementwise.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_ELEMENTWISE_SAPPHIRE_H
#define NK_ELEMENTWISE_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

#include "numkong/types.h"
#include "numkong/cast/sapphire.h" // nk_f32_to_f16_sapphire, nk_e4m3x16_to_f16x16_sapphire_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_sum_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512h a_f16_vec, b_f16_vec;
    __m512h sum_f16_vec;
nk_sum_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_loadu_ph(a);
        b_f16_vec = _mm512_loadu_ph(b);
        a += 32, b += 32, n -= 32;
    }
    sum_f16_vec = _mm512_add_ph(a_f16_vec, b_f16_vec);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto nk_sum_f16_sapphire_cycle;
}

NK_PUBLIC void nk_scale_f16_sapphire(nk_f16_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                     nk_f16_t *result) {
    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(alpha, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(beta, (nk_f16_t *)&beta_short);
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m512h a_f16x32;
    __m512h result_f16x32;
nk_scale_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_ph(a);
        a += 32, n -= 32;
    }
    result_f16x32 = _mm512_fmadd_ph(a_f16x32, alpha_f16x32, beta_f16x32);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(result_f16x32));
    result += 32;
    if (n) goto nk_scale_f16_sapphire_cycle;
}

NK_PUBLIC void nk_wsum_f16_sapphire(                   //
    nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_f16_sapphire(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_f16_sapphire(a, n, alpha, &zero, result); }
        else { nk_scale_f16_sapphire(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(&alpha_val, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(&beta_val, (nk_f16_t *)&beta_short);
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m512h a_f16x32, b_f16x32;
    __m512h a_scaled_f16x32, result_f16x32;
nk_wsum_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_ph(a);
        b_f16x32 = _mm512_loadu_ph(b);
        a += 32, b += 32, n -= 32;
    }
    a_scaled_f16x32 = _mm512_mul_ph(a_f16x32, alpha_f16x32);
    result_f16x32 = _mm512_fmadd_ph(b_f16x32, beta_f16x32, a_scaled_f16x32);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(result_f16x32));
    result += 32;
    if (n) goto nk_wsum_f16_sapphire_cycle;
}

NK_PUBLIC void nk_fma_f16_sapphire(                                       //
    nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_f16_t *result) {

    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(alpha, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(beta, (nk_f16_t *)&beta_short);
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m512h a_f16x32, b_f16x32, c_f16x32;
    __m512h ab_f16x32, ab_scaled_f16x32, result_f16x32;
nk_fma_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        c_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_loadu_ph(a);
        b_f16x32 = _mm512_loadu_ph(b);
        c_f16x32 = _mm512_loadu_ph(c);
        a += 32, b += 32, c += 32, n -= 32;
    }
    ab_f16x32 = _mm512_mul_ph(a_f16x32, b_f16x32);
    ab_scaled_f16x32 = _mm512_mul_ph(ab_f16x32, alpha_f16x32);
    result_f16x32 = _mm512_fmadd_ph(c_f16x32, beta_f16x32, ab_scaled_f16x32);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(result_f16x32));
    result += 32;
    if (n) goto nk_fma_f16_sapphire_cycle;
}

NK_PUBLIC void nk_scale_u8_sapphire(nk_u8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_u8_t *result) {
    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(alpha, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(beta, (nk_f16_t *)&beta_short);
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m512i a_u8x64, result_u8x64;
    __m512h a_low_f16x32, a_high_f16x32;
    __m512h result_low_f16x32, result_high_f16x32;
    __m512i result_low_i16x32, result_high_i16x32;
nk_scale_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        a += 64, n -= 64;
    }
    // Upcast:
    a_low_f16x32 = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8x64, _mm512_setzero_si512()));
    a_high_f16x32 = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8x64, _mm512_setzero_si512()));
    // Scale:
    result_low_f16x32 = _mm512_fmadd_ph(a_low_f16x32, alpha_f16x32, beta_f16x32);
    result_high_f16x32 = _mm512_fmadd_ph(a_high_f16x32, alpha_f16x32, beta_f16x32);
    // Downcast:
    result_low_i16x32 = _mm512_cvtph_epi16(result_low_f16x32);
    result_high_i16x32 = _mm512_cvtph_epi16(result_high_f16x32);
    result_u8x64 = _mm512_packus_epi16(result_low_i16x32, result_high_i16x32);
    _mm512_mask_storeu_epi8(result, mask, result_u8x64);
    result += 64;
    if (n) goto nk_scale_u8_sapphire_cycle;
}

NK_PUBLIC void nk_wsum_u8_sapphire(                  //
    nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_u8_ice(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_u8_sapphire(a, n, alpha, &zero, result); }
        else { nk_scale_u8_sapphire(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(&alpha_val, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(&beta_val, (nk_f16_t *)&beta_short);
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m512i a_u8x64, b_u8x64, result_u8x64;
    __m512h a_low_f16x32, a_high_f16x32, b_low_f16x32, b_high_f16x32;
    __m512h a_scaled_low_f16x32, a_scaled_high_f16x32, result_low_f16x32, result_high_f16x32;
    __m512i result_low_i16x32, result_high_i16x32;
nk_wsum_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        b_u8x64 = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    // Upcast:
    a_low_f16x32 = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8x64, _mm512_setzero_si512()));
    a_high_f16x32 = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8x64, _mm512_setzero_si512()));
    b_low_f16x32 = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(b_u8x64, _mm512_setzero_si512()));
    b_high_f16x32 = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(b_u8x64, _mm512_setzero_si512()));
    // Scale:
    a_scaled_low_f16x32 = _mm512_mul_ph(a_low_f16x32, alpha_f16x32);
    a_scaled_high_f16x32 = _mm512_mul_ph(a_high_f16x32, alpha_f16x32);
    // Add:
    result_low_f16x32 = _mm512_fmadd_ph(b_low_f16x32, beta_f16x32, a_scaled_low_f16x32);
    result_high_f16x32 = _mm512_fmadd_ph(b_high_f16x32, beta_f16x32, a_scaled_high_f16x32);
    // Downcast:
    result_low_i16x32 = _mm512_cvtph_epi16(result_low_f16x32);
    result_high_i16x32 = _mm512_cvtph_epi16(result_high_f16x32);
    result_u8x64 = _mm512_packus_epi16(result_low_i16x32, result_high_i16x32);
    _mm512_mask_storeu_epi8(result, mask, result_u8x64);
    result += 64;
    if (n) goto nk_wsum_u8_sapphire_cycle;
}

NK_PUBLIC void nk_scale_i8_sapphire(nk_i8_t const *a, nk_size_t n, nk_f32_t const *alpha, nk_f32_t const *beta,
                                    nk_i8_t *result) {
    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(alpha, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(beta, (nk_f16_t *)&beta_short);
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m256i a_low_i8x32, a_high_i8x32;
    __m512i result_i8x64;
    __m512h a_low_f16x32, a_high_f16x32;
    __m512h result_low_f16x32, result_high_f16x32;
    __m512i result_low_i16x32, result_high_i16x32;
nk_scale_i8_sapphire_cycle:
    if (n < 64) {
        // Tail: use masked 512-bit load and extract (runs once)
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        __m512i a_i8x64 = _mm512_maskz_loadu_epi8(mask, a);
        a_low_i8x32 = _mm512_castsi512_si256(a_i8x64);
        a_high_i8x32 = _mm512_extracti64x4_epi64(a_i8x64, 1);
        n = 0;
    }
    else {
        // Hot path: 2×256-bit loads to avoid VEXTRACTI64X4 (Port 5)
        a_low_i8x32 = _mm256_loadu_epi8(a);
        a_high_i8x32 = _mm256_loadu_epi8(a + 32);
        a += 64, n -= 64;
    }
    // Upcast from 256-bit halves:
    a_low_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(a_low_i8x32));
    a_high_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(a_high_i8x32));
    // Scale:
    result_low_f16x32 = _mm512_fmadd_ph(a_low_f16x32, alpha_f16x32, beta_f16x32);
    result_high_f16x32 = _mm512_fmadd_ph(a_high_f16x32, alpha_f16x32, beta_f16x32);
    // Downcast:
    result_low_i16x32 = _mm512_cvtph_epi16(result_low_f16x32);
    result_high_i16x32 = _mm512_cvtph_epi16(result_high_f16x32);
    result_i8x64 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(result_low_i16x32)),
                                      _mm512_cvtsepi16_epi8(result_high_i16x32), 1);
    _mm512_mask_storeu_epi8(result, mask, result_i8x64);
    result += 64;
    if (n) goto nk_scale_i8_sapphire_cycle;
}

NK_PUBLIC void nk_wsum_i8_sapphire(                  //
    nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    nk_f32_t alpha_val = *alpha;
    nk_f32_t beta_val = *beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha_val == 1 && beta_val == 1) {
        // In this case we can avoid expensive multiplications.
        nk_sum_i8_ice(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha_val == 0 || beta_val == 0) {
        // In this case we can avoid half of the load instructions.
        nk_f32_t zero = 0;
        if (beta_val == 0) { nk_scale_i8_sapphire(a, n, alpha, &zero, result); }
        else { nk_scale_i8_sapphire(b, n, beta, &zero, result); }
        return;
    }

    // The general case.
    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(&alpha_val, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(&beta_val, (nk_f16_t *)&beta_short);
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m256i a_low_i8x32, a_high_i8x32, b_low_i8x32, b_high_i8x32;
    __m512i result_i8x64;
    __m512h a_low_f16x32, a_high_f16x32, b_low_f16x32, b_high_f16x32;
    __m512h a_scaled_low_f16x32, a_scaled_high_f16x32, result_low_f16x32, result_high_f16x32;
    __m512i result_low_i16x32, result_high_i16x32;
nk_wsum_i8_sapphire_cycle:
    if (n < 64) {
        // Tail: use masked 512-bit loads and extract (runs once)
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        __m512i a_i8x64 = _mm512_maskz_loadu_epi8(mask, a);
        __m512i b_i8x64 = _mm512_maskz_loadu_epi8(mask, b);
        a_low_i8x32 = _mm512_castsi512_si256(a_i8x64);
        a_high_i8x32 = _mm512_extracti64x4_epi64(a_i8x64, 1);
        b_low_i8x32 = _mm512_castsi512_si256(b_i8x64);
        b_high_i8x32 = _mm512_extracti64x4_epi64(b_i8x64, 1);
        n = 0;
    }
    else {
        // Hot path: 2×256-bit loads per vector to avoid VEXTRACTI64X4 (Port 5)
        a_low_i8x32 = _mm256_loadu_epi8(a);
        a_high_i8x32 = _mm256_loadu_epi8(a + 32);
        b_low_i8x32 = _mm256_loadu_epi8(b);
        b_high_i8x32 = _mm256_loadu_epi8(b + 32);
        a += 64, b += 64, n -= 64;
    }
    // Upcast from 256-bit halves:
    a_low_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(a_low_i8x32));
    a_high_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(a_high_i8x32));
    b_low_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(b_low_i8x32));
    b_high_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(b_high_i8x32));
    // Scale:
    a_scaled_low_f16x32 = _mm512_mul_ph(a_low_f16x32, alpha_f16x32);
    a_scaled_high_f16x32 = _mm512_mul_ph(a_high_f16x32, alpha_f16x32);
    // Add:
    result_low_f16x32 = _mm512_fmadd_ph(b_low_f16x32, beta_f16x32, a_scaled_low_f16x32);
    result_high_f16x32 = _mm512_fmadd_ph(b_high_f16x32, beta_f16x32, a_scaled_high_f16x32);
    // Downcast:
    result_low_i16x32 = _mm512_cvtph_epi16(result_low_f16x32);
    result_high_i16x32 = _mm512_cvtph_epi16(result_high_f16x32);
    result_i8x64 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(result_low_i16x32)),
                                      _mm512_cvtsepi16_epi8(result_high_i16x32), 1);
    _mm512_mask_storeu_epi8(result, mask, result_i8x64);
    result += 64;
    if (n) goto nk_wsum_i8_sapphire_cycle;
}

NK_PUBLIC void nk_fma_i8_sapphire(                                     //
    nk_i8_t const *a, nk_i8_t const *b, nk_i8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_i8_t *result) {

    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(alpha, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(beta, (nk_f16_t *)&beta_short);
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m256i a_low_i8x32, a_high_i8x32, b_low_i8x32, b_high_i8x32, c_low_i8x32, c_high_i8x32;
    __m512i result_i8x64;
    __m512h a_low_f16x32, a_high_f16x32, b_low_f16x32, b_high_f16x32;
    __m512h c_low_f16x32, c_high_f16x32, ab_low_f16x32, ab_high_f16x32;
    __m512h ab_scaled_low_f16x32, ab_scaled_high_f16x32, result_low_f16x32, result_high_f16x32;
    __m512i result_low_i16x32, result_high_i16x32;
    __m512h min_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(-128));
    __m512h max_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(127));

nk_fma_i8_sapphire_cycle:
    if (n < 64) {
        // Tail: use masked 512-bit loads and extract (runs once)
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        __m512i a_i8x64 = _mm512_maskz_loadu_epi8(mask, a);
        __m512i b_i8x64 = _mm512_maskz_loadu_epi8(mask, b);
        __m512i c_i8x64 = _mm512_maskz_loadu_epi8(mask, c);
        a_low_i8x32 = _mm512_castsi512_si256(a_i8x64);
        a_high_i8x32 = _mm512_extracti64x4_epi64(a_i8x64, 1);
        b_low_i8x32 = _mm512_castsi512_si256(b_i8x64);
        b_high_i8x32 = _mm512_extracti64x4_epi64(b_i8x64, 1);
        c_low_i8x32 = _mm512_castsi512_si256(c_i8x64);
        c_high_i8x32 = _mm512_extracti64x4_epi64(c_i8x64, 1);
        n = 0;
    }
    else {
        // Hot path: 2×256-bit loads per vector to avoid VEXTRACTI64X4 (Port 5)
        a_low_i8x32 = _mm256_loadu_epi8(a);
        a_high_i8x32 = _mm256_loadu_epi8(a + 32);
        b_low_i8x32 = _mm256_loadu_epi8(b);
        b_high_i8x32 = _mm256_loadu_epi8(b + 32);
        c_low_i8x32 = _mm256_loadu_epi8(c);
        c_high_i8x32 = _mm256_loadu_epi8(c + 32);
        a += 64, b += 64, c += 64, n -= 64;
    }
    // Upcast from 256-bit halves:
    a_low_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(a_low_i8x32));
    a_high_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(a_high_i8x32));
    b_low_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(b_low_i8x32));
    b_high_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(b_high_i8x32));
    c_low_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(c_low_i8x32));
    c_high_f16x32 = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(c_high_i8x32));
    // Multiply:
    ab_low_f16x32 = _mm512_mul_ph(a_low_f16x32, b_low_f16x32);
    ab_high_f16x32 = _mm512_mul_ph(a_high_f16x32, b_high_f16x32);
    // Scale:
    ab_scaled_low_f16x32 = _mm512_mul_ph(ab_low_f16x32, alpha_f16x32);
    ab_scaled_high_f16x32 = _mm512_mul_ph(ab_high_f16x32, alpha_f16x32);
    // Add:
    result_low_f16x32 = _mm512_fmadd_ph(c_low_f16x32, beta_f16x32, ab_scaled_low_f16x32);
    result_high_f16x32 = _mm512_fmadd_ph(c_high_f16x32, beta_f16x32, ab_scaled_high_f16x32);
    // Clip the 16-bit result to 8-bit:
    result_low_f16x32 = _mm512_max_ph(_mm512_min_ph(result_low_f16x32, max_f16x32), min_f16x32);
    result_high_f16x32 = _mm512_max_ph(_mm512_min_ph(result_high_f16x32, max_f16x32), min_f16x32);
    // Downcast:
    result_low_i16x32 = _mm512_cvtph_epi16(result_low_f16x32);
    result_high_i16x32 = _mm512_cvtph_epi16(result_high_f16x32);
    // Merge back:
    result_i8x64 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(result_low_i16x32)),
                                      _mm512_cvtsepi16_epi8(result_high_i16x32), 1);
    _mm512_mask_storeu_epi8(result, mask, result_i8x64);
    result += 64;
    if (n) goto nk_fma_i8_sapphire_cycle;
}

NK_PUBLIC void nk_fma_u8_sapphire(                                     //
    nk_u8_t const *a, nk_u8_t const *b, nk_u8_t const *c, nk_size_t n, //
    nk_f32_t const *alpha, nk_f32_t const *beta, nk_u8_t *result) {

    short alpha_short, beta_short;
    nk_f32_to_f16_sapphire(alpha, (nk_f16_t *)&alpha_short);
    nk_f32_to_f16_sapphire(beta, (nk_f16_t *)&beta_short);
    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512h alpha_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(alpha_short));
    __m512h beta_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(beta_short));
    __m512i a_u8x64, b_u8x64, c_u8x64, result_u8x64;
    __m512h a_low_f16x32, a_high_f16x32, b_low_f16x32, b_high_f16x32;
    __m512h c_low_f16x32, c_high_f16x32, ab_low_f16x32, ab_high_f16x32;
    __m512h ab_scaled_low_f16x32, ab_scaled_high_f16x32, result_low_f16x32, result_high_f16x32;
    __m512i result_low_i16x32, result_high_i16x32;
    __m512h min_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(0));
    __m512h max_f16x32 = _mm512_cvtepi16_ph(_mm512_set1_epi16(255));

nk_fma_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
        b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
        c_u8x64 = _mm512_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8x64 = _mm512_loadu_epi8(a);
        b_u8x64 = _mm512_loadu_epi8(b);
        c_u8x64 = _mm512_loadu_epi8(c);
        a += 64, b += 64, c += 64, n -= 64;
    }
    // Upcast:
    a_low_f16x32 = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8x64, _mm512_setzero_si512()));
    a_high_f16x32 = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8x64, _mm512_setzero_si512()));
    b_low_f16x32 = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(b_u8x64, _mm512_setzero_si512()));
    b_high_f16x32 = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(b_u8x64, _mm512_setzero_si512()));
    c_low_f16x32 = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(c_u8x64, _mm512_setzero_si512()));
    c_high_f16x32 = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(c_u8x64, _mm512_setzero_si512()));
    // Multiply:
    ab_low_f16x32 = _mm512_mul_ph(a_low_f16x32, b_low_f16x32);
    ab_high_f16x32 = _mm512_mul_ph(a_high_f16x32, b_high_f16x32);
    // Scale:
    ab_scaled_low_f16x32 = _mm512_mul_ph(ab_low_f16x32, alpha_f16x32);
    ab_scaled_high_f16x32 = _mm512_mul_ph(ab_high_f16x32, alpha_f16x32);
    // Add:
    result_low_f16x32 = _mm512_fmadd_ph(c_low_f16x32, beta_f16x32, ab_scaled_low_f16x32);
    result_high_f16x32 = _mm512_fmadd_ph(c_high_f16x32, beta_f16x32, ab_scaled_high_f16x32);
    // Clip the 16-bit result to 8-bit:
    result_low_f16x32 = _mm512_max_ph(_mm512_min_ph(result_low_f16x32, max_f16x32), min_f16x32);
    result_high_f16x32 = _mm512_max_ph(_mm512_min_ph(result_high_f16x32, max_f16x32), min_f16x32);
    // Downcast:
    result_low_i16x32 = _mm512_cvtph_epi16(result_low_f16x32);
    result_high_i16x32 = _mm512_cvtph_epi16(result_high_f16x32);
    // Merge back:
    result_u8x64 = _mm512_packus_epi16(result_low_i16x32, result_high_i16x32);
    _mm512_mask_storeu_epi8(result, mask, result_u8x64);
    result += 64;
    if (n) goto nk_fma_u8_sapphire_cycle;
}

NK_PUBLIC void nk_sum_e4m3_sapphire(nk_e4m3_t const *a, nk_e4m3_t const *b, nk_size_t n, nk_e4m3_t *result) {
    __m256i a_e4m3x32, b_e4m3x32;
    __m256h a_lo_f16x16, a_hi_f16x16, b_lo_f16x16, b_hi_f16x16;
    __m256h sum_lo_f16x16, sum_hi_f16x16;
    __m128i result_lo_e4m3x16, result_hi_e4m3x16;
    __mmask32 mask = 0xFFFFFFFF;
nk_sum_e4m3_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
        a_e4m3x32 = _mm256_maskz_loadu_epi8(mask, a);
        b_e4m3x32 = _mm256_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_e4m3x32 = _mm256_loadu_si256((__m256i const *)a);
        b_e4m3x32 = _mm256_loadu_si256((__m256i const *)b);
        a += 32, b += 32, n -= 32;
    }

    // Convert e4m3x16 -> f16x16 (two halves)
    a_lo_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_castsi256_si128(a_e4m3x32));
    a_hi_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_extracti128_si256(a_e4m3x32, 1));
    b_lo_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_castsi256_si128(b_e4m3x32));
    b_hi_f16x16 = nk_e4m3x16_to_f16x16_sapphire_(_mm256_extracti128_si256(b_e4m3x32, 1));

    // Add in F16 - e4m3 sum is safe (max 896 < 65504)
    sum_lo_f16x16 = _mm256_add_ph(a_lo_f16x16, b_lo_f16x16);
    sum_hi_f16x16 = _mm256_add_ph(a_hi_f16x16, b_hi_f16x16);

    // Convert f16x16 -> e4m3x16
    result_lo_e4m3x16 = nk_f16x16_to_e4m3x16_sapphire_(sum_lo_f16x16);
    result_hi_e4m3x16 = nk_f16x16_to_e4m3x16_sapphire_(sum_hi_f16x16);

    // Pack and store
    __m256i result_e4m3x32 = _mm256_inserti128_si256(_mm256_castsi128_si256(result_lo_e4m3x16), result_hi_e4m3x16, 1);
    _mm256_mask_storeu_epi8(result, mask, result_e4m3x32);
    result += 32;
    if (n) goto nk_sum_e4m3_sapphire_cycle;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#endif // NK_ELEMENTWISE_SAPPHIRE_H