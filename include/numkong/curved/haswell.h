/**
 *  @brief SIMD-accelerated Bilinear Forms for Curved Spaces - x86 Haswell (AVX2) implementations.
 *  @file include/numkong/curved/haswell.h
 *  @sa include/numkong/curved.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  Implements f16 and bf16 bilinear forms using AVX2 with F16C conversion.
 */
#ifndef NK_CURVED_HASWELL_H
#define NK_CURVED_HASWELL_H

#include "numkong/types.h"

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_bilinear_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i)));
        __m256 cb_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b + j)));
            __m256 c_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cb_j_f32x8 = _mm256_fmadd_ps(b_f32x8, c_f32x8, cb_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, cb_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i = _mm256_cvtss_f32(_mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))));
            __m256 b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b + tail_start, tail_length);
            __m256 c_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cb_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(b_f32x8, c_f32x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                          nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        __m256 diff_i_f32x8 = _mm256_sub_ps(                          //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))), //
            _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(b + i))));
        __m256 cdiff_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(a + j))),
                _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b + j))));
            __m256 c_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cdiff_j_f32x8 = _mm256_fmadd_ps(diff_j_f32x8, c_f32x8, cdiff_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(diff_i_f32x8, cdiff_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t diff_i = _mm256_cvtss_f32(_mm256_sub_ps(             //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(a + i))), //
                _mm256_cvtph_ps(_mm_set1_epi16(*(short const *)(b + i)))));
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                nk_partial_load_f16x8_to_f32x8_haswell_(a + tail_start, tail_length),
                nk_partial_load_f16x8_to_f32x8_haswell_(b + tail_start, tail_length));
            __m256 c_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cdiff_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(diff_j_f32x8, c_f32x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_sqrt_f32_haswell_(sum);
}

NK_PUBLIC void nk_bilinear_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        // The `nk_bf16_to_f32_serial` is cheaper than `nk_bf16x8_to_f32x8_haswell_`
        nk_f32_t a_f32;
        nk_bf16_to_f32_serial(a + i, &a_f32);
        __m256 a_f32x8 = _mm256_set1_ps(a_f32);
        __m256 cb_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 b_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(b + j)));
            __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cb_j_f32x8 = _mm256_fmadd_ps(b_f32x8, c_f32x8, cb_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, cb_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            __m256 b_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(b + tail_start, tail_length);
            __m256 c_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cb_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(b_f32x8, c_f32x8));
            sum += a_i * cb_j;
        }
    }

    *result = sum;
}

NK_PUBLIC void nk_mahalanobis_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t a_i, b_i;
        nk_bf16_to_f32_serial(a + i, &a_i);
        nk_bf16_to_f32_serial(b + i, &b_i);
        __m256 diff_i_f32x8 = _mm256_sub_ps( //
            _mm256_set1_ps(a_i),             //
            _mm256_set1_ps(b_i));
        __m256 cdiff_j_f32x8 = _mm256_setzero_ps();
        for (nk_size_t j = 0; j + 8 <= n; j += 8) {
            __m256 diff_j_f32x8 = _mm256_sub_ps(                                        //
                nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(a + j))), //
                nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(b + j))));
            __m256 c_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(c + i * n + j)));
            cdiff_j_f32x8 = _mm256_fmadd_ps(diff_j_f32x8, c_f32x8, cdiff_j_f32x8);
        }
        sum_f32x8 = _mm256_fmadd_ps(diff_i_f32x8, cdiff_j_f32x8, sum_f32x8);
    }

    // Handle the tail of every row
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    nk_size_t const tail_length = n % 8;
    nk_size_t const tail_start = n - tail_length;
    if (tail_length) {
        for (nk_size_t i = 0; i != n; ++i) {
            nk_f32_t a_i, b_i;
            nk_bf16_to_f32_serial(a + i, &a_i);
            nk_bf16_to_f32_serial(b + i, &b_i);
            nk_f32_t diff_i = a_i - b_i;
            __m256 diff_j_f32x8 = _mm256_sub_ps( //
                nk_partial_load_bf16x8_to_f32x8_haswell_(a + tail_start, tail_length),
                nk_partial_load_bf16x8_to_f32x8_haswell_(b + tail_start, tail_length));
            __m256 c_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(c + i * n + tail_start, tail_length);
            nk_f32_t cdiff_j = nk_reduce_add_f32x8_haswell_(_mm256_mul_ps(diff_j_f32x8, c_f32x8));
            sum += diff_i * cdiff_j;
        }
    }

    *result = nk_sqrt_f32_haswell_(sum);
}

#if defined(__cplusplus)
}
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_CURVED_HASWELL_H
