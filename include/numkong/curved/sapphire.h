/**
 *  @brief SIMD-accelerated Bilinear Forms for Curved Spaces - Intel Sapphire Rapids (AVX-512-FP16) implementations.
 *  @file include/numkong/curved/sapphire.h
 *  @sa include/numkong/curved.h
 *  @author Ash Vardanian
 *  @date January 14, 2026
 *
 *  Implements f16 bilinear forms using AVX-512 with native FP16 support.
 */
#ifndef NK_CURVED_SAPPHIRE_H
#define NK_CURVED_SAPPHIRE_H

#include "numkong/types.h"

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_bilinear_f16_sapphire_under32unrolled(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c,
                                                        nk_size_t const n, nk_f32_t *result) {
    // The goal of this optimization is to avoid horizontal accumulation of the cb_j sums
    // until the very end of the computation.
    __mmask32 const mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
    __m512h const b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));

    // Independently accumulate the partial sums into separate variables to avoid data-dependencies.
    __m512h cb_j1_f16x32 = _mm512_setzero_ph();
    __m512h cb_j2_f16x32 = _mm512_setzero_ph();
    __m512h cb_j3_f16x32 = _mm512_setzero_ph();
    __m512h cb_j4_f16x32 = _mm512_setzero_ph();

    // Unroll the loop to process 4x ZMM registers at a time.
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        // If the code is compiled without native support for `_Float16`, we need a workaround
        // to avoid implicit casts from out `nk_f16_t` to `_Float16`.
        cb_j1_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 0))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 0]))),
            cb_j1_f16x32);
        cb_j2_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 1))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 1]))),
            cb_j2_f16x32);
        cb_j3_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 2))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 2]))),
            cb_j3_f16x32);
        cb_j4_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 3))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 3]))),
            cb_j4_f16x32);
    }

    // Handle the tail of the loop:
    if (i + 0 < n)
        cb_j1_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 0))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 0]))),
            cb_j1_f16x32);
    if (i + 1 < n)
        cb_j2_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 1))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 1]))),
            cb_j2_f16x32);
    if (i + 2 < n)
        cb_j3_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 2))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 2]))),
            cb_j3_f16x32);
    if (i + 3 < n)
        cb_j4_f16x32 = _mm512_fmadd_ph(
            _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c + n * (i + 3))),
            _mm512_mul_ph(b_f16x32, _mm512_castsi512_ph(_mm512_set1_epi16(((nk_i16_t const *)a)[i + 3]))),
            cb_j4_f16x32);

    // Combine cb_j sums
    __m512h sum_f16x32 = _mm512_add_ph(            //
        _mm512_add_ph(cb_j1_f16x32, cb_j2_f16x32), //
        _mm512_add_ph(cb_j3_f16x32, cb_j4_f16x32));
    *result = _mm512_reduce_add_ph(sum_f16x32);
}

NK_PUBLIC void nk_bilinear_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result) {

    // On modern x86 CPUs we have enough register space to load fairly large matrices with up to 32 cells
    // per row and 32 rows at a time, still keeping enough register space for temporaries.
    if (n <= 32) {
        nk_bilinear_f16_sapphire_under32unrolled(a, b, c, n, result);
        return;
    }

    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512h sum_f16x32 = _mm512_setzero_ph();

    for (nk_size_t i = 0; i != n; ++i) {
        __m512h a_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const *)(a + i)));
        __m512h cb_j_f16x32 = _mm512_setzero_ph();
        __m512i b_f16x32, c_f16x32;
        nk_size_t j = 0;

    nk_bilinear_f16_sapphire_cycle:
        if (j + 32 <= n) {
            b_f16x32 = _mm512_loadu_epi16(b + j);
            c_f16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            b_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        cb_j_f16x32 = _mm512_fmadd_ph(_mm512_castsi512_ph(b_f16x32), _mm512_castsi512_ph(c_f16x32), cb_j_f16x32);
        j += 32;
        if (j < n) goto nk_bilinear_f16_sapphire_cycle;
        sum_f16x32 = _mm512_fmadd_ph(a_f16x32, cb_j_f16x32, sum_f16x32);
    }

    *result = _mm512_reduce_add_ph(sum_f16x32);
}

NK_PUBLIC void nk_mahalanobis_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result) {
    nk_size_t const tail_length = n % 32;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length);
    __m512h sum_f16x32 = _mm512_setzero_ph();

    for (nk_size_t i = 0; i != n; ++i) {
        __m512h a_i_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const *)(a + i)));
        __m512h b_i_f16x32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(short const *)(b + i)));
        __m512h diff_i_f16x32 = _mm512_sub_ph(a_i_f16x32, b_i_f16x32);
        __m512h cdiff_j_f16x32 = _mm512_setzero_ph();
        __m512h diff_j_f16x32;
        __m512i a_j_f16x32, b_j_f16x32, c_f16x32;
        nk_size_t j = 0;

        // The nested loop is cleaner to implement with a `goto` in this case:
    nk_mahalanobis_f16_sapphire_cycle:
        if (j + 32 <= n) {
            a_j_f16x32 = _mm512_loadu_epi16(a + j);
            b_j_f16x32 = _mm512_loadu_epi16(b + j);
            c_f16x32 = _mm512_loadu_epi16(c + i * n + j);
        }
        else {
            a_j_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, a + tail_start);
            b_j_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, b + tail_start);
            c_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, c + i * n + tail_start);
        }
        diff_j_f16x32 = _mm512_sub_ph(_mm512_castsi512_ph(a_j_f16x32), _mm512_castsi512_ph(b_j_f16x32));
        cdiff_j_f16x32 = _mm512_fmadd_ph(diff_j_f16x32, _mm512_castsi512_ph(c_f16x32), cdiff_j_f16x32);
        j += 32;
        if (j < n) goto nk_mahalanobis_f16_sapphire_cycle;
        sum_f16x32 = _mm512_fmadd_ph(diff_i_f16x32, cdiff_j_f16x32, sum_f16x32);
    }

    *result = nk_sqrt_f32_haswell_(_mm512_reduce_add_ph(sum_f16x32));
}

NK_PUBLIC void nk_bilinear_f16c_sapphire(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f32c_t *results) {

    // We take into account, that FMS is the same as FMA with a negative multiplier.
    // To multiply a floating-point value by -1, we can use the `XOR` instruction to flip the sign bit.
    // This way we can avoid the shuffling and the need for separate real and imaginary parts.
    // For the imaginary part of the product, we would need to swap the real and imaginary parts of
    // one of the vectors.
    __m512i const sign_flip_i32x16 = _mm512_set1_epi32(0x80000000);
    __m512i const swap_adjacent_i8x64 = _mm512_set_epi8(                //
        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50, // 4th 128-bit lane
        45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, // 3rd 128-bit lane
        29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, // 2nd 128-bit lane
        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2            // 1st 128-bit lane
    );

    // Default case for arbitrary size `n`
    nk_size_t const tail_length = n % 16;
    nk_size_t const tail_start = n - tail_length;
    __mmask32 const tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, tail_length * 2);
    nk_f32_t sum_real = 0;
    nk_f32_t sum_imag = 0;

    for (nk_size_t i = 0; i != n; ++i) {
        nk_f32_t const a_i_real = a[i].real;
        nk_f32_t const a_i_imag = a[i].imag;
        __m512h cb_j_real_f16x32 = _mm512_setzero_ph();
        __m512h cb_j_imag_f16x32 = _mm512_setzero_ph();
        __m512i b_f16x32, c_f16x32;
        nk_size_t j = 0;

    nk_bilinear_f16c_sapphire_cycle:
        if (j + 16 <= n) {
            b_f16x32 = _mm512_loadu_epi16((nk_i16_t const *)(b + j));
            c_f16x32 = _mm512_loadu_epi16((nk_i16_t const *)(c + i * n + j));
        }
        else {
            b_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(b + tail_start));
            c_f16x32 = _mm512_maskz_loadu_epi16(tail_mask, (nk_i16_t const *)(c + i * n + tail_start));
        }
        cb_j_real_f16x32 = _mm512_fmadd_ph(                                    //
            _mm512_castsi512_ph(_mm512_xor_si512(c_f16x32, sign_flip_i32x16)), //
            _mm512_castsi512_ph(b_f16x32), cb_j_real_f16x32);
        cb_j_imag_f16x32 = _mm512_fmadd_ph(                                          //
            _mm512_castsi512_ph(_mm512_shuffle_epi8(c_f16x32, swap_adjacent_i8x64)), //
            _mm512_castsi512_ph(b_f16x32), cb_j_imag_f16x32);
        j += 16;
        if (j < n) goto nk_bilinear_f16c_sapphire_cycle;
        // Horizontal sums are the expensive part of the computation:
        nk_f32_t const cb_j_real = _mm512_reduce_add_ph(cb_j_real_f16x32);
        nk_f32_t const cb_j_imag = _mm512_reduce_add_ph(cb_j_imag_f16x32);
        sum_real += a_i_real * cb_j_real - a_i_imag * cb_j_imag;
        sum_imag += a_i_real * cb_j_imag + a_i_imag * cb_j_real;
    }

    // Reduce horizontal sums:
    results->real = sum_real;
    results->imag = sum_imag;
}

#if defined(__cplusplus)
}
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_
#endif // NK_CURVED_SAPPHIRE_H
