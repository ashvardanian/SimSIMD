/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/dot/haswell.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_HASWELL_H
#define NK_DOT_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/dot/serial.h" // `nk_popcount_b8`
#include "numkong/reduce/haswell.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f32_haswell(nk_f32_t const *a_scalars, nk_f32_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count_scalars; idx_scalars += 8) {
        __m256 a_f32x8 = _mm256_loadu_ps(a_scalars + idx_scalars);
        __m256 b_f32x8 = _mm256_loadu_ps(b_scalars + idx_scalars);
        sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    }
    nk_f32_t sum = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_f64_haswell(nk_f64_t const *a_scalars, nk_f64_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f64_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count_scalars; idx_scalars += 4) {
        __m256d a_f64x4 = _mm256_loadu_pd(a_scalars + idx_scalars);
        __m256d b_f64x4 = _mm256_loadu_pd(b_scalars + idx_scalars);
        sum_f64x4 = _mm256_fmadd_pd(a_f64x4, b_f64x4, sum_f64x4);
    }
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_f32c_haswell(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    // Using XOR to flip sign bits is cheaper than separate FMA/FMS. Throughput doubles from 2.5 GB/s to 5 GB/s.
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps((nk_f32_t const *)(a_pairs + idx_pairs));
        __m256 b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)(b_pairs + idx_pairs));
        __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
            _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_real_f32x8), sign_flip_i64x4));
    nk_f32_t sum_real = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    nk_f32_t sum_imag = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += a_pair.real * b_pair.real - a_pair.imag * b_pair.imag;
        sum_imag += a_pair.real * b_pair.imag + a_pair.imag * b_pair.real;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

NK_PUBLIC void nk_vdot_f32c_haswell(nk_f32c_t const *a_pairs, nk_f32c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    nk_size_t idx_pairs = 0;
    for (; idx_pairs + 4 <= count_pairs; idx_pairs += 4) {
        __m256 a_f32x8 = _mm256_loadu_ps((nk_f32_t const *)(a_pairs + idx_pairs));
        __m256 b_f32x8 = _mm256_loadu_ps((nk_f32_t const *)(b_pairs + idx_pairs));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        b_f32x8 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_imag_f32x8);
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_imag_f32x8), sign_flip_i64x4));
    nk_f32_t sum_real = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    nk_f32_t sum_imag = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
    for (; idx_pairs != count_pairs; ++idx_pairs) {
        nk_f32c_t a_pair = a_pairs[idx_pairs], b_pair = b_pairs[idx_pairs];
        sum_real += a_pair.real * b_pair.real + a_pair.imag * b_pair.imag;
        sum_imag += a_pair.real * b_pair.imag - a_pair.imag * b_pair.real;
    }
    result->real = sum_real;
    result->imag = sum_imag;
}

NK_INTERNAL void nk_partial_store_f32x8_haswell_(__m256 vec, nk_f32_t *x, nk_size_t n) {
    nk_b256_vec_t u;
    u.ymm_ps = vec;
    if (n > 0) x[0] = u.f32s[0];
    if (n > 1) x[1] = u.f32s[1];
    if (n > 2) x[2] = u.f32s[2];
    if (n > 3) x[3] = u.f32s[3];
    if (n > 4) x[4] = u.f32s[4];
    if (n > 5) x[5] = u.f32s[5];
    if (n > 6) x[6] = u.f32s[6];
    if (n > 7) x[7] = u.f32s[7];
}

NK_INTERNAL void nk_partial_store_i32x8_haswell_(__m256i vec, nk_i32_t *x, nk_size_t n) {
    nk_b256_vec_t u;
    u.ymm = vec;
    if (n > 0) x[0] = u.i32s[0];
    if (n > 1) x[1] = u.i32s[1];
    if (n > 2) x[2] = u.i32s[2];
    if (n > 3) x[3] = u.i32s[3];
    if (n > 4) x[4] = u.i32s[4];
    if (n > 5) x[5] = u.i32s[5];
    if (n > 6) x[6] = u.i32s[6];
    if (n > 7) x[7] = u.i32s[7];
}

NK_PUBLIC void nk_dot_f16_haswell(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_f16_haswell_cycle:
    if (count_scalars < 8) {
        a_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(a_scalars, count_scalars);
        b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_scalars));
        b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_scalars));
        count_scalars -= 8, a_scalars += 8, b_scalars += 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_f16_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_f16c_haswell(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    while (count_pairs >= 4) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_pairs));
        __m256 b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_pairs));
        __m256 b_swapped_f32x8 = _mm256_castsi256_ps(
            _mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_swapped_f32x8, sum_imag_f32x8);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_real_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_real_f32x8), sign_flip_i64x4));
    nk_f32c_t tail_result;
    nk_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    result->imag = tail_result.imag + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
}

NK_PUBLIC void nk_vdot_f16c_haswell(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    __m256 sum_real_f32x8 = _mm256_setzero_ps();
    __m256 sum_imag_f32x8 = _mm256_setzero_ps();
    __m256i sign_flip_i64x4 = _mm256_set1_epi64x(0x8000000000000000);
    __m256i swap_adjacent_i8x32 = _mm256_set_epi8( //
        11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
    while (count_pairs >= 4) {
        __m256 a_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a_pairs));
        __m256 b_f32x8 = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b_pairs));
        sum_real_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_real_f32x8);
        b_f32x8 = _mm256_castsi256_ps(_mm256_shuffle_epi8(_mm256_castps_si256(b_f32x8), swap_adjacent_i8x32));
        sum_imag_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_imag_f32x8);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Flip the sign bit in every second scalar before accumulation:
    sum_imag_f32x8 = _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(sum_imag_f32x8), sign_flip_i64x4));
    nk_f32c_t tail_result;
    nk_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_real_f32x8);
    result->imag = tail_result.imag + (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_imag_f32x8);
}

NK_PUBLIC void nk_dot_i8_haswell(nk_i8_t const *a_scalars, nk_i8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_i32_t *result) {
    __m256i sum_low_i32x8 = _mm256_setzero_si256();
    __m256i sum_high_i32x8 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    // Use two 128-bit loads instead of 256-bit load + extract to avoid Port 5 contention.
    // VEXTRACTI128 uses Port 5; two smaller loads use Port 2/3 (2 ports available).
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m128i a_low_i8x16 = _mm_lddqu_si128((__m128i const *)(a_scalars + idx_scalars));
        __m128i a_high_i8x16 = _mm_lddqu_si128((__m128i const *)(a_scalars + idx_scalars + 16));
        __m128i b_low_i8x16 = _mm_lddqu_si128((__m128i const *)(b_scalars + idx_scalars));
        __m128i b_high_i8x16 = _mm_lddqu_si128((__m128i const *)(b_scalars + idx_scalars + 16));
        // Upcast `int8` to `int16` - no extracts needed
        __m256i a_low_i16x16 = _mm256_cvtepi8_epi16(a_low_i8x16);
        __m256i a_high_i16x16 = _mm256_cvtepi8_epi16(a_high_i8x16);
        __m256i b_low_i16x16 = _mm256_cvtepi8_epi16(b_low_i8x16);
        __m256i b_high_i16x16 = _mm256_cvtepi8_epi16(b_high_i8x16);
        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        sum_low_i32x8 = _mm256_add_epi32(sum_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        sum_high_i32x8 = _mm256_add_epi32(sum_high_i32x8, _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
    }
    nk_i32_t sum = nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(sum_low_i32x8, sum_high_i32x8));
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_i32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_dot_u8_haswell(nk_u8_t const *a_scalars, nk_u8_t const *b_scalars, nk_size_t count_scalars,
                                 nk_u32_t *result) {
    __m256i sum_low_i32x8 = _mm256_setzero_si256();
    __m256i sum_high_i32x8 = _mm256_setzero_si256();
    __m256i const zeros_i8x32 = _mm256_setzero_si256();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 32 <= count_scalars; idx_scalars += 32) {
        __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a_scalars + idx_scalars));
        __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b_scalars + idx_scalars));
        // Upcast `uint8` to `int16`. Unpacking is faster than extracts.
        __m256i a_low_i16x16 = _mm256_unpacklo_epi8(a_u8x32, zeros_i8x32);
        __m256i a_high_i16x16 = _mm256_unpackhi_epi8(a_u8x32, zeros_i8x32);
        __m256i b_low_i16x16 = _mm256_unpacklo_epi8(b_u8x32, zeros_i8x32);
        __m256i b_high_i16x16 = _mm256_unpackhi_epi8(b_u8x32, zeros_i8x32);
        // Multiply and accumulate at `int16` level, accumulate at `int32` level
        sum_low_i32x8 = _mm256_add_epi32(sum_low_i32x8, _mm256_madd_epi16(a_low_i16x16, b_low_i16x16));
        sum_high_i32x8 = _mm256_add_epi32(sum_high_i32x8, _mm256_madd_epi16(a_high_i16x16, b_high_i16x16));
    }
    nk_u32_t sum = (nk_u32_t)nk_reduce_add_i32x8_haswell_(_mm256_add_epi32(sum_low_i32x8, sum_high_i32x8));
    for (; idx_scalars < count_scalars; ++idx_scalars) sum += (nk_u32_t)a_scalars[idx_scalars] * b_scalars[idx_scalars];
    *result = sum;
}

NK_INTERNAL __m128i nk_f32x8_to_bf16x8_haswell_(__m256 x) {
    // Pack the 32-bit integers into 16-bit integers.
    // This is less trivial than unpacking: https://stackoverflow.com/a/77781241/2766161
    // The best approach is to shuffle within lanes first: https://stackoverflow.com/a/49723746/2766161
    // Our shuffling mask will drop the low 2-bytes from every 4-byte word.
    __m256i trunc_elements = _mm256_shuffle_epi8(                       //
        _mm256_castps_si256(x),                                         //
        _mm256_set_epi8(                                                //
            -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 11, 10, 7, 6, 3, 2, //
            -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 11, 10, 7, 6, 3, 2  //
            ));
    __m256i ordered = _mm256_permute4x64_epi64(trunc_elements, 0x58);
    __m128i result = _mm256_castsi256_si128(ordered);
    return result;
}

NK_PUBLIC void nk_dot_bf16_haswell(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m128i a_bf16x8, b_bf16x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_bf16_haswell_cycle:
    if (count_scalars < 8) {
        nk_b256_vec_t a_vec, b_vec;
        nk_partial_load_b16x16_haswell_(a_scalars, count_scalars, &a_vec);
        nk_partial_load_b16x16_haswell_(b_scalars, count_scalars, &b_vec);
        a_bf16x8 = a_vec.xmms[0];
        b_bf16x8 = b_vec.xmms[0];
        count_scalars = 0;
    }
    else {
        a_bf16x8 = _mm_lddqu_si128((__m128i const *)a_scalars);
        b_bf16x8 = _mm_lddqu_si128((__m128i const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(nk_bf16x8_to_f32x8_haswell_(a_bf16x8), nk_bf16x8_to_f32x8_haswell_(b_bf16x8),
                                sum_f32x8);
    if (count_scalars) goto nk_dot_bf16_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_e4m3_haswell(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_e4m3_haswell_cycle:
    if (count_scalars < 8) {
        a_f32x8 = nk_partial_load_e4m3x8_to_f32x8_haswell_(a_scalars, count_scalars);
        b_f32x8 = nk_partial_load_e4m3x8_to_f32x8_haswell_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)a_scalars));
        b_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_e4m3_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_PUBLIC void nk_dot_e5m2_haswell(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    __m256 a_f32x8, b_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_dot_e5m2_haswell_cycle:
    if (count_scalars < 8) {
        a_f32x8 = nk_partial_load_e5m2x8_to_f32x8_haswell_(a_scalars, count_scalars);
        b_f32x8 = nk_partial_load_e5m2x8_to_f32x8_haswell_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)a_scalars));
        b_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)b_scalars));
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x8 = _mm256_fmadd_ps(a_f32x8, b_f32x8, sum_f32x8);
    if (count_scalars) goto nk_dot_e5m2_haswell_cycle;
    *result = (nk_f32_t)nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

typedef struct nk_dot_f32x8_state_haswell_t {
    __m256 sum_f32x8;
} nk_dot_f32x8_state_haswell_t;

NK_INTERNAL void nk_dot_f32x8_init_haswell(nk_dot_f32x8_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

NK_INTERNAL void nk_dot_f32x8_update_haswell(nk_dot_f32x8_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(a.ymm_ps, b.ymm_ps, sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

NK_INTERNAL void nk_dot_f32x8_finalize_haswell(                                               //
    nk_dot_f32x8_state_haswell_t const *state_a, nk_dot_f32x8_state_haswell_t const *state_b, //
    nk_dot_f32x8_state_haswell_t const *state_c, nk_dot_f32x8_state_haswell_t const *state_d, //
    nk_f32_t *results) {
    // ILP-optimized 4-way horizontal reduction for f32 in AVX2
    __m128 sum_f32x4_a = _mm_add_ps(_mm256_castps256_ps128(state_a->sum_f32x8),
                                    _mm256_extractf128_ps(state_a->sum_f32x8, 1));
    __m128 sum_f32x4_b = _mm_add_ps(_mm256_castps256_ps128(state_b->sum_f32x8),
                                    _mm256_extractf128_ps(state_b->sum_f32x8, 1));
    __m128 sum_f32x4_c = _mm_add_ps(_mm256_castps256_ps128(state_c->sum_f32x8),
                                    _mm256_extractf128_ps(state_c->sum_f32x8, 1));
    __m128 sum_f32x4_d = _mm_add_ps(_mm256_castps256_ps128(state_d->sum_f32x8),
                                    _mm256_extractf128_ps(state_d->sum_f32x8, 1));
    __m128 transpose_ab_low_f32x4 = _mm_unpacklo_ps(sum_f32x4_a, sum_f32x4_b);
    __m128 transpose_cd_low_f32x4 = _mm_unpacklo_ps(sum_f32x4_c, sum_f32x4_d);
    __m128 transpose_ab_high_f32x4 = _mm_unpackhi_ps(sum_f32x4_a, sum_f32x4_b);
    __m128 transpose_cd_high_f32x4 = _mm_unpackhi_ps(sum_f32x4_c, sum_f32x4_d);
    __m128 sum_lane0_f32x4 = _mm_movelh_ps(transpose_ab_low_f32x4, transpose_cd_low_f32x4);
    __m128 sum_lane1_f32x4 = _mm_movehl_ps(transpose_cd_low_f32x4, transpose_ab_low_f32x4);
    __m128 sum_lane2_f32x4 = _mm_movelh_ps(transpose_ab_high_f32x4, transpose_cd_high_f32x4);
    __m128 sum_lane3_f32x4 = _mm_movehl_ps(transpose_cd_high_f32x4, transpose_ab_high_f32x4);
    __m128 final_sum_f32x4 = _mm_add_ps(_mm_add_ps(sum_lane0_f32x4, sum_lane1_f32x4),
                                        _mm_add_ps(sum_lane2_f32x4, sum_lane3_f32x4));
    _mm_storeu_ps(results, final_sum_f32x4);
}

typedef struct nk_dot_f16x16_state_haswell_t {
    __m256 sum_f32x8;
} nk_dot_f16x16_state_haswell_t;

NK_INTERNAL void nk_dot_f16x16_init_haswell(nk_dot_f16x16_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

NK_INTERNAL void nk_dot_f16x16_update_haswell(nk_dot_f16x16_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 0))),
                                _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 0))), sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(a.f16s + 8))),
                                _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)(b.f16s + 8))), sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

NK_INTERNAL void nk_dot_f16x16_finalize_haswell(                                                //
    nk_dot_f16x16_state_haswell_t const *state_a, nk_dot_f16x16_state_haswell_t const *state_b, //
    nk_dot_f16x16_state_haswell_t const *state_c, nk_dot_f16x16_state_haswell_t const *state_d, //
    nk_f32_t *results) {
    nk_dot_f32x8_finalize_haswell(                                                                    //
        (nk_dot_f32x8_state_haswell_t const *)state_a, (nk_dot_f32x8_state_haswell_t const *)state_b, //
        (nk_dot_f32x8_state_haswell_t const *)state_c, (nk_dot_f32x8_state_haswell_t const *)state_d, results);
}

typedef struct nk_dot_bf16x16_state_haswell_t {
    __m256 sum_f32x8;
} nk_dot_bf16x16_state_haswell_t;

NK_INTERNAL void nk_dot_bf16x16_init_haswell(nk_dot_bf16x16_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

NK_INTERNAL void nk_dot_bf16x16_update_haswell(nk_dot_bf16x16_state_haswell_t *state, nk_b256_vec_t a,
                                               nk_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)(a.bf16s + 0))),
                                nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)(b.bf16s + 0))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)(a.bf16s + 8))),
                                nk_bf16x8_to_f32x8_haswell_(_mm_lddqu_si128((__m128i const *)(b.bf16s + 8))),
                                sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

NK_INTERNAL void nk_dot_bf16x16_finalize_haswell(                                                 //
    nk_dot_bf16x16_state_haswell_t const *state_a, nk_dot_bf16x16_state_haswell_t const *state_b, //
    nk_dot_bf16x16_state_haswell_t const *state_c, nk_dot_bf16x16_state_haswell_t const *state_d, //
    nk_f32_t *results) {
    nk_dot_f32x8_finalize_haswell(                                                                    //
        (nk_dot_f32x8_state_haswell_t const *)state_a, (nk_dot_f32x8_state_haswell_t const *)state_b, //
        (nk_dot_f32x8_state_haswell_t const *)state_c, (nk_dot_f32x8_state_haswell_t const *)state_d, results);
}

typedef struct nk_dot_e4m3x32_state_haswell_t {
    __m256 sum_f32x8;
} nk_dot_e4m3x32_state_haswell_t;

NK_INTERNAL void nk_dot_e4m3x32_init_haswell(nk_dot_e4m3x32_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

NK_INTERNAL void nk_dot_e4m3x32_update_haswell(nk_dot_e4m3x32_state_haswell_t *state, nk_b256_vec_t a,
                                               nk_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 0))),
                                nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 0))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 8))),
                                nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 8))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 16))),
                                nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 16))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e4m3s + 24))),
                                nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e4m3s + 24))),
                                sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

NK_INTERNAL void nk_dot_e4m3x32_finalize_haswell(                                                 //
    nk_dot_e4m3x32_state_haswell_t const *state_a, nk_dot_e4m3x32_state_haswell_t const *state_b, //
    nk_dot_e4m3x32_state_haswell_t const *state_c, nk_dot_e4m3x32_state_haswell_t const *state_d, //
    nk_f32_t *results) {
    nk_dot_f32x8_finalize_haswell(                                                                    //
        (nk_dot_f32x8_state_haswell_t const *)state_a, (nk_dot_f32x8_state_haswell_t const *)state_b, //
        (nk_dot_f32x8_state_haswell_t const *)state_c, (nk_dot_f32x8_state_haswell_t const *)state_d, results);
}

typedef struct nk_dot_e5m2x32_state_haswell_t {
    __m256 sum_f32x8;
} nk_dot_e5m2x32_state_haswell_t;

NK_INTERNAL void nk_dot_e5m2x32_init_haswell(nk_dot_e5m2x32_state_haswell_t *state) {
    state->sum_f32x8 = _mm256_setzero_ps();
}

NK_INTERNAL void nk_dot_e5m2x32_update_haswell(nk_dot_e5m2x32_state_haswell_t *state, nk_b256_vec_t a,
                                               nk_b256_vec_t b) {
    __m256 sum_f32x8 = state->sum_f32x8;
    sum_f32x8 = _mm256_fmadd_ps(nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 0))),
                                nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 0))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 8))),
                                nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 8))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 16))),
                                nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 16))),
                                sum_f32x8);
    sum_f32x8 = _mm256_fmadd_ps(nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(a.e5m2s + 24))),
                                nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)(b.e5m2s + 24))),
                                sum_f32x8);
    state->sum_f32x8 = sum_f32x8;
}

NK_INTERNAL void nk_dot_e5m2x32_finalize_haswell(                                                 //
    nk_dot_e5m2x32_state_haswell_t const *state_a, nk_dot_e5m2x32_state_haswell_t const *state_b, //
    nk_dot_e5m2x32_state_haswell_t const *state_c, nk_dot_e5m2x32_state_haswell_t const *state_d, //
    nk_f32_t *results) {
    nk_dot_f32x8_finalize_haswell(                                                                    //
        (nk_dot_f32x8_state_haswell_t const *)state_a, (nk_dot_f32x8_state_haswell_t const *)state_b, //
        (nk_dot_f32x8_state_haswell_t const *)state_c, (nk_dot_f32x8_state_haswell_t const *)state_d, results);
}

/**
 *  @brief Running state for 256-bit dot accumulation over i8 scalars on Haswell.
 */
typedef struct nk_dot_i8x32_state_haswell_t {
    __m256i sum_i32x8_low;
    __m256i sum_i32x8_high;
} nk_dot_i8x32_state_haswell_t;

NK_INTERNAL void nk_dot_i8x32_init_haswell(nk_dot_i8x32_state_haswell_t *state) {
    state->sum_i32x8_low = _mm256_setzero_si256();
    state->sum_i32x8_high = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_i8x32_update_haswell(nk_dot_i8x32_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m256i sum_i32x8_low = state->sum_i32x8_low;
    __m256i sum_i32x8_high = state->sum_i32x8_high;
    // Use two 128-bit loads instead of 256-bit load + extract to avoid Port 5 contention.
    __m128i a_low_i8x16 = _mm_lddqu_si128((__m128i const *)(a.i8s + 0));
    __m128i a_high_i8x16 = _mm_lddqu_si128((__m128i const *)(a.i8s + 16));
    __m128i b_low_i8x16 = _mm_lddqu_si128((__m128i const *)(b.i8s + 0));
    __m128i b_high_i8x16 = _mm_lddqu_si128((__m128i const *)(b.i8s + 16));
    __m256i a_i16x16_low = _mm256_cvtepi8_epi16(a_low_i8x16);
    __m256i a_i16x16_high = _mm256_cvtepi8_epi16(a_high_i8x16);
    __m256i b_i16x16_low = _mm256_cvtepi8_epi16(b_low_i8x16);
    __m256i b_i16x16_high = _mm256_cvtepi8_epi16(b_high_i8x16);
    sum_i32x8_low = _mm256_add_epi32(sum_i32x8_low, _mm256_madd_epi16(a_i16x16_low, b_i16x16_low));
    sum_i32x8_high = _mm256_add_epi32(sum_i32x8_high, _mm256_madd_epi16(a_i16x16_high, b_i16x16_high));
    state->sum_i32x8_low = sum_i32x8_low;
    state->sum_i32x8_high = sum_i32x8_high;
}

NK_INTERNAL void nk_dot_i8x32_finalize_haswell(                                               //
    nk_dot_i8x32_state_haswell_t const *state_a, nk_dot_i8x32_state_haswell_t const *state_b, //
    nk_dot_i8x32_state_haswell_t const *state_c, nk_dot_i8x32_state_haswell_t const *state_d, //
    nk_i32_t *results) {
    // First, combine the low and high accumulators for each state
    __m256i sum_i32x8_a = _mm256_add_epi32(state_a->sum_i32x8_low, state_a->sum_i32x8_high);
    __m256i sum_i32x8_b = _mm256_add_epi32(state_b->sum_i32x8_low, state_b->sum_i32x8_high);
    __m256i sum_i32x8_c = _mm256_add_epi32(state_c->sum_i32x8_low, state_c->sum_i32x8_high);
    __m256i sum_i32x8_d = _mm256_add_epi32(state_d->sum_i32x8_low, state_d->sum_i32x8_high);
    // ILP-optimized 4-way horizontal reduction for i32 in AVX2
    // Step 1: 8->4 for all 4 states
    __m128i sum_i32x4_a = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_a), _mm256_extracti128_si256(sum_i32x8_a, 1));
    __m128i sum_i32x4_b = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_b), _mm256_extracti128_si256(sum_i32x8_b, 1));
    __m128i sum_i32x4_c = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_c), _mm256_extracti128_si256(sum_i32x8_c, 1));
    __m128i sum_i32x4_d = _mm_add_epi32(_mm256_castsi256_si128(sum_i32x8_d), _mm256_extracti128_si256(sum_i32x8_d, 1));
    // Step 2: Transpose 4x4 matrix
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_a, sum_i32x4_b);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_i32x4_c, sum_i32x4_d);
    __m128i sum_lane0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    // Step 3: Vertical sum and store as i32
    __m128i final_sum_i32x4 = _mm_add_epi32(_mm_add_epi32(sum_lane0_i32x4, sum_lane1_i32x4),
                                            _mm_add_epi32(sum_lane2_i32x4, sum_lane3_i32x4));
    _mm_storeu_si128((__m128i *)results, final_sum_i32x4);
}

typedef struct nk_dot_u8x32_state_haswell_t {
    __m256i sum_u32x8_low;
    __m256i sum_u32x8_high;
} nk_dot_u8x32_state_haswell_t;

NK_INTERNAL void nk_dot_u8x32_init_haswell(nk_dot_u8x32_state_haswell_t *state) {
    state->sum_u32x8_low = _mm256_setzero_si256();
    state->sum_u32x8_high = _mm256_setzero_si256();
}

NK_INTERNAL void nk_dot_u8x32_update_haswell(nk_dot_u8x32_state_haswell_t *state, nk_b256_vec_t a, nk_b256_vec_t b) {
    __m256i sum_u32x8_low = state->sum_u32x8_low;
    __m256i sum_u32x8_high = state->sum_u32x8_high;
    __m256i const zeros_u8x32 = _mm256_setzero_si256();

    __m256i a_u8x32 = _mm256_lddqu_si256((__m256i const *)(a.u8s + 0));
    __m256i b_u8x32 = _mm256_lddqu_si256((__m256i const *)(b.u8s + 0));
    __m256i a_u16x16_low = _mm256_unpacklo_epi8(a_u8x32, zeros_u8x32);
    __m256i a_u16x16_high = _mm256_unpackhi_epi8(a_u8x32, zeros_u8x32);
    __m256i b_u16x16_low = _mm256_unpacklo_epi8(b_u8x32, zeros_u8x32);
    __m256i b_u16x16_high = _mm256_unpackhi_epi8(b_u8x32, zeros_u8x32);
    sum_u32x8_low = _mm256_add_epi32(sum_u32x8_low, _mm256_madd_epi16(a_u16x16_low, b_u16x16_low));
    sum_u32x8_high = _mm256_add_epi32(sum_u32x8_high, _mm256_madd_epi16(a_u16x16_high, b_u16x16_high));

    state->sum_u32x8_low = sum_u32x8_low;
    state->sum_u32x8_high = sum_u32x8_high;
}

NK_INTERNAL void nk_dot_u8x32_finalize_haswell(                                               //
    nk_dot_u8x32_state_haswell_t const *state_a, nk_dot_u8x32_state_haswell_t const *state_b, //
    nk_dot_u8x32_state_haswell_t const *state_c, nk_dot_u8x32_state_haswell_t const *state_d, //
    nk_u32_t *results) {
    // State is layout-compatible with i8x32 (both contain sum_*_low and sum_*_high)
    // Result storage is also compatible (same bit pattern, different signedness interpretation)
    nk_dot_i8x32_finalize_haswell(                                                                    //
        (nk_dot_i8x32_state_haswell_t const *)state_a, (nk_dot_i8x32_state_haswell_t const *)state_b, //
        (nk_dot_i8x32_state_haswell_t const *)state_c, (nk_dot_i8x32_state_haswell_t const *)state_d,
        (nk_i32_t *)results);
}

/** @brief Type-agnostic 256-bit full load (Haswell AVX2). */
NK_INTERNAL void nk_load_b256_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = _mm256_loadu_si256((const __m256i *)src);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_DOT_HASWELL_H