/**
 *  @brief SIMD-accelerated horizontal reduction operations for Intel Haswell CPUs.
 *  @file include/numkong/reduce/haswell.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_HASWELL_H
#define NK_REDUCE_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_popcount_b8`

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Type-agnostic partial load for 32-bit elements (8 elements max) into 256-bit vector (Haswell AVX2). */
NK_INTERNAL void nk_partial_load_b32x8_haswell_(void const *src, nk_size_t n, nk_b256_vec_t *dst) {
    nk_u32_t const *s = (nk_u32_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (nk_size_t i = 0; i < n && i < 8; ++i) dst->u32s[i] = s[i];
}

/** @brief Type-agnostic partial load for 16-bit elements (16 elements max) into 256-bit vector (Haswell AVX2). */
NK_INTERNAL void nk_partial_load_b16x16_haswell_(void const *src, nk_size_t n, nk_b256_vec_t *dst) {
    nk_u16_t const *s = (nk_u16_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (nk_size_t i = 0; i < n && i < 16; ++i) dst->u16s[i] = s[i];
}

/** @brief Type-agnostic partial load for 8-bit elements (32 elements max) into 256-bit vector (Haswell AVX2). */
NK_INTERNAL void nk_partial_load_b8x32_haswell_(void const *src, nk_size_t n, nk_b256_vec_t *dst) {
    nk_u8_t const *s = (nk_u8_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (nk_size_t i = 0; i < n && i < 32; ++i) dst->u8s[i] = s[i];
}

/** @brief Type-agnostic partial store for 32-bit elements (8 elements max) from 256-bit vector (Haswell AVX2). */
NK_INTERNAL void nk_partial_store_b32x8_haswell_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u32_t *d = (nk_u32_t *)dst;
    for (nk_size_t i = 0; i < n && i < 8; ++i) d[i] = src->u32s[i];
}

/** @brief Type-agnostic partial load for 64-bit elements (4 elements max) into 256-bit vector (Haswell AVX2). */
NK_INTERNAL void nk_partial_load_b64x4_haswell_(void const *src, nk_size_t n, nk_b256_vec_t *dst) {
    nk_u64_t const *s = (nk_u64_t const *)src;
    dst->ymm = _mm256_setzero_si256();
    for (nk_size_t i = 0; i < n && i < 4; ++i) dst->u64s[i] = s[i];
}

/** @brief Type-agnostic partial store for 64-bit elements (4 elements max) from 256-bit vector (Haswell AVX2). */
NK_INTERNAL void nk_partial_store_b64x4_haswell_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    nk_u64_t *d = (nk_u64_t *)dst;
    for (nk_size_t i = 0; i < n && i < 4; ++i) d[i] = src->u64s[i];
}

/** @brief Partial load for f16 elements (up to 8) with conversion to f32 via F16C. */
NK_INTERNAL __m256 nk_partial_load_f16x8_to_f32x8_haswell_(nk_f16_t const *src, nk_size_t n) {
    nk_b256_vec_t vec;
    nk_partial_load_b16x16_haswell_(src, n, &vec);
    return _mm256_cvtph_ps(vec.xmms[0]);
}

/** @brief Convert 8x bf16 to 8x f32 by shifting left 16 bits (AVX2). */
NK_INTERNAL __m256 nk_bf16x8_to_f32x8_haswell_(__m128i bf16_i16x8) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_i16x8), 16));
}

/** @brief Partial load for bf16 elements (up to 8) with conversion to f32. */
NK_INTERNAL __m256 nk_partial_load_bf16x8_to_f32x8_haswell_(nk_bf16_t const *src, nk_size_t n) {
    nk_b256_vec_t vec;
    nk_partial_load_b16x16_haswell_(src, n, &vec);
    return nk_bf16x8_to_f32x8_haswell_(vec.xmms[0]);
}

/** @brief Convert 8x e4m3 to 8x f32 via bit manipulation (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). F32: sign<<31, (exp+120)<<23, mant<<20. */
NK_INTERNAL __m256 nk_e4m3x8_to_f32x8_haswell_(__m128i e4m3_i8x8) {
    __m256i v_i32x8 = _mm256_cvtepu8_epi32(e4m3_i8x8);
    __m256i sign_i32x8 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(v_i32x8, 7), _mm256_set1_epi32(1)), 31);
    __m256i exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(v_i32x8, 3), _mm256_set1_epi32(0x0F));
    __m256i mant_i32x8 = _mm256_and_si256(v_i32x8, _mm256_set1_epi32(0x07));
    __m256i f32_exp_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exp_i32x8, _mm256_set1_epi32(120)), 23);
    __m256i f32_mant_i32x8 = _mm256_slli_epi32(mant_i32x8, 20);
    __m256i f32_bits_i32x8 = _mm256_or_si256(sign_i32x8, _mm256_or_si256(f32_exp_i32x8, f32_mant_i32x8));
    __m256i zero_mask_i32x8 = _mm256_cmpeq_epi32(exp_i32x8, _mm256_setzero_si256());
    f32_bits_i32x8 = _mm256_andnot_si256(zero_mask_i32x8, f32_bits_i32x8);
    return _mm256_castsi256_ps(f32_bits_i32x8);
}

/** @brief Partial load for e4m3 elements (up to 8) with conversion to f32. */
NK_INTERNAL __m256 nk_partial_load_e4m3x8_to_f32x8_haswell_(nk_e4m3_t const *src, nk_size_t n) {
    nk_b256_vec_t vec;
    vec.ymm = _mm256_setzero_si256();
    for (nk_size_t i = 0; i < n && i < 8; ++i) vec.e4m3s[i] = src[i];
    return nk_e4m3x8_to_f32x8_haswell_(vec.xmms[0]);
}

/** @brief Convert 8x e5m2 to 8x f32 via bit manipulation (AVX2).
 *  E5M2 format: S EEEEE MM (bias=15). F32: sign<<31, (exp+112)<<23, mant<<21. */
NK_INTERNAL __m256 nk_e5m2x8_to_f32x8_haswell_(__m128i e5m2_i8x8) {
    __m256i v_i32x8 = _mm256_cvtepu8_epi32(e5m2_i8x8);
    __m256i sign_i32x8 = _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(v_i32x8, 7), _mm256_set1_epi32(1)), 31);
    __m256i exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(v_i32x8, 2), _mm256_set1_epi32(0x1F));
    __m256i mant_i32x8 = _mm256_and_si256(v_i32x8, _mm256_set1_epi32(0x03));
    __m256i f32_exp_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exp_i32x8, _mm256_set1_epi32(112)), 23);
    __m256i f32_mant_i32x8 = _mm256_slli_epi32(mant_i32x8, 21);
    __m256i f32_bits_i32x8 = _mm256_or_si256(sign_i32x8, _mm256_or_si256(f32_exp_i32x8, f32_mant_i32x8));
    __m256i zero_mask_i32x8 = _mm256_cmpeq_epi32(exp_i32x8, _mm256_setzero_si256());
    f32_bits_i32x8 = _mm256_andnot_si256(zero_mask_i32x8, f32_bits_i32x8);
    return _mm256_castsi256_ps(f32_bits_i32x8);
}

/** @brief Partial load for e5m2 elements (up to 8) with conversion to f32. */
NK_INTERNAL __m256 nk_partial_load_e5m2x8_to_f32x8_haswell_(nk_e5m2_t const *src, nk_size_t n) {
    nk_b256_vec_t vec;
    vec.ymm = _mm256_setzero_si256();
    for (nk_size_t i = 0; i < n && i < 8; ++i) vec.e5m2s[i] = src[i];
    return nk_e5m2x8_to_f32x8_haswell_(vec.xmms[0]);
}

/** @brief Horizontal sum of 4 doubles in a YMM register. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x4_haswell_(__m256d sum_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(sum_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(sum_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(lo_f64x2, hi_f64x2);
    sum_f64x2 = _mm_hadd_pd(sum_f64x2, sum_f64x2);
    return _mm_cvtsd_f64(sum_f64x2);
}

/** @brief Horizontal sum of 8 floats in a YMM register (native f32 precision). */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x8_haswell_(__m256 sum_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(sum_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(sum_f32x8, 1);
    __m128 sum_f32x4 = _mm_add_ps(lo_f32x4, hi_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    return _mm_cvtss_f32(sum_f32x4);
}

/** @brief Horizontal sum of 8 i32s in a YMM register. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x8_haswell_(__m256i sum_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(sum_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(sum_i32x8, 1);
    __m128i sum_i32x4 = _mm_add_epi32(lo_i32x4, hi_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    return _mm_cvtsi128_si32(sum_i32x4);
}

/** @brief Horizontal sum of 4 i64s in a YMM register. */
NK_INTERNAL nk_i64_t nk_reduce_add_i64x4_haswell_(__m256i sum_i64x4) {
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(sum_i64x2, sum_i64x2);
    __m128i final_i64 = _mm_add_epi64(sum_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal min of 8 signed i8s in a YMM register. */
NK_INTERNAL nk_i8_t nk_reduce_min_i8x32_haswell_(__m256i min_i8x32) {
    __m128i lo_i8x16 = _mm256_castsi256_si128(min_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(min_i8x32, 1);
    __m128i min_i8x16 = _mm_min_epi8(lo_i8x16, hi_i8x16);
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shufflelo_epi16(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_epi16(min_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(min_i8x16);
}

/** @brief Horizontal max of 8 signed i8s in a YMM register. */
NK_INTERNAL nk_i8_t nk_reduce_max_i8x32_haswell_(__m256i max_i8x32) {
    __m128i lo_i8x16 = _mm256_castsi256_si128(max_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(max_i8x32, 1);
    __m128i max_i8x16 = _mm_max_epi8(lo_i8x16, hi_i8x16);
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shufflelo_epi16(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_epi16(max_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(max_i8x16);
}

/** @brief Horizontal min of 8 unsigned u8s in a YMM register. */
NK_INTERNAL nk_u8_t nk_reduce_min_u8x32_haswell_(__m256i min_u8x32) {
    __m128i lo_u8x16 = _mm256_castsi256_si128(min_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(min_u8x32, 1);
    __m128i min_u8x16 = _mm_min_epu8(lo_u8x16, hi_u8x16);
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shufflelo_epi16(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_epi16(min_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(min_u8x16);
}

/** @brief Horizontal max of 8 unsigned u8s in a YMM register. */
NK_INTERNAL nk_u8_t nk_reduce_max_u8x32_haswell_(__m256i max_u8x32) {
    __m128i lo_u8x16 = _mm256_castsi256_si128(max_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(max_u8x32, 1);
    __m128i max_u8x16 = _mm_max_epu8(lo_u8x16, hi_u8x16);
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shufflelo_epi16(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_epi16(max_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(max_u8x16);
}

/** @brief Horizontal min of 16 signed i16s in a YMM register. */
NK_INTERNAL nk_i16_t nk_reduce_min_i16x16_haswell_(__m256i min_i16x16) {
    __m128i lo_i16x8 = _mm256_castsi256_si128(min_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(min_i16x16, 1);
    __m128i min_i16x8 = _mm_min_epi16(lo_i16x8, hi_i16x8);
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shufflelo_epi16(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(min_i16x8);
}

/** @brief Horizontal max of 16 signed i16s in a YMM register. */
NK_INTERNAL nk_i16_t nk_reduce_max_i16x16_haswell_(__m256i max_i16x16) {
    __m128i lo_i16x8 = _mm256_castsi256_si128(max_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(max_i16x16, 1);
    __m128i max_i16x8 = _mm_max_epi16(lo_i16x8, hi_i16x8);
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shufflelo_epi16(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(max_i16x8);
}

/** @brief Horizontal min of 16 unsigned u16s in a YMM register. */
NK_INTERNAL nk_u16_t nk_reduce_min_u16x16_haswell_(__m256i min_u16x16) {
    __m128i lo_u16x8 = _mm256_castsi256_si128(min_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(min_u16x16, 1);
    __m128i min_u16x8 = _mm_min_epu16(lo_u16x8, hi_u16x8);
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shufflelo_epi16(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(min_u16x8);
}

/** @brief Horizontal max of 16 unsigned u16s in a YMM register. */
NK_INTERNAL nk_u16_t nk_reduce_max_u16x16_haswell_(__m256i max_u16x16) {
    __m128i lo_u16x8 = _mm256_castsi256_si128(max_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(max_u16x16, 1);
    __m128i max_u16x8 = _mm_max_epu16(lo_u16x8, hi_u16x8);
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shufflelo_epi16(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(max_u16x8);
}

/** @brief Horizontal min of 8 signed i32s in a YMM register. */
NK_INTERNAL nk_i32_t nk_reduce_min_i32x8_haswell_(__m256i min_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(min_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(min_i32x8, 1);
    __m128i min_i32x4 = _mm_min_epi32(lo_i32x4, hi_i32x4);
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(min_i32x4);
}

/** @brief Horizontal max of 8 signed i32s in a YMM register. */
NK_INTERNAL nk_i32_t nk_reduce_max_i32x8_haswell_(__m256i max_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(max_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(max_i32x8, 1);
    __m128i max_i32x4 = _mm_max_epi32(lo_i32x4, hi_i32x4);
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(max_i32x4);
}

/** @brief Horizontal min of 8 unsigned u32s in a YMM register. */
NK_INTERNAL nk_u32_t nk_reduce_min_u32x8_haswell_(__m256i min_u32x8) {
    __m128i lo_u32x4 = _mm256_castsi256_si128(min_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(min_u32x8, 1);
    __m128i min_u32x4 = _mm_min_epu32(lo_u32x4, hi_u32x4);
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(min_u32x4);
}

/** @brief Horizontal max of 8 unsigned u32s in a YMM register. */
NK_INTERNAL nk_u32_t nk_reduce_max_u32x8_haswell_(__m256i max_u32x8) {
    __m128i lo_u32x4 = _mm256_castsi256_si128(max_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(max_u32x8, 1);
    __m128i max_u32x4 = _mm_max_epu32(lo_u32x4, hi_u32x4);
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(max_u32x4);
}

/** @brief Horizontal min of 4 signed i64s in a YMM register using comparison+blend. */
NK_INTERNAL nk_i64_t nk_reduce_min_i64x4_haswell_(__m256i min_i64x4) {
    __m128i lo_i64x2 = _mm256_castsi256_si128(min_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(lo_i64x2, hi_i64x2);
    __m128i min_i64x2 = _mm_blendv_epi8(lo_i64x2, hi_i64x2, cmp_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(min_i64x2, min_i64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(min_i64x2, hi_lane_i64);
    __m128i result_i64 = _mm_blendv_epi8(min_i64x2, hi_lane_i64, cmp_final);
    return _mm_cvtsi128_si64(result_i64);
}

/** @brief Horizontal max of 4 signed i64s in a YMM register using comparison+blend. */
NK_INTERNAL nk_i64_t nk_reduce_max_i64x4_haswell_(__m256i max_i64x4) {
    __m128i lo_i64x2 = _mm256_castsi256_si128(max_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(lo_i64x2, hi_i64x2);
    __m128i max_i64x2 = _mm_blendv_epi8(hi_i64x2, lo_i64x2, cmp_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(max_i64x2, max_i64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(max_i64x2, hi_lane_i64);
    __m128i result_i64 = _mm_blendv_epi8(hi_lane_i64, max_i64x2, cmp_final);
    return _mm_cvtsi128_si64(result_i64);
}

/** @brief Horizontal min of 4 unsigned u64s in a YMM register using XOR trick for unsigned comparison. */
NK_INTERNAL nk_u64_t nk_reduce_min_u64x4_haswell_(__m256i min_u64x4) {
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    __m128i lo_u64x2 = _mm256_castsi256_si128(min_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(_mm_xor_si128(lo_u64x2, sign_bit_i64), _mm_xor_si128(hi_u64x2, sign_bit_i64));
    __m128i min_u64x2 = _mm_blendv_epi8(lo_u64x2, hi_u64x2, cmp_i64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(min_u64x2, min_u64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(_mm_xor_si128(min_u64x2, sign_bit_i64),
                                        _mm_xor_si128(hi_lane_u64, sign_bit_i64));
    __m128i result_u64 = _mm_blendv_epi8(min_u64x2, hi_lane_u64, cmp_final);
    return (nk_u64_t)_mm_cvtsi128_si64(result_u64);
}

/** @brief Horizontal max of 4 unsigned u64s in a YMM register using XOR trick for unsigned comparison. */
NK_INTERNAL nk_u64_t nk_reduce_max_u64x4_haswell_(__m256i max_u64x4) {
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    __m128i lo_u64x2 = _mm256_castsi256_si128(max_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(_mm_xor_si128(lo_u64x2, sign_bit_i64), _mm_xor_si128(hi_u64x2, sign_bit_i64));
    __m128i max_u64x2 = _mm_blendv_epi8(hi_u64x2, lo_u64x2, cmp_i64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(max_u64x2, max_u64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(_mm_xor_si128(max_u64x2, sign_bit_i64),
                                        _mm_xor_si128(hi_lane_u64, sign_bit_i64));
    __m128i result_u64 = _mm_blendv_epi8(hi_lane_u64, max_u64x2, cmp_final);
    return (nk_u64_t)_mm_cvtsi128_si64(result_u64);
}

/** @brief Horizontal min of 8 floats in a YMM register. */
NK_INTERNAL nk_f32_t nk_reduce_min_f32x8_haswell_(__m256 min_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(min_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(min_f32x8, 1);
    __m128 min_f32x4 = _mm_min_ps(lo_f32x4, hi_f32x4);
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(min_f32x4);
}

/** @brief Horizontal max of 8 floats in a YMM register. */
NK_INTERNAL nk_f32_t nk_reduce_max_f32x8_haswell_(__m256 max_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(max_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(max_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(lo_f32x4, hi_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max_f32x4);
}

/** @brief Horizontal min of 4 doubles in a YMM register. */
NK_INTERNAL nk_f64_t nk_reduce_min_f64x4_haswell_(__m256d min_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(min_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(min_f64x4, 1);
    __m128d min_f64x2 = _mm_min_pd(lo_f64x2, hi_f64x2);
    min_f64x2 = _mm_min_pd(min_f64x2, _mm_shuffle_pd(min_f64x2, min_f64x2, 1));
    return _mm_cvtsd_f64(min_f64x2);
}

/** @brief Horizontal max of 4 doubles in a YMM register. */
NK_INTERNAL nk_f64_t nk_reduce_max_f64x4_haswell_(__m256d max_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(max_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(max_f64x4, 1);
    __m128d max_f64x2 = _mm_max_pd(lo_f64x2, hi_f64x2);
    max_f64x2 = _mm_max_pd(max_f64x2, _mm_shuffle_pd(max_f64x2, max_f64x2, 1));
    return _mm_cvtsd_f64(max_f64x2);
}

/**
 *  @brief Returns AVX2 blend mask for strided access of 32-bit elements (8-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Returns -1 (all bits set) for positions to keep, 0 for positions to blend away.
 *  Use with _mm256_blendv_ps(identity, data, mask) where identity is 0/+inf/-inf.
 */
NK_INTERNAL __m256i nk_stride_blend_b32x8_(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0); // 4 elems
    case 3: return _mm256_setr_epi32(-1, 0, 0, -1, 0, 0, -1, 0);  // 3 elems
    case 4: return _mm256_setr_epi32(-1, 0, 0, 0, -1, 0, 0, 0);   // 2 elems
    case 5: return _mm256_setr_epi32(-1, 0, 0, 0, 0, -1, 0, 0);   // 2 elems
    case 6: return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, -1, 0);   // 2 elems
    case 7: return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, -1);   // 2 elems
    case 8: return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0);    // 1 elem
    default: return _mm256_setzero_si256();
    }
}

/**
 *  @brief Returns AVX2 blend mask for strided access of 64-bit elements (4-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Returns -1 (all bits set) for positions to keep, 0 for positions to blend away.
 *  Use with _mm256_blendv_pd(identity, data, mask) where identity is 0/+inf/-inf.
 */
NK_INTERNAL __m256i nk_stride_blend_b64x4_(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm256_setr_epi64x(-1, 0, -1, 0); // 2 elems
    case 3: return _mm256_setr_epi64x(-1, 0, 0, -1); // 2 elems (wraps)
    case 4: return _mm256_setr_epi64x(-1, 0, 0, 0);  // 1 elem
    default: return _mm256_setr_epi64x(-1, 0, 0, 0); // 1 elem for stride 5+
    }
}

NK_INTERNAL void nk_reduce_add_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count, nk_f64_t *result) {
    // Accumulate in f64 for precision
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        __m128 lo_f32x4 = _mm256_castps256_ps128(data_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(data_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f32_haswell_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Blend-based strided access: load full register, blend with zero, sum all
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m256 zero_f32x8 = _mm256_setzero_ps();
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        // Blend: keep stride elements, replace others with zero
        __m256 masked_f32x8 = _mm256_blendv_ps(zero_f32x8, data_f32x8, _mm256_castsi256_ps(blend_mask_i32x8));
        // Sum all - zeros don't contribute
        __m128 lo_f32x4 = _mm256_castps256_ps128(masked_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(masked_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }

    // Scalar tail
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    nk_f32_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f32_haswell_gather_(                //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 gathered_f32x8 = _mm256_i32gather_ps(data + idx_scalars * stride_elements, indices_i32x8,
                                                    sizeof(nk_f32_t));
        __m128 lo_f32x4 = _mm256_castps256_ps128(gathered_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(gathered_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(nk_f32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f32_haswell(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_add_f32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f32_haswell_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_f32_haswell_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f32_haswell_gather_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, data_f64x4);
    }
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f64_haswell_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Blend-based strided access: load full register, blend with zero, sum all
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d zero_f64x4 = _mm256_setzero_pd();
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with zero
        __m256d masked_f64x4 = _mm256_blendv_pd(zero_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, masked_f64x4);
    }

    // Scalar tail
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(sum_f64x4);
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_add_f64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f64_haswell_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_f64_haswell_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // First pass: find minimum value
    __m256 min_f32x8 = _mm256_loadu_ps(data);
    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] < min_val) min_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != min_val) continue;
        *min_value = min_val;
        *min_index = idx_scalars;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_min_f32_haswell_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Blend-based strided access: load full register, blend with +inf, find min
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m256 pos_inf_f32x8 = _mm256_set1_ps(__builtin_huge_valf());
    __m256 min_f32x8 = pos_inf_f32x8;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        // Blend: keep stride elements, replace others with +inf
        __m256 masked_f32x8 = _mm256_blendv_ps(pos_inf_f32x8, data_f32x8, _mm256_castsi256_ps(blend_mask_i32x8));
        min_f32x8 = _mm256_min_ps(min_f32x8, masked_f32x8);
    }

    // Scalar tail + horizontal reduce
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    nk_f32_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr < min_val) min_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != min_val) continue;
        *min_value = min_val;
        *min_index = i;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_f32_haswell(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_min_f32_haswell_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_min_f32_haswell_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    __m256 max_f32x8 = _mm256_loadu_ps(data);
    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] > max_val) max_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != max_val) continue;
        *max_value = max_val;
        *max_index = idx_scalars;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_max_f32_haswell_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Blend-based strided access: load full register, blend with -inf, find max
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m256 neg_inf_f32x8 = _mm256_set1_ps(-__builtin_huge_valf());
    __m256 max_f32x8 = neg_inf_f32x8;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        // Blend: keep stride elements, replace others with -inf
        __m256 masked_f32x8 = _mm256_blendv_ps(neg_inf_f32x8, data_f32x8, _mm256_castsi256_ps(blend_mask_i32x8));
        max_f32x8 = _mm256_max_ps(max_f32x8, masked_f32x8);
    }

    // Scalar tail + horizontal reduce
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    nk_f32_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr > max_val) max_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != max_val) continue;
        *max_value = max_val;
        *max_index = i;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_f32_haswell(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_max_f32_haswell_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_max_f32_haswell_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *min_value, nk_size_t *min_index) {
    __m256d min_f64x4 = _mm256_loadu_pd(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        min_f64x4 = _mm256_min_pd(min_f64x4, data_f64x4);
    }
    nk_f64_t min_val = nk_reduce_min_f64x4_haswell_(min_f64x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] < min_val) min_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != min_val) continue;
        *min_value = min_val;
        *min_index = idx_scalars;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_min_f64_haswell_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Blend-based strided access: load full register, blend with +inf, find min
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d pos_inf_f64x4 = _mm256_set1_pd(__builtin_huge_val());
    __m256d min_f64x4 = pos_inf_f64x4;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with +inf
        __m256d masked_f64x4 = _mm256_blendv_pd(pos_inf_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        min_f64x4 = _mm256_min_pd(min_f64x4, masked_f64x4);
    }

    // Scalar tail + horizontal reduce
    nk_f64_t min_val = nk_reduce_min_f64x4_haswell_(min_f64x4);
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr < min_val) min_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != min_val) continue;
        *min_value = min_val;
        *min_index = i;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4)
        nk_reduce_min_f64_haswell_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_f64_haswell_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *max_value, nk_size_t *max_index) {
    __m256d max_f64x4 = _mm256_loadu_pd(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        max_f64x4 = _mm256_max_pd(max_f64x4, data_f64x4);
    }
    nk_f64_t max_val = nk_reduce_max_f64x4_haswell_(max_f64x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] > max_val) max_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != max_val) continue;
        *max_value = max_val;
        *max_index = idx_scalars;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_max_f64_haswell_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Blend-based strided access: load full register, blend with -inf, find max
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d neg_inf_f64x4 = _mm256_set1_pd(-__builtin_huge_val());
    __m256d max_f64x4 = neg_inf_f64x4;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with -inf
        __m256d masked_f64x4 = _mm256_blendv_pd(neg_inf_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        max_f64x4 = _mm256_max_pd(max_f64x4, masked_f64x4);
    }

    // Scalar tail + horizontal reduce
    nk_f64_t max_val = nk_reduce_max_f64x4_haswell_(max_f64x4);
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr > max_val) max_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != max_val) continue;
        *max_value = max_val;
        *max_index = i;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4)
        nk_reduce_max_f64_haswell_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_f64_haswell_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i8_haswell_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // Widen lower 16 bytes: i8 -> i16 -> i32 -> i64
        __m128i lo_i8x16 = _mm256_castsi256_si128(data_i8x32);
        __m256i lo_i16x16 = _mm256_cvtepi8_epi16(lo_i8x16);
        __m128i lo_lo_i16x8 = _mm256_castsi256_si128(lo_i16x16);
        __m128i lo_hi_i16x8 = _mm256_extracti128_si256(lo_i16x16, 1);
        __m256i lo_lo_i32x8 = _mm256_cvtepi16_epi32(lo_lo_i16x8);
        __m256i lo_hi_i32x8 = _mm256_cvtepi16_epi32(lo_hi_i16x8);
        __m128i a_i32x4 = _mm256_castsi256_si128(lo_lo_i32x8);
        __m128i b_i32x4 = _mm256_extracti128_si256(lo_lo_i32x8, 1);
        __m128i c_i32x4 = _mm256_castsi256_si128(lo_hi_i32x8);
        __m128i d_i32x4 = _mm256_extracti128_si256(lo_hi_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(a_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(b_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(c_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(d_i32x4));
        // Widen upper 16 bytes
        __m128i hi_i8x16 = _mm256_extracti128_si256(data_i8x32, 1);
        __m256i hi_i16x16 = _mm256_cvtepi8_epi16(hi_i8x16);
        __m128i hi_lo_i16x8 = _mm256_castsi256_si128(hi_i16x16);
        __m128i hi_hi_i16x8 = _mm256_extracti128_si256(hi_i16x16, 1);
        __m256i hi_lo_i32x8 = _mm256_cvtepi16_epi32(hi_lo_i16x8);
        __m256i hi_hi_i32x8 = _mm256_cvtepi16_epi32(hi_hi_i16x8);
        __m128i e_i32x4 = _mm256_castsi256_si128(hi_lo_i32x8);
        __m128i f_i32x4 = _mm256_extracti128_si256(hi_lo_i32x8, 1);
        __m128i g_i32x4 = _mm256_castsi256_si128(hi_hi_i32x8);
        __m128i h_i32x4 = _mm256_extracti128_si256(hi_hi_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(e_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(f_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(g_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(h_i32x4));
    }
    nk_i64_t sum = nk_reduce_add_i64x4_haswell_(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u8_haswell_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_u8x16 = _mm256_castsi256_si128(data_u8x32);
        __m256i lo_u16x16 = _mm256_cvtepu8_epi16(lo_u8x16);
        __m128i lo_lo_u16x8 = _mm256_castsi256_si128(lo_u16x16);
        __m128i lo_hi_u16x8 = _mm256_extracti128_si256(lo_u16x16, 1);
        __m256i lo_lo_u32x8 = _mm256_cvtepu16_epi32(lo_lo_u16x8);
        __m256i lo_hi_u32x8 = _mm256_cvtepu16_epi32(lo_hi_u16x8);
        __m128i a_u32x4 = _mm256_castsi256_si128(lo_lo_u32x8);
        __m128i b_u32x4 = _mm256_extracti128_si256(lo_lo_u32x8, 1);
        __m128i c_u32x4 = _mm256_castsi256_si128(lo_hi_u32x8);
        __m128i d_u32x4 = _mm256_extracti128_si256(lo_hi_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(a_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(b_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(c_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(d_u32x4));
        __m128i hi_u8x16 = _mm256_extracti128_si256(data_u8x32, 1);
        __m256i hi_u16x16 = _mm256_cvtepu8_epi16(hi_u8x16);
        __m128i hi_lo_u16x8 = _mm256_castsi256_si128(hi_u16x16);
        __m128i hi_hi_u16x8 = _mm256_extracti128_si256(hi_u16x16, 1);
        __m256i hi_lo_u32x8 = _mm256_cvtepu16_epi32(hi_lo_u16x8);
        __m256i hi_hi_u32x8 = _mm256_cvtepu16_epi32(hi_hi_u16x8);
        __m128i e_u32x4 = _mm256_castsi256_si128(hi_lo_u32x8);
        __m128i f_u32x4 = _mm256_extracti128_si256(hi_lo_u32x8, 1);
        __m128i g_u32x4 = _mm256_castsi256_si128(hi_hi_u32x8);
        __m128i h_u32x4 = _mm256_extracti128_si256(hi_hi_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(e_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(f_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(g_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(h_u32x4));
    }
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i16_haswell_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_i16x8 = _mm256_castsi256_si128(data_i16x16);
        __m128i hi_i16x8 = _mm256_extracti128_si256(data_i16x16, 1);
        __m256i lo_i32x8 = _mm256_cvtepi16_epi32(lo_i16x8);
        __m256i hi_i32x8 = _mm256_cvtepi16_epi32(hi_i16x8);
        __m128i a_i32x4 = _mm256_castsi256_si128(lo_i32x8);
        __m128i b_i32x4 = _mm256_extracti128_si256(lo_i32x8, 1);
        __m128i c_i32x4 = _mm256_castsi256_si128(hi_i32x8);
        __m128i d_i32x4 = _mm256_extracti128_si256(hi_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(a_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(b_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(c_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(d_i32x4));
    }
    nk_i64_t sum = nk_reduce_add_i64x4_haswell_(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u16_haswell_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_u16x8 = _mm256_castsi256_si128(data_u16x16);
        __m128i hi_u16x8 = _mm256_extracti128_si256(data_u16x16, 1);
        __m256i lo_u32x8 = _mm256_cvtepu16_epi32(lo_u16x8);
        __m256i hi_u32x8 = _mm256_cvtepu16_epi32(hi_u16x8);
        __m128i a_u32x4 = _mm256_castsi256_si128(lo_u32x8);
        __m128i b_u32x4 = _mm256_extracti128_si256(lo_u32x8, 1);
        __m128i c_u32x4 = _mm256_castsi256_si128(hi_u32x8);
        __m128i d_u32x4 = _mm256_extracti128_si256(hi_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(a_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(b_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(c_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(d_u32x4));
    }
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i32_haswell_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_i32x4 = _mm256_castsi256_si128(data_i32x8);
        __m128i hi_i32x4 = _mm256_extracti128_si256(data_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(lo_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(hi_i32x4));
    }
    nk_i64_t sum = nk_reduce_add_i64x4_haswell_(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u32_haswell_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_u32x4 = _mm256_castsi256_si128(data_u32x8);
        __m128i hi_u32x4 = _mm256_extracti128_si256(data_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(lo_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(hi_u32x4));
    }
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i64_haswell_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, data_i64x4);
    }
    nk_i64_t sum = nk_reduce_add_i64x4_haswell_(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u64_haswell_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, data_u64x4);
    }
    nk_u64_t sum = (nk_u64_t)nk_reduce_add_i64x4_haswell_(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_min_i8_haswell_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *min_value, nk_size_t *min_index) {
    __m256i min_i8x32 = _mm256_set1_epi8(127);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_i8x32 = _mm256_min_epi8(min_i8x32, data_i8x32);
    }
    nk_i8_t min_val = nk_reduce_min_i8x32_haswell_(min_i8x32);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    // Second pass for index
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_i8_haswell_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *max_value, nk_size_t *max_index) {
    __m256i max_i8x32 = _mm256_set1_epi8(-128);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_i8x32 = _mm256_max_epi8(max_i8x32, data_i8x32);
    }
    nk_i8_t max_val = nk_reduce_max_i8x32_haswell_(max_i8x32);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_u8_haswell_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *min_value, nk_size_t *min_index) {
    __m256i min_u8x32 = _mm256_set1_epi8((char)255);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_u8x32 = _mm256_min_epu8(min_u8x32, data_u8x32);
    }
    nk_u8_t min_val = nk_reduce_min_u8x32_haswell_(min_u8x32);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_u8_haswell_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *max_value, nk_size_t *max_index) {
    __m256i max_u8x32 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_u8x32 = _mm256_max_epu8(max_u8x32, data_u8x32);
    }
    nk_u8_t max_val = nk_reduce_max_u8x32_haswell_(max_u8x32);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_i16_haswell_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *min_value, nk_size_t *min_index) {
    __m256i min_i16x16 = _mm256_set1_epi16(32767);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_i16x16 = _mm256_min_epi16(min_i16x16, data_i16x16);
    }
    nk_i16_t min_val = nk_reduce_min_i16x16_haswell_(min_i16x16);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_i16_haswell_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *max_value, nk_size_t *max_index) {
    __m256i max_i16x16 = _mm256_set1_epi16(-32768);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_i16x16 = _mm256_max_epi16(max_i16x16, data_i16x16);
    }
    nk_i16_t max_val = nk_reduce_max_i16x16_haswell_(max_i16x16);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_u16_haswell_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *min_value, nk_size_t *min_index) {
    __m256i min_u16x16 = _mm256_set1_epi16((short)65535);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_u16x16 = _mm256_min_epu16(min_u16x16, data_u16x16);
    }
    nk_u16_t min_val = nk_reduce_min_u16x16_haswell_(min_u16x16);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_u16_haswell_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *max_value, nk_size_t *max_index) {
    __m256i max_u16x16 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_u16x16 = _mm256_max_epu16(max_u16x16, data_u16x16);
    }
    nk_u16_t max_val = nk_reduce_max_u16x16_haswell_(max_u16x16);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_i32_haswell_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *min_value, nk_size_t *min_index) {
    __m256i min_i32x8 = _mm256_set1_epi32(0x7FFFFFFF);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_i32x8 = _mm256_min_epi32(min_i32x8, data_i32x8);
    }
    nk_i32_t min_val = nk_reduce_min_i32x8_haswell_(min_i32x8);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_i32_haswell_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *max_value, nk_size_t *max_index) {
    __m256i max_i32x8 = _mm256_set1_epi32((nk_i32_t)0x80000000);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_i32x8 = _mm256_max_epi32(max_i32x8, data_i32x8);
    }
    nk_i32_t max_val = nk_reduce_max_i32x8_haswell_(max_i32x8);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_u32_haswell_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *min_value, nk_size_t *min_index) {
    __m256i min_u32x8 = _mm256_set1_epi32((nk_i32_t)0xFFFFFFFF);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_u32x8 = _mm256_min_epu32(min_u32x8, data_u32x8);
    }
    nk_u32_t min_val = nk_reduce_min_u32x8_haswell_(min_u32x8);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_u32_haswell_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *max_value, nk_size_t *max_index) {
    __m256i max_u32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_u32x8 = _mm256_max_epu32(max_u32x8, data_u32x8);
    }
    nk_u32_t max_val = nk_reduce_max_u32x8_haswell_(max_u32x8);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_i64_haswell_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *min_value, nk_size_t *min_index) {
    __m256i min_i64x4 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // Manual 64-bit signed min using comparison
        __m128i lo_data_i64x2 = _mm256_castsi256_si128(data_i64x4);
        __m128i hi_data_i64x2 = _mm256_extracti128_si256(data_i64x4, 1);
        __m128i lo_min_i64x2 = _mm256_castsi256_si128(min_i64x4);
        __m128i hi_min_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(lo_min_i64x2, lo_data_i64x2);
        __m128i hi_cmp = _mm_cmpgt_epi64(hi_min_i64x2, hi_data_i64x2);
        lo_min_i64x2 = _mm_blendv_epi8(lo_min_i64x2, lo_data_i64x2, lo_cmp);
        hi_min_i64x2 = _mm_blendv_epi8(hi_min_i64x2, hi_data_i64x2, hi_cmp);
        min_i64x4 = _mm256_setr_m128i(lo_min_i64x2, hi_min_i64x2);
    }
    nk_i64_t min_val = nk_reduce_min_i64x4_haswell_(min_i64x4);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_i64_haswell_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *max_value, nk_size_t *max_index) {
    __m256i max_i64x4 = _mm256_set1_epi64x((nk_i64_t)0x8000000000000000LL);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_data_i64x2 = _mm256_castsi256_si128(data_i64x4);
        __m128i hi_data_i64x2 = _mm256_extracti128_si256(data_i64x4, 1);
        __m128i lo_max_i64x2 = _mm256_castsi256_si128(max_i64x4);
        __m128i hi_max_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(lo_data_i64x2, lo_max_i64x2);
        __m128i hi_cmp = _mm_cmpgt_epi64(hi_data_i64x2, hi_max_i64x2);
        lo_max_i64x2 = _mm_blendv_epi8(lo_max_i64x2, lo_data_i64x2, lo_cmp);
        hi_max_i64x2 = _mm_blendv_epi8(hi_max_i64x2, hi_data_i64x2, hi_cmp);
        max_i64x4 = _mm256_setr_m128i(lo_max_i64x2, hi_max_i64x2);
    }
    nk_i64_t max_val = nk_reduce_max_i64x4_haswell_(max_i64x4);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void nk_reduce_min_u64_haswell_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *min_value, nk_size_t *min_index) {
    __m256i min_u64x4 = _mm256_set1_epi64x((nk_i64_t)0xFFFFFFFFFFFFFFFFULL);
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // Unsigned comparison via XOR with sign bit
        __m128i lo_data_u64x2 = _mm256_castsi256_si128(data_u64x4);
        __m128i hi_data_u64x2 = _mm256_extracti128_si256(data_u64x4, 1);
        __m128i lo_min_u64x2 = _mm256_castsi256_si128(min_u64x4);
        __m128i hi_min_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(_mm_xor_si128(lo_min_u64x2, sign_bit_i64),
                                         _mm_xor_si128(lo_data_u64x2, sign_bit_i64));
        __m128i hi_cmp = _mm_cmpgt_epi64(_mm_xor_si128(hi_min_u64x2, sign_bit_i64),
                                         _mm_xor_si128(hi_data_u64x2, sign_bit_i64));
        lo_min_u64x2 = _mm_blendv_epi8(lo_min_u64x2, lo_data_u64x2, lo_cmp);
        hi_min_u64x2 = _mm_blendv_epi8(hi_min_u64x2, hi_data_u64x2, hi_cmp);
        min_u64x4 = _mm256_setr_m128i(lo_min_u64x2, hi_min_u64x2);
    }
    nk_u64_t min_val = nk_reduce_min_u64x4_haswell_(min_u64x4);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void nk_reduce_max_u64_haswell_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *max_value, nk_size_t *max_index) {
    __m256i max_u64x4 = _mm256_setzero_si256();
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_data_u64x2 = _mm256_castsi256_si128(data_u64x4);
        __m128i hi_data_u64x2 = _mm256_extracti128_si256(data_u64x4, 1);
        __m128i lo_max_u64x2 = _mm256_castsi256_si128(max_u64x4);
        __m128i hi_max_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(_mm_xor_si128(lo_data_u64x2, sign_bit_i64),
                                         _mm_xor_si128(lo_max_u64x2, sign_bit_i64));
        __m128i hi_cmp = _mm_cmpgt_epi64(_mm_xor_si128(hi_data_u64x2, sign_bit_i64),
                                         _mm_xor_si128(hi_max_u64x2, sign_bit_i64));
        lo_max_u64x2 = _mm_blendv_epi8(lo_max_u64x2, lo_data_u64x2, lo_cmp);
        hi_max_u64x2 = _mm_blendv_epi8(hi_max_u64x2, hi_data_u64x2, hi_cmp);
        max_u64x4 = _mm256_setr_m128i(lo_max_u64x2, hi_max_u64x2);
    }
    nk_u64_t max_val = nk_reduce_max_u64x4_haswell_(max_u64x4);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_add_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_add_i8_haswell_contiguous_(data, count, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_add_u8_haswell_contiguous_(data, count, result);
    else nk_reduce_add_u8_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_add_i16_haswell_contiguous_(data, count, result);
    else nk_reduce_add_i16_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_add_u16_haswell_contiguous_(data, count, result);
    else nk_reduce_add_u16_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_add_i32_haswell_contiguous_(data, count, result);
    else nk_reduce_add_i32_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_add_u32_haswell_contiguous_(data, count, result);
    else nk_reduce_add_u32_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_add_i64_haswell_contiguous_(data, count, result);
    else nk_reduce_add_i64_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_add_u64_haswell_contiguous_(data, count, result);
    else nk_reduce_add_u64_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_min_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_min_i8_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_max_i8_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_min_u8_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_max_u8_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_min_i16_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_max_i16_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_min_u16_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_max_u16_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_min_i32_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_max_i32_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_min_u32_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_max_u32_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_min_i64_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_max_i64_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_min_u64_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_max_u64_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_REDUCE_HASWELL_H