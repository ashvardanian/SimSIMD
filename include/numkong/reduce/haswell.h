/**
 *  @brief SIMD-accelerated horizontal reduction operations for Intel Haswell CPUs.
 *  @file include/numkong/reduce/haswell.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section haswell_reduce_instructions Key AVX2 Reduction Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_hadd_ps              VHADDPS (YMM, YMM, YMM)         7cy         0.5/cy      p01+p5
 *      _mm256_extractf128_ps       VEXTRACTF128 (XMM, YMM, I8)     3cy         1/cy        p5
 *      _mm_hadd_ps                 HADDPS (XMM, XMM, XMM)          5cy         1/cy        p01+p5
 *      _mm_add_ps                  ADDPS (XMM, XMM, XMM)           3cy         1/cy        p01
 *
 *  Horizontal reductions require cross-lane operations: extract high 128 bits, add to low lane,
 *  then apply 128-bit hadd twice. Total latency is ~11-13 cycles for YMM-to-scalar reduction.
 *  Using dual accumulators in calling code helps hide this finalization latency.
 */
#ifndef NK_REDUCE_HASWELL_H
#define NK_REDUCE_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_reduce_add_f32_serial`
#include "numkong/cast/haswell.h"  // `nk_bf16x8_to_f32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

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
 *  @brief Convert 32× FP8 bytes to unsigned-comparable form for ordering.
 *
 *  Transforms FP8 bit patterns so that unsigned byte comparison (VPMINUB/VPMAXUB)
 *  preserves the correct FP8 numeric ordering across positive and negative values.
 *
 *  @param raw_i8x32 Raw FP8 bytes (E4M3 or E5M2 - same transformation)
 *  @return Transformed bytes suitable for unsigned comparison
 *
 *  @note Same function works for both E4M3 and E5M2 (sign bit position identical)
 *  @note Port usage: 1× VPCMPGTB (p01), 1× VPBLENDVB (p015×2), 1× VPXOR (p015) = ~4 uops
 */
NK_INTERNAL __m256i nk_fp8x32_to_u8x32_comparable_haswell_(__m256i raw_i8x32) {
    // In AVX2, use signed comparison: 0 > x means x < 0 (negative)
    __m256i neg_i8x32 = _mm256_cmpgt_epi8(_mm256_setzero_si256(), raw_i8x32);
    __m256i pos_xor_i8x32 = _mm256_set1_epi8((char)0x80);
    __m256i neg_xor_i8x32 = _mm256_set1_epi8((char)0xFF);
    __m256i xor_i8x32 = _mm256_blendv_epi8(pos_xor_i8x32, neg_xor_i8x32, neg_i8x32);
    return _mm256_xor_si256(raw_i8x32, xor_i8x32);
}

/**
 *  @brief Convert 32× comparable bytes back to FP8 format.
 *
 *  Reverses the transformation applied by nk_fp8x32_to_u8x32_comparable_haswell_.
 *  Values < 0x80 in comparable form were originally negative FP8.
 *
 *  @param cmp_i8x32 Bytes in comparable form
 *  @return Original FP8 bytes (E4M3 or E5M2)
 *
 *  @note Port usage: 1× VPCMPGTB (p01), 1× VPBLENDVB (p015×2), 1× VPXOR (p015) = ~4 uops
 */
NK_INTERNAL __m256i nk_u8x32_comparable_to_fp8x32_haswell_(__m256i cmp_i8x32) {
    // Values < 0x80 were negative FP8, values >= 0x80 were positive
    __m256i threshold_i8x32 = _mm256_set1_epi8((char)0x80);
    __m256i was_neg_i8x32 = _mm256_cmpgt_epi8(threshold_i8x32, cmp_i8x32);
    __m256i neg_xor_i8x32 = _mm256_set1_epi8((char)0xFF);
    __m256i pos_xor_i8x32 = _mm256_set1_epi8((char)0x80);
    __m256i xor_i8x32 = _mm256_blendv_epi8(pos_xor_i8x32, neg_xor_i8x32, was_neg_i8x32);
    return _mm256_xor_si256(cmp_i8x32, xor_i8x32);
}

/**
 *  @brief IEEE-compliant min selection mask for E4M3 vectors in comparable form.
 *
 *  Returns blend mask indicating which lanes should select from 'a' to get
 *  element-wise minimum with IEEE NaN semantics: min(x, NaN) = x, min(NaN, NaN) = NaN.
 *
 *  @param a_cmp_u8x32 First operand in comparable form
 *  @param b_cmp_u8x32 Second operand in comparable form
 *  @param nan_cmp_u8x32 NaN value in comparable form (0xFF for E4M3)
 *  @return Blend mask: 0xFF = select a, 0x00 = select b
 *
 *  Usage: min = _mm256_blendv_epi8(b_cmp, a_cmp, mask);
 *
 *  @note Port usage: 2× VPCMPEQB (p01), 1× VPMINUB (p01), ~4× logic (p015) = ~7 uops
 */
NK_INTERNAL __m256i nk_min_mask_e4m3x32_haswell_( //
    __m256i a_cmp_u8x32, __m256i b_cmp_u8x32, __m256i nan_cmp_u8x32) {
    // Detect NaN: in comparable form, E4M3 NaN (0x7F) maps to 0xFF
    __m256i a_nan_i8x32 = _mm256_cmpeq_epi8(a_cmp_u8x32, nan_cmp_u8x32);
    __m256i b_nan_i8x32 = _mm256_cmpeq_epi8(b_cmp_u8x32, nan_cmp_u8x32);

    // Use min_epu8 then check which one won
    __m256i min_cmp_u8x32 = _mm256_min_epu8(a_cmp_u8x32, b_cmp_u8x32);
    __m256i a_is_min_i8x32 = _mm256_cmpeq_epi8(min_cmp_u8x32, a_cmp_u8x32);

    // Select a if: a is not NaN AND (a <= b OR b is NaN)
    // Note: a_is_min is true when a == min, which includes a == b case
    __m256i not_a_nan = _mm256_xor_si256(a_nan_i8x32, _mm256_set1_epi8((char)0xFF));
    __m256i a_wins_or_b_nan = _mm256_or_si256(a_is_min_i8x32, b_nan_i8x32);
    return _mm256_and_si256(not_a_nan, a_wins_or_b_nan);
}

/**
 *  @brief IEEE-compliant max selection mask for E4M3 vectors in comparable form.
 *
 *  Returns blend mask indicating which lanes should select from 'a' to get
 *  element-wise maximum with IEEE NaN semantics: max(x, NaN) = x, max(NaN, NaN) = NaN.
 *
 *  @param a_cmp_u8x32 First operand in comparable form
 *  @param b_cmp_u8x32 Second operand in comparable form
 *  @param nan_cmp_u8x32 NaN value in comparable form (0xFF for E4M3)
 *  @return Blend mask: 0xFF = select a, 0x00 = select b
 *
 *  Usage: max = _mm256_blendv_epi8(b_cmp, a_cmp, mask);
 */
NK_INTERNAL __m256i nk_max_mask_e4m3x32_haswell_( //
    __m256i a_cmp_u8x32, __m256i b_cmp_u8x32, __m256i nan_cmp_u8x32) {
    __m256i a_nan_i8x32 = _mm256_cmpeq_epi8(a_cmp_u8x32, nan_cmp_u8x32);
    __m256i b_nan_i8x32 = _mm256_cmpeq_epi8(b_cmp_u8x32, nan_cmp_u8x32);

    __m256i max_cmp_u8x32 = _mm256_max_epu8(a_cmp_u8x32, b_cmp_u8x32);
    __m256i a_is_max_i8x32 = _mm256_cmpeq_epi8(max_cmp_u8x32, a_cmp_u8x32);

    // Select a if: a is not NaN AND (a >= b OR b is NaN)
    __m256i not_a_nan = _mm256_xor_si256(a_nan_i8x32, _mm256_set1_epi8((char)0xFF));
    __m256i a_wins_or_b_nan = _mm256_or_si256(a_is_max_i8x32, b_nan_i8x32);
    return _mm256_and_si256(not_a_nan, a_wins_or_b_nan);
}

/**
 *  @brief IEEE-compliant min selection mask for E5M2 vectors in comparable form.
 *
 *  E5M2 has multiple NaN encodings (0x7D-0x7F per sign), requiring threshold
 *  comparison instead of equality. In comparable form, NaNs map to >= 0xFD.
 *
 *  @param a_cmp_u8x32 First operand in comparable form
 *  @param b_cmp_u8x32 Second operand in comparable form
 *  @param nan_threshold_cmp_u8x32 NaN threshold in comparable form (0xFD for E5M2)
 *  @return Blend mask: 0xFF = select a, 0x00 = select b
 */
NK_INTERNAL __m256i nk_min_mask_e5m2x32_haswell_( //
    __m256i a_cmp_u8x32, __m256i b_cmp_u8x32, __m256i nan_threshold_cmp_u8x32) {
    // Detect NaN: max(x, threshold) == x means x >= threshold
    __m256i a_nan_i8x32 = _mm256_cmpeq_epi8(_mm256_max_epu8(a_cmp_u8x32, nan_threshold_cmp_u8x32), a_cmp_u8x32);
    __m256i b_nan_i8x32 = _mm256_cmpeq_epi8(_mm256_max_epu8(b_cmp_u8x32, nan_threshold_cmp_u8x32), b_cmp_u8x32);

    __m256i min_cmp_u8x32 = _mm256_min_epu8(a_cmp_u8x32, b_cmp_u8x32);
    __m256i a_is_min_i8x32 = _mm256_cmpeq_epi8(min_cmp_u8x32, a_cmp_u8x32);

    __m256i not_a_nan = _mm256_xor_si256(a_nan_i8x32, _mm256_set1_epi8((char)0xFF));
    __m256i a_wins_or_b_nan = _mm256_or_si256(a_is_min_i8x32, b_nan_i8x32);
    return _mm256_and_si256(not_a_nan, a_wins_or_b_nan);
}

/**
 *  @brief IEEE-compliant max selection mask for E5M2 vectors in comparable form.
 *
 *  @param a_cmp_u8x32 First operand in comparable form
 *  @param b_cmp_u8x32 Second operand in comparable form
 *  @param nan_threshold_cmp_u8x32 NaN threshold in comparable form (0xFD for E5M2)
 *  @return Blend mask: 0xFF = select a, 0x00 = select b
 */
NK_INTERNAL __m256i nk_max_mask_e5m2x32_haswell_( //
    __m256i a_cmp_u8x32, __m256i b_cmp_u8x32, __m256i nan_threshold_cmp_u8x32) {
    __m256i a_nan_i8x32 = _mm256_cmpeq_epi8(_mm256_max_epu8(a_cmp_u8x32, nan_threshold_cmp_u8x32), a_cmp_u8x32);
    __m256i b_nan_i8x32 = _mm256_cmpeq_epi8(_mm256_max_epu8(b_cmp_u8x32, nan_threshold_cmp_u8x32), b_cmp_u8x32);

    __m256i max_cmp_u8x32 = _mm256_max_epu8(a_cmp_u8x32, b_cmp_u8x32);
    __m256i a_is_max_i8x32 = _mm256_cmpeq_epi8(max_cmp_u8x32, a_cmp_u8x32);

    __m256i not_a_nan = _mm256_xor_si256(a_nan_i8x32, _mm256_set1_epi8((char)0xFF));
    __m256i a_wins_or_b_nan = _mm256_or_si256(a_is_max_i8x32, b_nan_i8x32);
    return _mm256_and_si256(not_a_nan, a_wins_or_b_nan);
}

/** @brief Horizontal argmin: returns index of first minimum unsigned byte in YMM register. */
NK_INTERNAL nk_size_t nk_argmin_u8x32_haswell_(__m256i data_u8x32) {
    nk_u8_t min_val = nk_reduce_min_u8x32_haswell_(data_u8x32);
    __m256i eq_i8x32 = _mm256_cmpeq_epi8(data_u8x32, _mm256_set1_epi8((char)min_val));
    int eq_bits = _mm256_movemask_epi8(eq_i8x32);
    return (nk_size_t)_tzcnt_u32((unsigned int)eq_bits);
}

/** @brief Horizontal argmax: returns index of first maximum unsigned byte in YMM register. */
NK_INTERNAL nk_size_t nk_argmax_u8x32_haswell_(__m256i data_u8x32) {
    nk_u8_t max_val = nk_reduce_max_u8x32_haswell_(data_u8x32);
    __m256i eq_i8x32 = _mm256_cmpeq_epi8(data_u8x32, _mm256_set1_epi8((char)max_val));
    int eq_bits = _mm256_movemask_epi8(eq_i8x32);
    return (nk_size_t)_tzcnt_u32((unsigned int)eq_bits);
}

/**
 *  @brief Returns AVX2 blend mask for strided access of 8-bit elements (32-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Returns -1 (all bits set) for positions to keep, 0 for positions to AND away.
 *  Use with _mm256_and_si256(data, mask) to zero out non-stride positions.
 */
NK_INTERNAL __m256i nk_stride_blend_u1x32_(nk_size_t stride) {
    switch (stride) {
    case 2:
        return _mm256_setr_epi8(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1,
                                0, -1, 0, -1, 0, -1, 0);
    case 3:
        return _mm256_setr_epi8(-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
                                0, -1, 0, 0, -1, 0);
    case 4:
        return _mm256_setr_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                                0, -1, 0, 0, 0);
    case 5:
        return _mm256_setr_epi8(-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0,
                                0, 0, 0, -1, 0);
    case 6:
        return _mm256_setr_epi8(-1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0,
                                0, 0, -1, 0);
    case 7:
        return _mm256_setr_epi8(-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
                                -1, 0, 0, 0);
    case 8:
        return _mm256_setr_epi8(-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0,
                                0, 0, 0, 0);
    default: return _mm256_setzero_si256();
    }
}

/**
 *  @brief Returns AVX2 blend mask for strided access of 16-bit elements (16-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Returns -1 (all bits set) for positions to keep, 0 for positions to AND away.
 *  Use with _mm256_and_si256(data, mask) to zero out non-stride positions.
 */
NK_INTERNAL __m256i nk_stride_blend_b16x16_(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm256_setr_epi16(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
    case 3: return _mm256_setr_epi16(-1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);
    case 4: return _mm256_setr_epi16(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
    case 5: return _mm256_setr_epi16(-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1);
    case 6: return _mm256_setr_epi16(-1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0);
    case 7: return _mm256_setr_epi16(-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0);
    case 8: return _mm256_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0);
    default: return _mm256_setzero_si256();
    }
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
    // Use dual accumulators to hide VADDPD latency (3 cycles on Haswell) and two
    // 128-bit loads instead of 256-bit load + VEXTRACTF128 to reduce Port 5 pressure.
    __m256d sum_low_f64x4 = _mm256_setzero_pd();
    __m256d sum_high_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m128 low_f32x4 = _mm_loadu_ps(data + idx_scalars);
        __m128 high_f32x4 = _mm_loadu_ps(data + idx_scalars + 4);
        sum_low_f64x4 = _mm256_add_pd(sum_low_f64x4, _mm256_cvtps_pd(low_f32x4));
        sum_high_f64x4 = _mm256_add_pd(sum_high_f64x4, _mm256_cvtps_pd(high_f32x4));
    }
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_low_f64x4, sum_high_f64x4));
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f32_haswell_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Blend-based strided access: load chunks, blend with zero, sum all.
    // Use 2×128-bit loads with 128-bit blends to avoid per-iteration VEXTRACTF128.
    // Extract blend mask halves once outside loop, dual accumulators for ILP.
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m128 blend_mask_low_f32x4 = _mm_castsi128_ps(_mm256_castsi256_si128(blend_mask_i32x8));
    __m128 blend_mask_high_f32x4 = _mm_castsi128_ps(_mm256_extracti128_si256(blend_mask_i32x8, 1));
    __m128 zero_f32x4 = _mm_setzero_ps();
    __m256d sum_low_f64x4 = _mm256_setzero_pd();
    __m256d sum_high_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m128 low_f32x4 = _mm_loadu_ps(data + idx_scalars);
        __m128 high_f32x4 = _mm_loadu_ps(data + idx_scalars + 4);
        // Blend: keep stride elements, replace others with zero
        __m128 masked_low_f32x4 = _mm_blendv_ps(zero_f32x4, low_f32x4, blend_mask_low_f32x4);
        __m128 masked_high_f32x4 = _mm_blendv_ps(zero_f32x4, high_f32x4, blend_mask_high_f32x4);
        // Sum all - zeros don't contribute
        sum_low_f64x4 = _mm256_add_pd(sum_low_f64x4, _mm256_cvtps_pd(masked_low_f32x4));
        sum_high_f64x4 = _mm256_add_pd(sum_high_f64x4, _mm256_cvtps_pd(masked_high_f32x4));
    }

    // Scalar tail
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sum_low_f64x4, sum_high_f64x4));
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
    if (count == 0) *result = 0;
    else if (!aligned) nk_reduce_add_f32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f32_haswell_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_f32_haswell_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f32_haswell_gather_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d compensation_f64x4 = _mm256_setzero_pd();
    __m256d absolute_mask_f64x4 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d term_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        __m256d tentative_f64x4 = _mm256_add_pd(sum_f64x4, term_f64x4);
        __m256d absolute_sum_f64x4 = _mm256_and_pd(sum_f64x4, absolute_mask_f64x4);
        __m256d absolute_term_f64x4 = _mm256_and_pd(term_f64x4, absolute_mask_f64x4);
        __m256d sum_bigger_f64x4 = _mm256_cmp_pd(absolute_sum_f64x4, absolute_term_f64x4, _CMP_GE_OQ);
        __m256d correction_sum_bigger_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, tentative_f64x4), term_f64x4);
        __m256d correction_term_bigger_f64x4 = _mm256_add_pd(_mm256_sub_pd(term_f64x4, tentative_f64x4), sum_f64x4);
        __m256d correction_f64x4 = _mm256_blendv_pd(correction_term_bigger_f64x4, correction_sum_bigger_f64x4,
                                                    sum_bigger_f64x4);
        compensation_f64x4 = _mm256_add_pd(compensation_f64x4, correction_f64x4);
        sum_f64x4 = tentative_f64x4;
    }
    __m256d total_f64x4 = _mm256_add_pd(sum_f64x4, compensation_f64x4);
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(total_f64x4);
    nk_f64_t compensation = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        nk_f64_t term = data[idx_scalars], tentative = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - tentative) + term)
                                                                : ((term - tentative) + sum);
        sum = tentative;
    }
    *result = sum + compensation;
}

NK_INTERNAL void nk_reduce_add_f64_haswell_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Blend-based strided access: load full register, blend with zero, sum all
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d zero_f64x4 = _mm256_setzero_pd();
    __m256d sum_f64x4 = _mm256_setzero_pd();
    __m256d compensation_f64x4 = _mm256_setzero_pd();
    __m256d absolute_mask_f64x4 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with zero
        __m256d term_f64x4 = _mm256_blendv_pd(zero_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        __m256d tentative_f64x4 = _mm256_add_pd(sum_f64x4, term_f64x4);
        __m256d absolute_sum_f64x4 = _mm256_and_pd(sum_f64x4, absolute_mask_f64x4);
        __m256d absolute_term_f64x4 = _mm256_and_pd(term_f64x4, absolute_mask_f64x4);
        __m256d sum_bigger_f64x4 = _mm256_cmp_pd(absolute_sum_f64x4, absolute_term_f64x4, _CMP_GE_OQ);
        __m256d correction_sum_bigger_f64x4 = _mm256_add_pd(_mm256_sub_pd(sum_f64x4, tentative_f64x4), term_f64x4);
        __m256d correction_term_bigger_f64x4 = _mm256_add_pd(_mm256_sub_pd(term_f64x4, tentative_f64x4), sum_f64x4);
        __m256d correction_f64x4 = _mm256_blendv_pd(correction_term_bigger_f64x4, correction_sum_bigger_f64x4,
                                                    sum_bigger_f64x4);
        compensation_f64x4 = _mm256_add_pd(compensation_f64x4, correction_f64x4);
        sum_f64x4 = tentative_f64x4;
    }

    // Scalar tail with Neumaier
    __m256d total_f64x4 = _mm256_add_pd(sum_f64x4, compensation_f64x4);
    nk_f64_t sum = nk_reduce_add_f64x4_haswell_(total_f64x4);
    nk_f64_t compensation = 0;
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        nk_f64_t term = *ptr, tentative = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - tentative) + term)
                                                                : ((term - tentative) + sum);
        sum = tentative;
    }
    *result = sum + compensation;
}

NK_PUBLIC void nk_reduce_add_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *result = 0;
    else if (!aligned) nk_reduce_add_f64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f64_haswell_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_f64_haswell_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m256 min_f32x8 = _mm256_loadu_ps(data);
    __m256i min_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i current_idx_i32x8 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    __m256i step_i32x8 = _mm256_set1_epi32(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        __m256 lt_mask = _mm256_cmp_ps(data_f32x8, min_f32x8, _CMP_LT_OQ);
        min_f32x8 = _mm256_blendv_ps(min_f32x8, data_f32x8, lt_mask);
        min_idx_i32x8 = _mm256_blendv_epi8(min_idx_i32x8, current_idx_i32x8, _mm256_castps_si256(lt_mask));
        current_idx_i32x8 = _mm256_add_epi32(current_idx_i32x8, step_i32x8);
    }

    // Handle scalar tail
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    nk_size_t min_idx = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] < min_val) {
            min_val = data[idx_scalars];
            min_idx = idx_scalars;
        }
    }

    // Find winning lane: compare each lane to horizontal min
    __m256 eq_mask = _mm256_cmp_ps(min_f32x8, _mm256_set1_ps(min_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_ps(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i32_t indices[8];
        _mm256_storeu_si256((__m256i *)indices, min_idx_i32x8);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        // Return whichever index came first
        *min_index = (min_idx && min_idx < vec_idx) ? min_idx : vec_idx;
    }
    else { *min_index = min_idx; }
    *min_value = min_val;
}

NK_INTERNAL void nk_reduce_min_f32_haswell_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Blend-based strided access: load full register, blend with +inf, find min
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m256 pos_inf_f32x8 = _mm256_set1_ps(NK_F32_MAX);
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
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_min_f32_haswell_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_min_f32_haswell_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f32_haswell_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m256 max_f32x8 = _mm256_loadu_ps(data);
    __m256i max_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i current_idx_i32x8 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    __m256i step_i32x8 = _mm256_set1_epi32(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        __m256 gt_mask = _mm256_cmp_ps(data_f32x8, max_f32x8, _CMP_GT_OQ);
        max_f32x8 = _mm256_blendv_ps(max_f32x8, data_f32x8, gt_mask);
        max_idx_i32x8 = _mm256_blendv_epi8(max_idx_i32x8, current_idx_i32x8, _mm256_castps_si256(gt_mask));
        current_idx_i32x8 = _mm256_add_epi32(current_idx_i32x8, step_i32x8);
    }

    // Handle scalar tail
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    nk_size_t max_idx = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] > max_val) {
            max_val = data[idx_scalars];
            max_idx = idx_scalars;
        }
    }

    // Find winning lane: compare each lane to horizontal max
    __m256 eq_mask = _mm256_cmp_ps(max_f32x8, _mm256_set1_ps(max_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_ps(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i32_t indices[8];
        _mm256_storeu_si256((__m256i *)indices, max_idx_i32x8);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *max_index = (max_idx && max_idx < vec_idx) ? max_idx : vec_idx;
    }
    else { *max_index = max_idx; }
    *max_value = max_val;
}

NK_INTERNAL void nk_reduce_max_f32_haswell_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Blend-based strided access: load full register, blend with -inf, find max
    __m256i blend_mask_i32x8 = nk_stride_blend_b32x8_(stride_elements);
    __m256 neg_inf_f32x8 = _mm256_set1_ps(NK_F32_MIN);
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
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_max_f32_haswell_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_max_f32_haswell_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m256d min_f64x4 = _mm256_loadu_pd(data);
    __m256i min_idx_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i current_idx_i64x4 = _mm256_setr_epi64x(4, 5, 6, 7);
    __m256i step_i64x4 = _mm256_set1_epi64x(4);

    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        __m256d lt_mask = _mm256_cmp_pd(data_f64x4, min_f64x4, _CMP_LT_OQ);
        min_f64x4 = _mm256_blendv_pd(min_f64x4, data_f64x4, lt_mask);
        min_idx_i64x4 = _mm256_blendv_epi8(min_idx_i64x4, current_idx_i64x4, _mm256_castpd_si256(lt_mask));
        current_idx_i64x4 = _mm256_add_epi64(current_idx_i64x4, step_i64x4);
    }

    // Handle scalar tail
    nk_f64_t min_val = nk_reduce_min_f64x4_haswell_(min_f64x4);
    nk_size_t min_idx = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] < min_val) {
            min_val = data[idx_scalars];
            min_idx = idx_scalars;
        }
    }

    // Find winning lane: compare each lane to horizontal min
    __m256d eq_mask = _mm256_cmp_pd(min_f64x4, _mm256_set1_pd(min_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_pd(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i64_t indices[4];
        _mm256_storeu_si256((__m256i *)indices, min_idx_i64x4);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *min_index = (min_idx && min_idx < vec_idx) ? min_idx : vec_idx;
    }
    else { *min_index = min_idx; }
    *min_value = min_val;
}

NK_INTERNAL void nk_reduce_min_f64_haswell_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Blend-based strided access: load full register, blend with +inf, find min
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d pos_inf_f64x4 = _mm256_set1_pd(NK_F64_MAX);
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
    if (count == 0) *min_value = NK_F64_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4)
        nk_reduce_min_f64_haswell_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_f64_haswell_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f64_haswell_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m256d max_f64x4 = _mm256_loadu_pd(data);
    __m256i max_idx_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i current_idx_i64x4 = _mm256_setr_epi64x(4, 5, 6, 7);
    __m256i step_i64x4 = _mm256_set1_epi64x(4);

    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        __m256d gt_mask = _mm256_cmp_pd(data_f64x4, max_f64x4, _CMP_GT_OQ);
        max_f64x4 = _mm256_blendv_pd(max_f64x4, data_f64x4, gt_mask);
        max_idx_i64x4 = _mm256_blendv_epi8(max_idx_i64x4, current_idx_i64x4, _mm256_castpd_si256(gt_mask));
        current_idx_i64x4 = _mm256_add_epi64(current_idx_i64x4, step_i64x4);
    }

    // Handle scalar tail
    nk_f64_t max_val = nk_reduce_max_f64x4_haswell_(max_f64x4);
    nk_size_t max_idx = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] > max_val) {
            max_val = data[idx_scalars];
            max_idx = idx_scalars;
        }
    }

    // Find winning lane: compare each lane to horizontal max
    __m256d eq_mask = _mm256_cmp_pd(max_f64x4, _mm256_set1_pd(max_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_pd(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i64_t indices[4];
        _mm256_storeu_si256((__m256i *)indices, max_idx_i64x4);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *max_index = (max_idx && max_idx < vec_idx) ? max_idx : vec_idx;
    }
    else { *max_index = max_idx; }
    *max_value = max_val;
}

NK_INTERNAL void nk_reduce_max_f64_haswell_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Blend-based strided access: load full register, blend with -inf, find max
    __m256i blend_mask_i64x4 = nk_stride_blend_b64x4_(stride_elements);
    __m256d neg_inf_f64x4 = _mm256_set1_pd(NK_F64_MIN);
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
    if (count == 0) *max_value = NK_F64_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4)
        nk_reduce_max_f64_haswell_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_f64_haswell_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i8_haswell_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    // Use 2×128-bit loads instead of 256-bit + VEXTRACTI128 to reduce Port 5 pressure.
    // Dual accumulators break dependency chain for better ILP.
    __m256i sum_low_i64x4 = _mm256_setzero_si256();
    __m256i sum_high_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        // Two 128-bit loads instead of 256-bit + extract
        __m128i low_i8x16 = _mm_loadu_si128((__m128i const *)(data + idx));
        __m128i high_i8x16 = _mm_loadu_si128((__m128i const *)(data + idx + 16));
        // Widen lower 16 bytes: i8 → i16 → i32 → i64
        __m256i low_i16x16 = _mm256_cvtepi8_epi16(low_i8x16);
        __m128i low_low_i16x8 = _mm256_castsi256_si128(low_i16x16);
        __m128i low_high_i16x8 = _mm256_extracti128_si256(low_i16x16, 1);
        __m256i low_low_i32x8 = _mm256_cvtepi16_epi32(low_low_i16x8);
        __m256i low_high_i32x8 = _mm256_cvtepi16_epi32(low_high_i16x8);
        __m128i a_i32x4 = _mm256_castsi256_si128(low_low_i32x8);
        __m128i b_i32x4 = _mm256_extracti128_si256(low_low_i32x8, 1);
        __m128i c_i32x4 = _mm256_castsi256_si128(low_high_i32x8);
        __m128i d_i32x4 = _mm256_extracti128_si256(low_high_i32x8, 1);
        sum_low_i64x4 = _mm256_add_epi64(sum_low_i64x4, _mm256_cvtepi32_epi64(a_i32x4));
        sum_low_i64x4 = _mm256_add_epi64(sum_low_i64x4, _mm256_cvtepi32_epi64(b_i32x4));
        sum_low_i64x4 = _mm256_add_epi64(sum_low_i64x4, _mm256_cvtepi32_epi64(c_i32x4));
        sum_low_i64x4 = _mm256_add_epi64(sum_low_i64x4, _mm256_cvtepi32_epi64(d_i32x4));
        // Widen upper 16 bytes
        __m256i high_i16x16 = _mm256_cvtepi8_epi16(high_i8x16);
        __m128i high_low_i16x8 = _mm256_castsi256_si128(high_i16x16);
        __m128i high_high_i16x8 = _mm256_extracti128_si256(high_i16x16, 1);
        __m256i high_low_i32x8 = _mm256_cvtepi16_epi32(high_low_i16x8);
        __m256i high_high_i32x8 = _mm256_cvtepi16_epi32(high_high_i16x8);
        __m128i e_i32x4 = _mm256_castsi256_si128(high_low_i32x8);
        __m128i f_i32x4 = _mm256_extracti128_si256(high_low_i32x8, 1);
        __m128i g_i32x4 = _mm256_castsi256_si128(high_high_i32x8);
        __m128i h_i32x4 = _mm256_extracti128_si256(high_high_i32x8, 1);
        sum_high_i64x4 = _mm256_add_epi64(sum_high_i64x4, _mm256_cvtepi32_epi64(e_i32x4));
        sum_high_i64x4 = _mm256_add_epi64(sum_high_i64x4, _mm256_cvtepi32_epi64(f_i32x4));
        sum_high_i64x4 = _mm256_add_epi64(sum_high_i64x4, _mm256_cvtepi32_epi64(g_i32x4));
        sum_high_i64x4 = _mm256_add_epi64(sum_high_i64x4, _mm256_cvtepi32_epi64(h_i32x4));
    }
    nk_i64_t sum = nk_reduce_add_i64x4_haswell_(_mm256_add_epi64(sum_low_i64x4, sum_high_i64x4));
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
    if (count == 0) *min_value = NK_I8_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_min_i8_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I8_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_max_i8_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U8_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_min_u8_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U8_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_max_u8_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I16_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_min_i16_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I16_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_max_i16_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U16_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_min_u16_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U16_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_max_u16_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I32_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_min_i32_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I32_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_max_i32_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U32_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_min_u32_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U32_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_max_u32_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I64_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_min_i64_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I64_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_max_i64_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U64_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_min_u64_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U64_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_max_u64_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_f16_haswell_contiguous_( //
    nk_f16_t const *data, nk_size_t count, nk_f32_t *result) {
    __m256 data_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_reduce_add_f16_haswell_contiguous_cycle:
    if (count < 8) {
        data_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(data, count);
        count = 0;
    }
    else {
        data_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)data));
        data += 8, count -= 8;
    }
    sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    if (count) goto nk_reduce_add_f16_haswell_contiguous_cycle;
    *result = nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_INTERNAL void nk_reduce_add_f16_haswell_strided_(                  //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_f16_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.f16s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = _mm256_cvtph_ps(vec.xmms[0]);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    }
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f16_haswell(                          //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (!aligned) nk_reduce_add_f16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f16_haswell_contiguous_(data, count, result);
    else nk_reduce_add_f16_haswell_strided_(data, count, stride_elements, result);
}

NK_INTERNAL void nk_reduce_min_f16_haswell_contiguous_( //
    nk_f16_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    // Single-pass: track both min value and index in SIMD
    __m256 min_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)data));
    __m256i min_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i current_idx_i32x8 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    __m256i step_i32x8 = _mm256_set1_epi32(8);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(data + idx)));
        __m256 lt_mask = _mm256_cmp_ps(data_f32x8, min_f32x8, _CMP_LT_OQ);
        min_f32x8 = _mm256_blendv_ps(min_f32x8, data_f32x8, lt_mask);
        min_idx_i32x8 = _mm256_blendv_epi8(min_idx_i32x8, current_idx_i32x8, _mm256_castps_si256(lt_mask));
        current_idx_i32x8 = _mm256_add_epi32(current_idx_i32x8, step_i32x8);
    }

    // Handle scalar tail
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    nk_size_t min_idx = 0;
    for (; idx < count; idx++) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(&data[idx], &val);
        if (val < min_val) {
            min_val = val;
            min_idx = idx;
        }
    }

    // Find winning lane
    __m256 eq_mask = _mm256_cmp_ps(min_f32x8, _mm256_set1_ps(min_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_ps(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i32_t indices[8];
        _mm256_storeu_si256((__m256i *)indices, min_idx_i32x8);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *min_index = (min_idx && min_idx < vec_idx) ? min_idx : vec_idx;
    }
    else { *min_index = min_idx; }
    *min_value = min_val;
}

NK_INTERNAL void nk_reduce_min_f16_haswell_strided_(                  //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    __m256 min_f32x8 = _mm256_set1_ps(NK_F32_MAX);
    nk_f16_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.f16s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = _mm256_cvtph_ps(vec.xmms[0]);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(ptr, &val);
        if (val < min_val) min_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(ptr, &val);
        if (val == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_f16_haswell(                          //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_f16_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_f16_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_f16_haswell_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f16_haswell_contiguous_( //
    nk_f16_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    // Single-pass: track both max value and index in SIMD
    __m256 max_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)data));
    __m256i max_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i current_idx_i32x8 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    __m256i step_i32x8 = _mm256_set1_epi32(8);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m256 data_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(data + idx)));
        __m256 gt_mask = _mm256_cmp_ps(data_f32x8, max_f32x8, _CMP_GT_OQ);
        max_f32x8 = _mm256_blendv_ps(max_f32x8, data_f32x8, gt_mask);
        max_idx_i32x8 = _mm256_blendv_epi8(max_idx_i32x8, current_idx_i32x8, _mm256_castps_si256(gt_mask));
        current_idx_i32x8 = _mm256_add_epi32(current_idx_i32x8, step_i32x8);
    }

    // Handle scalar tail
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    nk_size_t max_idx = 0;
    for (; idx < count; idx++) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(&data[idx], &val);
        if (val > max_val) {
            max_val = val;
            max_idx = idx;
        }
    }

    // Find winning lane
    __m256 eq_mask = _mm256_cmp_ps(max_f32x8, _mm256_set1_ps(max_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_ps(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i32_t indices[8];
        _mm256_storeu_si256((__m256i *)indices, max_idx_i32x8);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *max_index = (max_idx && max_idx < vec_idx) ? max_idx : vec_idx;
    }
    else { *max_index = max_idx; }
    *max_value = max_val;
}

NK_INTERNAL void nk_reduce_max_f16_haswell_strided_(                  //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
    nk_f16_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.f16s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = _mm256_cvtph_ps(vec.xmms[0]);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(ptr, &val);
        if (val > max_val) max_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_f16_to_f32_haswell(ptr, &val);
        if (val == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_f16_haswell(                          //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_f16_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_f16_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_f16_haswell_strided_(data, count, stride_elements, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_bf16_haswell_contiguous_( //
    nk_bf16_t const *data, nk_size_t count, nk_f32_t *result) {
    __m256 data_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_reduce_add_bf16_haswell_contiguous_cycle:
    if (count < 8) {
        data_f32x8 = nk_partial_load_bf16x8_to_f32x8_haswell_(data, count);
        count = 0;
    }
    else {
        data_f32x8 = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)data));
        data += 8, count -= 8;
    }
    sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    if (count) goto nk_reduce_add_bf16_haswell_contiguous_cycle;
    *result = nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_INTERNAL void nk_reduce_add_bf16_haswell_strided_(                  //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_bf16_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.bf16s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_bf16x8_to_f32x8_haswell_(vec.xmms[0]);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    }
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_bf16_haswell(                          //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (!aligned) nk_reduce_add_bf16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_bf16_haswell_contiguous_(data, count, result);
    else nk_reduce_add_bf16_haswell_strided_(data, count, stride_elements, result);
}

NK_INTERNAL void nk_reduce_min_bf16_haswell_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    // Single-pass: track both min value and index in SIMD
    __m128i data_i16x8 = _mm_loadu_si128((__m128i const *)data);
    __m256 min_f32x8 = nk_bf16x8_to_f32x8_haswell_(data_i16x8);
    __m256i min_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i current_idx_i32x8 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    __m256i step_i32x8 = _mm256_set1_epi32(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        data_i16x8 = _mm_loadu_si128((__m128i const *)(data + idx_scalars));
        __m256 data_f32x8 = nk_bf16x8_to_f32x8_haswell_(data_i16x8);
        __m256 lt_mask = _mm256_cmp_ps(data_f32x8, min_f32x8, _CMP_LT_OQ);
        min_f32x8 = _mm256_blendv_ps(min_f32x8, data_f32x8, lt_mask);
        min_idx_i32x8 = _mm256_blendv_epi8(min_idx_i32x8, current_idx_i32x8, _mm256_castps_si256(lt_mask));
        current_idx_i32x8 = _mm256_add_epi32(current_idx_i32x8, step_i32x8);
    }

    // Handle scalar tail
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    nk_size_t min_idx = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(&data[idx_scalars], &val);
        if (val < min_val) {
            min_val = val;
            min_idx = idx_scalars;
        }
    }

    // Find winning lane: compare each lane to horizontal min
    __m256 eq_mask = _mm256_cmp_ps(min_f32x8, _mm256_set1_ps(min_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_ps(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i32_t indices[8];
        _mm256_storeu_si256((__m256i *)indices, min_idx_i32x8);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *min_index = (min_idx && min_idx < vec_idx) ? min_idx : vec_idx;
    }
    else { *min_index = min_idx; }
    *min_value = min_val;
}

NK_INTERNAL void nk_reduce_min_bf16_haswell_strided_(                  //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    __m256 min_f32x8 = _mm256_set1_ps(NK_F32_MAX);
    nk_bf16_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.bf16s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_bf16x8_to_f32x8_haswell_(vec.xmms[0]);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(ptr, &val);
        if (val < min_val) min_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(ptr, &val);
        if (val == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_bf16_haswell(                          //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_bf16_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_bf16_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_bf16_haswell_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_bf16_haswell_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    // Single-pass: track both max value and index in SIMD
    __m128i data_i16x8 = _mm_loadu_si128((__m128i const *)data);
    __m256 max_f32x8 = nk_bf16x8_to_f32x8_haswell_(data_i16x8);
    __m256i max_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i current_idx_i32x8 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
    __m256i step_i32x8 = _mm256_set1_epi32(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        data_i16x8 = _mm_loadu_si128((__m128i const *)(data + idx_scalars));
        __m256 data_f32x8 = nk_bf16x8_to_f32x8_haswell_(data_i16x8);
        __m256 gt_mask = _mm256_cmp_ps(data_f32x8, max_f32x8, _CMP_GT_OQ);
        max_f32x8 = _mm256_blendv_ps(max_f32x8, data_f32x8, gt_mask);
        max_idx_i32x8 = _mm256_blendv_epi8(max_idx_i32x8, current_idx_i32x8, _mm256_castps_si256(gt_mask));
        current_idx_i32x8 = _mm256_add_epi32(current_idx_i32x8, step_i32x8);
    }

    // Handle scalar tail
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    nk_size_t max_idx = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(&data[idx_scalars], &val);
        if (val > max_val) {
            max_val = val;
            max_idx = idx_scalars;
        }
    }

    // Find winning lane: compare each lane to horizontal max
    __m256 eq_mask = _mm256_cmp_ps(max_f32x8, _mm256_set1_ps(max_val), _CMP_EQ_OQ);
    int eq_bits = _mm256_movemask_ps(eq_mask);
    if (eq_bits) {
        unsigned int first_lane = (unsigned int)_tzcnt_u32((unsigned int)eq_bits);
        nk_i32_t indices[8];
        _mm256_storeu_si256((__m256i *)indices, max_idx_i32x8);
        nk_size_t vec_idx = (nk_size_t)indices[first_lane];
        *max_index = (max_idx && max_idx < vec_idx) ? max_idx : vec_idx;
    }
    else { *max_index = max_idx; }
    *max_value = max_val;
}

NK_INTERNAL void nk_reduce_max_bf16_haswell_strided_(                  //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
    nk_bf16_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.bf16s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_bf16x8_to_f32x8_haswell_(vec.xmms[0]);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(ptr, &val);
        if (val > max_val) max_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_bf16_to_f32_serial(ptr, &val);
        if (val == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_bf16_haswell(                          //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_bf16_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_bf16_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_bf16_haswell_strided_(data, count, stride_elements, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_e4m3_haswell_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count, nk_f32_t *result) {
    __m256 data_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_reduce_add_e4m3_haswell_contiguous_cycle:
    if (count < 8) {
        data_f32x8 = nk_partial_load_e4m3x8_to_f32x8_haswell_(data, count);
        count = 0;
    }
    else {
        data_f32x8 = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)data));
        data += 8, count -= 8;
    }
    sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    if (count) goto nk_reduce_add_e4m3_haswell_contiguous_cycle;
    *result = nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_INTERNAL void nk_reduce_add_e4m3_haswell_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.e4m3s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_e4m3x8_to_f32x8_haswell_(vec.xmms[0]);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    }
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial(ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_e4m3_haswell(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (!aligned) nk_reduce_add_e4m3_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_e4m3_haswell_contiguous_(data, count, result);
    else nk_reduce_add_e4m3_haswell_strided_(data, count, stride_elements, result);
}

NK_INTERNAL void nk_reduce_min_e4m3_haswell_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

    // E4M3 NaN (0x7F) maps to 0xFF in comparable form
    __m256i nan_cmp_u8x32 = _mm256_set1_epi8((char)0xFF);
    __m256i one_i64x4 = _mm256_set1_epi64x(1);
    __m256i bit_isolate_i64x4 = _mm256_setr_epi64x(1, 2, 4, 8);
    nk_b256_vec_t min_vec, argmin_vec;
    min_vec.ymm = nan_cmp_u8x32;             // Identity for min (0xFF)
    argmin_vec.ymm = _mm256_setzero_si256(); // Track which iteration each qword's min came from
    __m256i current_chunk_i64x4 = _mm256_setzero_si256();
    __m256i data_i8x32;

nk_reduce_min_e4m3_haswell_cycle_:
    if (count < 32) {
        nk_b256_vec_t data_vec;
        nk_partial_load_b8x32_serial_(data, count, &data_vec);
        // Blend tail with 0xFF (min identity) - create mask where byte index >= count
        __m256i indices = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
                                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        __m256i tail_mask = _mm256_cmpgt_epi8(indices, _mm256_set1_epi8((char)(count - 1)));
        data_i8x32 = _mm256_or_si256(data_vec.ymm, tail_mask); // OR with 0xFF where mask is set
        count = 0;
    }
    else {
        data_i8x32 = _mm256_loadu_si256((__m256i const *)data);
        data += 32, count -= 32;
    }
    __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(data_i8x32);
    __m256i min_mask_i8x32 = nk_min_mask_e4m3x32_haswell_(min_vec.ymm, data_cmp_u8x32, nan_cmp_u8x32);
    __m256i new_min_cmp_u8x32 = _mm256_blendv_epi8(data_cmp_u8x32, min_vec.ymm, min_mask_i8x32);
    __m256i changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min_cmp_u8x32, min_vec.ymm),
                                             _mm256_set1_epi8((char)0xFF));
    int changed_bits = _mm256_movemask_epi8(changed_i8x32);

    // Convert 32-bit byte-change mask to 4-bit qword-change mask using SWAR + PEXT:
    // 1. OR-cascade collapses each 8-bit group to its bit 0
    // 2. PEXT extracts bits 0,8,16,24 into consecutive bits
    // 3. Broadcast + isolate + compare expands back to blend mask
    nk_u32_t x = (nk_u32_t)changed_bits;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    nk_u32_t mask4 = _pext_u32(x, 0x01010101);
    __m256i mask_broadcast = _mm256_set1_epi64x((nk_i64_t)mask4);
    __m256i chunk_updated_i64x4 = _mm256_cmpeq_epi64(_mm256_and_si256(mask_broadcast, bit_isolate_i64x4),
                                                     bit_isolate_i64x4);

    argmin_vec.ymm = _mm256_blendv_epi8(argmin_vec.ymm, current_chunk_i64x4, chunk_updated_i64x4);
    min_vec.ymm = new_min_cmp_u8x32;
    current_chunk_i64x4 = _mm256_add_epi64(current_chunk_i64x4, one_i64x4);
    if (count) goto nk_reduce_min_e4m3_haswell_cycle_;

    // Horizontal reduction: find lane with minimum value, extract its iteration index
    nk_size_t first_lane = nk_argmin_u8x32_haswell_(min_vec.ymm);
    nk_size_t chunk_idx = (nk_size_t)argmin_vec.i64s[first_lane / 8];
    *min_index = chunk_idx * 32 + (first_lane % 32);

    // Convert min value back to FP8, then to F32
    min_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(min_vec.ymm);
    nk_e4m3_to_f32_serial(&min_vec.e4m3s[first_lane], min_value);
}

NK_INTERNAL void nk_reduce_min_e4m3_haswell_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    __m256 min_f32x8 = _mm256_set1_ps(NK_F32_MAX);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.e4m3s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_e4m3x8_to_f32x8_haswell_(vec.xmms[0]);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial(ptr, &val);
        if (val < min_val) min_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial(ptr, &val);
        if (val == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_e4m3_haswell(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_e4m3_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e4m3_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e4m3_haswell_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e4m3_haswell_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

    // E4M3 NaN (0x7F) maps to 0xFF in comparable form
    __m256i nan_cmp_u8x32 = _mm256_set1_epi8((char)0xFF);
    __m256i one_i64x4 = _mm256_set1_epi64x(1);
    __m256i bit_isolate_i64x4 = _mm256_setr_epi64x(1, 2, 4, 8);
    nk_b256_vec_t max_vec, argmax_vec;
    max_vec.ymm = _mm256_setzero_si256();    // Identity for max (0x00)
    argmax_vec.ymm = _mm256_setzero_si256(); // Track which iteration each qword's max came from
    __m256i current_chunk_i64x4 = _mm256_setzero_si256();
    __m256i data_i8x32;

nk_reduce_max_e4m3_haswell_cycle_:
    if (count < 32) {
        nk_b256_vec_t data_vec;
        nk_partial_load_b8x32_serial_(data, count, &data_vec);
        data_i8x32 = data_vec.ymm; // zeros in tail are already max identity (0x00)
        count = 0;
    }
    else {
        data_i8x32 = _mm256_loadu_si256((__m256i const *)data);
        data += 32, count -= 32;
    }
    __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(data_i8x32);
    __m256i max_mask_i8x32 = nk_max_mask_e4m3x32_haswell_(max_vec.ymm, data_cmp_u8x32, nan_cmp_u8x32);
    __m256i new_max_cmp_u8x32 = _mm256_blendv_epi8(data_cmp_u8x32, max_vec.ymm, max_mask_i8x32);
    __m256i changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max_cmp_u8x32, max_vec.ymm),
                                             _mm256_set1_epi8((char)0xFF));
    int changed_bits = _mm256_movemask_epi8(changed_i8x32);

    // Convert 32-bit byte-change mask to 4-bit qword-change mask using SWAR + PEXT
    nk_u32_t x = (nk_u32_t)changed_bits;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    nk_u32_t mask4 = _pext_u32(x, 0x01010101);
    __m256i mask_broadcast = _mm256_set1_epi64x((nk_i64_t)mask4);
    __m256i chunk_updated_i64x4 = _mm256_cmpeq_epi64(_mm256_and_si256(mask_broadcast, bit_isolate_i64x4),
                                                     bit_isolate_i64x4);

    argmax_vec.ymm = _mm256_blendv_epi8(argmax_vec.ymm, current_chunk_i64x4, chunk_updated_i64x4);
    max_vec.ymm = new_max_cmp_u8x32;
    current_chunk_i64x4 = _mm256_add_epi64(current_chunk_i64x4, one_i64x4);
    if (count) goto nk_reduce_max_e4m3_haswell_cycle_;

    // Horizontal reduction: find lane with maximum value, extract its iteration index
    nk_size_t first_lane = nk_argmax_u8x32_haswell_(max_vec.ymm);
    nk_size_t chunk_idx = (nk_size_t)argmax_vec.i64s[first_lane / 8];
    *max_index = chunk_idx * 32 + (first_lane % 32);

    // Convert max value back to FP8, then to F32
    max_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(max_vec.ymm);
    nk_e4m3_to_f32_serial(&max_vec.e4m3s[first_lane], max_value);
}

NK_INTERNAL void nk_reduce_max_e4m3_haswell_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.e4m3s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_e4m3x8_to_f32x8_haswell_(vec.xmms[0]);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial(ptr, &val);
        if (val > max_val) max_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32_serial(ptr, &val);
        if (val == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_e4m3_haswell(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_e4m3_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e4m3_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e4m3_haswell_strided_(data, count, stride_elements, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_e5m2_haswell_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count, nk_f32_t *result) {
    __m256 data_f32x8;
    __m256 sum_f32x8 = _mm256_setzero_ps();
nk_reduce_add_e5m2_haswell_contiguous_cycle:
    if (count < 8) {
        data_f32x8 = nk_partial_load_e5m2x8_to_f32x8_haswell_(data, count);
        count = 0;
    }
    else {
        data_f32x8 = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)data));
        data += 8, count -= 8;
    }
    sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    if (count) goto nk_reduce_add_e5m2_haswell_contiguous_cycle;
    *result = nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

NK_INTERNAL void nk_reduce_add_e5m2_haswell_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.e5m2s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_e5m2x8_to_f32x8_haswell_(vec.xmms[0]);
        sum_f32x8 = _mm256_add_ps(sum_f32x8, data_f32x8);
    }
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial(ptr, &val);
        sum += val;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_e5m2_haswell(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (!aligned) nk_reduce_add_e5m2_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_e5m2_haswell_contiguous_(data, count, result);
    else nk_reduce_add_e5m2_haswell_strided_(data, count, stride_elements, result);
}

NK_INTERNAL void nk_reduce_min_e5m2_haswell_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

    // E5M2 NaN (0x7D-0x7F) maps to 0xFD-0xFF in comparable form
    __m256i nan_threshold_cmp_u8x32 = _mm256_set1_epi8((char)0xFD);
    __m256i one_i64x4 = _mm256_set1_epi64x(1);
    __m256i bit_isolate_i64x4 = _mm256_setr_epi64x(1, 2, 4, 8);
    nk_b256_vec_t min_vec, argmin_vec;
    min_vec.ymm = _mm256_set1_epi8((char)0xFF); // Identity for min
    argmin_vec.ymm = _mm256_setzero_si256();
    __m256i current_chunk_i64x4 = _mm256_setzero_si256();
    __m256i data_i8x32;

nk_reduce_min_e5m2_haswell_cycle_:
    if (count < 32) {
        nk_b256_vec_t data_vec;
        nk_partial_load_b8x32_serial_(data, count, &data_vec);
        // Blend tail with 0xFF (min identity) - create mask where byte index >= count
        __m256i indices = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
                                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        __m256i tail_mask = _mm256_cmpgt_epi8(indices, _mm256_set1_epi8((char)(count - 1)));
        data_i8x32 = _mm256_or_si256(data_vec.ymm, tail_mask); // OR with 0xFF where mask is set
        count = 0;
    }
    else {
        data_i8x32 = _mm256_loadu_si256((__m256i const *)data);
        data += 32, count -= 32;
    }
    __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(data_i8x32);
    __m256i min_mask_i8x32 = nk_min_mask_e5m2x32_haswell_(min_vec.ymm, data_cmp_u8x32, nan_threshold_cmp_u8x32);
    __m256i new_min_cmp_u8x32 = _mm256_blendv_epi8(data_cmp_u8x32, min_vec.ymm, min_mask_i8x32);
    __m256i changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_min_cmp_u8x32, min_vec.ymm),
                                             _mm256_set1_epi8((char)0xFF));
    int changed_bits = _mm256_movemask_epi8(changed_i8x32);

    // Convert 32-bit byte-change mask to 4-bit qword-change mask using SWAR + PEXT
    nk_u32_t x = (nk_u32_t)changed_bits;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    nk_u32_t mask4 = _pext_u32(x, 0x01010101);
    __m256i mask_broadcast = _mm256_set1_epi64x((nk_i64_t)mask4);
    __m256i chunk_updated_i64x4 = _mm256_cmpeq_epi64(_mm256_and_si256(mask_broadcast, bit_isolate_i64x4),
                                                     bit_isolate_i64x4);

    argmin_vec.ymm = _mm256_blendv_epi8(argmin_vec.ymm, current_chunk_i64x4, chunk_updated_i64x4);
    min_vec.ymm = new_min_cmp_u8x32;
    current_chunk_i64x4 = _mm256_add_epi64(current_chunk_i64x4, one_i64x4);
    if (count) goto nk_reduce_min_e5m2_haswell_cycle_;

    // Horizontal reduction: find lane with minimum value, extract its iteration index
    nk_size_t first_lane = nk_argmin_u8x32_haswell_(min_vec.ymm);
    nk_size_t chunk_idx = (nk_size_t)argmin_vec.i64s[first_lane / 8];
    *min_index = chunk_idx * 32 + (first_lane % 32);

    // Convert min value back to FP8, then to F32
    min_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(min_vec.ymm);
    nk_e5m2_to_f32_serial(&min_vec.e5m2s[first_lane], min_value);
}

NK_INTERNAL void nk_reduce_min_e5m2_haswell_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    __m256 min_f32x8 = _mm256_set1_ps(NK_F32_MAX);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.e5m2s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_e5m2x8_to_f32x8_haswell_(vec.xmms[0]);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    nk_f32_t min_val = nk_reduce_min_f32x8_haswell_(min_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial(ptr, &val);
        if (val < min_val) min_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial(ptr, &val);
        if (val == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_e5m2_haswell(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_e5m2_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e5m2_haswell_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e5m2_haswell_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e5m2_haswell_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

    // E5M2 NaN (0x7D-0x7F) maps to 0xFD-0xFF in comparable form
    __m256i nan_threshold_cmp_u8x32 = _mm256_set1_epi8((char)0xFD);
    __m256i one_i64x4 = _mm256_set1_epi64x(1);
    __m256i bit_isolate_i64x4 = _mm256_setr_epi64x(1, 2, 4, 8);
    nk_b256_vec_t max_vec, argmax_vec;
    max_vec.ymm = _mm256_setzero_si256(); // Identity for max (0x00)
    argmax_vec.ymm = _mm256_setzero_si256();
    __m256i current_chunk_i64x4 = _mm256_setzero_si256();
    __m256i data_i8x32;

nk_reduce_max_e5m2_haswell_cycle_:
    if (count < 32) {
        nk_b256_vec_t data_vec;
        nk_partial_load_b8x32_serial_(data, count, &data_vec);
        data_i8x32 = data_vec.ymm; // zeros in tail are already max identity (0x00)
        count = 0;
    }
    else {
        data_i8x32 = _mm256_loadu_si256((__m256i const *)data);
        data += 32, count -= 32;
    }
    __m256i data_cmp_u8x32 = nk_fp8x32_to_u8x32_comparable_haswell_(data_i8x32);
    __m256i max_mask_i8x32 = nk_max_mask_e5m2x32_haswell_(max_vec.ymm, data_cmp_u8x32, nan_threshold_cmp_u8x32);
    __m256i new_max_cmp_u8x32 = _mm256_blendv_epi8(data_cmp_u8x32, max_vec.ymm, max_mask_i8x32);
    __m256i changed_i8x32 = _mm256_xor_si256(_mm256_cmpeq_epi8(new_max_cmp_u8x32, max_vec.ymm),
                                             _mm256_set1_epi8((char)0xFF));
    int changed_bits = _mm256_movemask_epi8(changed_i8x32);

    // Convert 32-bit byte-change mask to 4-bit qword-change mask using SWAR + PEXT
    nk_u32_t x = (nk_u32_t)changed_bits;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    nk_u32_t mask4 = _pext_u32(x, 0x01010101);
    __m256i mask_broadcast = _mm256_set1_epi64x((nk_i64_t)mask4);
    __m256i chunk_updated_i64x4 = _mm256_cmpeq_epi64(_mm256_and_si256(mask_broadcast, bit_isolate_i64x4),
                                                     bit_isolate_i64x4);

    argmax_vec.ymm = _mm256_blendv_epi8(argmax_vec.ymm, current_chunk_i64x4, chunk_updated_i64x4);
    max_vec.ymm = new_max_cmp_u8x32;
    current_chunk_i64x4 = _mm256_add_epi64(current_chunk_i64x4, one_i64x4);
    if (count) goto nk_reduce_max_e5m2_haswell_cycle_;

    // Horizontal reduction: find lane with maximum value, extract its iteration index
    nk_size_t first_lane = nk_argmax_u8x32_haswell_(max_vec.ymm);
    nk_size_t chunk_idx = (nk_size_t)argmax_vec.i64s[first_lane / 8];
    *max_index = chunk_idx * 32 + (first_lane % 32);

    // Convert max value back to FP8, then to F32
    max_vec.ymm = nk_u8x32_comparable_to_fp8x32_haswell_(max_vec.ymm);
    nk_e5m2_to_f32_serial(&max_vec.e5m2s[first_lane], max_value);
}

NK_INTERNAL void nk_reduce_max_e5m2_haswell_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b256_vec_t vec;
        vec.ymm = _mm256_setzero_si256();
        for (nk_size_t i = 0; i < 8; ++i) {
            vec.e5m2s[i] = *ptr;
            ptr += stride_elements;
        }
        __m256 data_f32x8 = nk_e5m2x8_to_f32x8_haswell_(vec.xmms[0]);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    nk_f32_t max_val = nk_reduce_max_f32x8_haswell_(max_f32x8);
    for (; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial(ptr, &val);
        if (val > max_val) max_val = val;
    }
    // Second pass for index
    ptr = data;
    for (idx = 0; idx < count; idx++, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32_serial(ptr, &val);
        if (val == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_e5m2_haswell(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_e5m2_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e5m2_haswell_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e5m2_haswell_strided_(data, count, stride_elements, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_REDUCE_HASWELL_H
