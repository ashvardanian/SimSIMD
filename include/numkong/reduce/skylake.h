/**
 *  @brief SIMD-accelerated horizontal reduction operations for Intel Skylake-X CPUs.
 *  @file include/numkong/reduce/skylake.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_SKYLAKE_H
#define NK_REDUCE_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Horizontal sum of 16 floats in a ZMM register (native f32 precision). */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x16_skylake_(__m512 sum_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(sum_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(sum_f32x16, 1);
    __m256 sum_f32x8 = _mm256_add_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(sum_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(sum_f32x8, 1);
    __m128 sum_f32x4 = _mm_add_ps(lo_f32x4, hi_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    return _mm_cvtss_f32(sum_f32x4);
}

/** @brief Horizontal sum of 8 doubles in a ZMM register. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x8_skylake_(__m512d sum_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(sum_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(sum_f64x8, 1);
    __m256d sum_f64x4 = _mm256_add_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(sum_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(sum_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(lo_f64x2, hi_f64x2);
    sum_f64x2 = _mm_hadd_pd(sum_f64x2, sum_f64x2);
    return _mm_cvtsd_f64(sum_f64x2);
}

/** @brief Horizontal min of 16 floats in a ZMM register. */
NK_INTERNAL nk_f32_t nk_reduce_min_f32x16_skylake_(__m512 min_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(min_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(min_f32x16, 1);
    __m256 min_f32x8 = _mm256_min_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(min_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(min_f32x8, 1);
    __m128 min_f32x4 = _mm_min_ps(lo_f32x4, hi_f32x4);
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(min_f32x4);
}

/** @brief Horizontal max of 16 floats in a ZMM register. */
NK_INTERNAL nk_f32_t nk_reduce_max_f32x16_skylake_(__m512 max_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(max_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(max_f32x16, 1);
    __m256 max_f32x8 = _mm256_max_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(max_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(max_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(lo_f32x4, hi_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max_f32x4);
}

/** @brief Horizontal min of 8 doubles in a ZMM register. */
NK_INTERNAL nk_f64_t nk_reduce_min_f64x8_skylake_(__m512d min_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(min_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(min_f64x8, 1);
    __m256d min_f64x4 = _mm256_min_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(min_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(min_f64x4, 1);
    __m128d min_f64x2 = _mm_min_pd(lo_f64x2, hi_f64x2);
    min_f64x2 = _mm_min_pd(min_f64x2, _mm_shuffle_pd(min_f64x2, min_f64x2, 1));
    return _mm_cvtsd_f64(min_f64x2);
}

/** @brief Horizontal max of 8 doubles in a ZMM register. */
NK_INTERNAL nk_f64_t nk_reduce_max_f64x8_skylake_(__m512d max_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(max_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(max_f64x8, 1);
    __m256d max_f64x4 = _mm256_max_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(max_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(max_f64x4, 1);
    __m128d max_f64x2 = _mm_max_pd(lo_f64x2, hi_f64x2);
    max_f64x2 = _mm_max_pd(max_f64x2, _mm_shuffle_pd(max_f64x2, max_f64x2, 1));
    return _mm_cvtsd_f64(max_f64x2);
}

/** @brief Horizontal sum of 16 i32s in a ZMM register. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x16_skylake_(__m512i sum_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(sum_i32x16);
    __m256i hi_i32x8 = _mm512_extracti32x8_epi32(sum_i32x16, 1);
    __m256i sum_i32x8 = _mm256_add_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(sum_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(sum_i32x8, 1);
    __m128i sum_i32x4 = _mm_add_epi32(lo_i32x4, hi_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    return _mm_cvtsi128_si32(sum_i32x4);
}

/** @brief Horizontal sum of 8 i64s in a ZMM register. */
NK_INTERNAL nk_i64_t nk_reduce_add_i64x8_skylake_(__m512i sum_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(sum_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(sum_i64x8, 1);
    __m256i sum_i64x4 = _mm256_add_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    sum_i64x2 = _mm_add_epi64(sum_i64x2, _mm_shuffle_epi32(sum_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si64(sum_i64x2);
}

/**
 *  @brief Returns AVX-512 mask for strided access of 8-bit elements (64-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  With 64 elements per register, useful for strides 2-16 (yielding 4+ elements per load).
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask64 nk_stride_mask_u1x64_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask64)0x5555555555555555ull;  // 32 elems
    case 3: return (__mmask64)0x1249249249249249ull;  // 21 elems
    case 4: return (__mmask64)0x1111111111111111ull;  // 16 elems
    case 5: return (__mmask64)0x1084210842108421ull;  // 12 elems
    case 6: return (__mmask64)0x1041041041041041ull;  // 10 elems
    case 7: return (__mmask64)0x0408102040810204ull;  // 9 elems
    case 8: return (__mmask64)0x0101010101010101ull;  // 8 elems
    case 9: return (__mmask64)0x0080200802008020ull;  // 7 elems
    case 10: return (__mmask64)0x0040100401004010ull; // 6 elems
    case 11: return (__mmask64)0x0020080200802008ull; // 5 elems
    case 12: return (__mmask64)0x0010040100401004ull; // 5 elems
    case 13: return (__mmask64)0x0008020080200802ull; // 4 elems
    case 14: return (__mmask64)0x0004010040100401ull; // 4 elems
    case 15: return (__mmask64)0x0002008020080200ull; // 4 elems
    case 16: return (__mmask64)0x0001000100010001ull; // 4 elems
    default: return (__mmask64)0;
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 32-bit elements (16-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask16 nk_stride_mask_b32x16_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask16)0x5555; // [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] → 8 elems
    case 3: return (__mmask16)0x1249; // [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0] → 5 elems
    case 4: return (__mmask16)0x1111; // [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0] → 4 elems
    case 5: return (__mmask16)0x0421; // [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0] → 3 elems
    case 6: return (__mmask16)0x0041; // [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] → 2 elems
    case 7: return (__mmask16)0x0081; // [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] → 2 elems
    case 8: return (__mmask16)0x0101; // [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] → 2 elems
    default: return (__mmask16)0;     // Invalid stride - caller should use gather or serial
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 16-bit elements (32-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column int16 matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask32 nk_stride_mask_b16x32_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask32)0x55555555;  // 16 elems
    case 3: return (__mmask32)0x09249249;  // 11 elems
    case 4: return (__mmask32)0x11111111;  // 8 elems
    case 5: return (__mmask32)0x01084210;  // 6 elems
    case 6: return (__mmask32)0x01041041;  // 5 elems
    case 7: return (__mmask32)0x00408102;  // 4 elems
    case 8: return (__mmask32)0x01010101;  // 4 elems
    case 9: return (__mmask32)0x00802008;  // 3 elems
    case 10: return (__mmask32)0x00401004; // 3 elems
    case 11: return (__mmask32)0x00200802; // 3 elems
    case 12: return (__mmask32)0x00100401; // 2 elems
    case 13: return (__mmask32)0x00080200; // 2 elems
    case 14: return (__mmask32)0x00040100; // 2 elems
    case 15: return (__mmask32)0x00020080; // 2 elems
    case 16: return (__mmask32)0x00010001; // 2 elems
    default: return (__mmask32)0;
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 64-bit elements (8-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask8 nk_stride_mask_b64x8_(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask8)0x55; // [1,0,1,0,1,0,1,0] → 4 elems
    case 3: return (__mmask8)0x49; // [1,0,0,1,0,0,1,0] → 3 elems
    case 4: return (__mmask8)0x11; // [1,0,0,0,1,0,0,0] → 2 elems
    case 5: return (__mmask8)0x21; // [1,0,0,0,0,1,0,0] → 2 elems
    case 6: return (__mmask8)0x41; // [1,0,0,0,0,0,1,0] → 2 elems
    case 7: return (__mmask8)0x01; // [1,0,0,0,0,0,0,0] → 1 elem
    case 8: return (__mmask8)0x01; // [1,0,0,0,0,0,0,0] → 1 elem
    default: return (__mmask8)0;
    }
}

/**
 *  @brief Returns initial logical index vector for 32-bit strided access (16-element register).
 *
 *  For min/max with index tracking: non-stride positions get 0 (don't matter, masked out).
 *  Stride positions get sequential logical indices: 0, 1, 2, ...
 */
NK_INTERNAL __m512i nk_stride_logidx_i32x16_(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm512_setr_epi32(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0); // 8 elems
    case 3: return _mm512_setr_epi32(0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 0); // 5 elems
    case 4: return _mm512_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0); // 4 elems
    case 5: return _mm512_setr_epi32(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0); // 3 elems
    case 6: return _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0); // 2 elems
    case 7: return _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0); // 2 elems
    case 8: return _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0); // 2 elems
    default: return _mm512_setzero_si512();
    }
}

/**
 *  @brief Returns initial logical index vector for 64-bit strided access (8-element register).
 *
 *  For min/max with index tracking: non-stride positions get 0 (don't matter, masked out).
 *  Stride positions get sequential logical indices: 0, 1, 2, ...
 */
NK_INTERNAL __m512i nk_stride_logidx_i64x8_(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm512_setr_epi64(0, 0, 1, 0, 2, 0, 3, 0); // 4 elems
    case 3: return _mm512_setr_epi64(0, 0, 0, 1, 0, 0, 2, 0); // 3 elems
    case 4: return _mm512_setr_epi64(0, 0, 0, 0, 1, 0, 0, 0); // 2 elems
    case 5: return _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 0, 0); // 2 elems
    case 6: return _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 1, 0); // 2 elems
    case 7: return _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 0, 0); // 1 elem
    case 8: return _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 0, 0); // 1 elem
    default: return _mm512_setzero_si512();
    }
}

/**
 *  @brief Returns number of logical elements per 16-scalar chunk for given stride.
 */
NK_INTERNAL nk_size_t nk_stride_elems_b32x16_(nk_size_t stride) {
    switch (stride) {
    case 2: return 8;
    case 3: return 5;
    case 4: return 4;
    case 5: return 3;
    case 6: return 2;
    case 7: return 2;
    case 8: return 2;
    default: return 0;
    }
}

/**
 *  @brief Returns number of logical elements per 8-scalar chunk for given stride.
 */
NK_INTERNAL nk_size_t nk_stride_elems_b64x8_(nk_size_t stride) {
    switch (stride) {
    case 2: return 4;
    case 3: return 3;
    case 4: return 2;
    case 5: return 2;
    case 6: return 2;
    case 7: return 1;
    case 8: return 1;
    default: return 0;
    }
}

/** @brief Horizontal min of 64 signed i8s in a ZMM register. */
NK_INTERNAL nk_i8_t nk_reduce_min_i8x64_skylake_(__m512i min_i8x64) {
    __m256i lo_i8x32 = _mm512_castsi512_si256(min_i8x64);
    __m256i hi_i8x32 = _mm512_extracti64x4_epi64(min_i8x64, 1);
    __m256i min_i8x32 = _mm256_min_epi8(lo_i8x32, hi_i8x32);
    __m128i lo_i8x16 = _mm256_castsi256_si128(min_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(min_i8x32, 1);
    __m128i min_i8x16 = _mm_min_epi8(lo_i8x16, hi_i8x16);
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shufflelo_epi16(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_epi16(min_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(min_i8x16);
}

/** @brief Horizontal max of 64 signed i8s in a ZMM register. */
NK_INTERNAL nk_i8_t nk_reduce_max_i8x64_skylake_(__m512i max_i8x64) {
    __m256i lo_i8x32 = _mm512_castsi512_si256(max_i8x64);
    __m256i hi_i8x32 = _mm512_extracti64x4_epi64(max_i8x64, 1);
    __m256i max_i8x32 = _mm256_max_epi8(lo_i8x32, hi_i8x32);
    __m128i lo_i8x16 = _mm256_castsi256_si128(max_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(max_i8x32, 1);
    __m128i max_i8x16 = _mm_max_epi8(lo_i8x16, hi_i8x16);
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shufflelo_epi16(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_epi16(max_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(max_i8x16);
}

/** @brief Horizontal min of 64 unsigned u8s in a ZMM register. */
NK_INTERNAL nk_u8_t nk_reduce_min_u8x64_skylake_(__m512i min_u8x64) {
    __m256i lo_u8x32 = _mm512_castsi512_si256(min_u8x64);
    __m256i hi_u8x32 = _mm512_extracti64x4_epi64(min_u8x64, 1);
    __m256i min_u8x32 = _mm256_min_epu8(lo_u8x32, hi_u8x32);
    __m128i lo_u8x16 = _mm256_castsi256_si128(min_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(min_u8x32, 1);
    __m128i min_u8x16 = _mm_min_epu8(lo_u8x16, hi_u8x16);
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shufflelo_epi16(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_epi16(min_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(min_u8x16);
}

/** @brief Horizontal max of 64 unsigned u8s in a ZMM register. */
NK_INTERNAL nk_u8_t nk_reduce_max_u8x64_skylake_(__m512i max_u8x64) {
    __m256i lo_u8x32 = _mm512_castsi512_si256(max_u8x64);
    __m256i hi_u8x32 = _mm512_extracti64x4_epi64(max_u8x64, 1);
    __m256i max_u8x32 = _mm256_max_epu8(lo_u8x32, hi_u8x32);
    __m128i lo_u8x16 = _mm256_castsi256_si128(max_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(max_u8x32, 1);
    __m128i max_u8x16 = _mm_max_epu8(lo_u8x16, hi_u8x16);
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shufflelo_epi16(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_epi16(max_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(max_u8x16);
}

/** @brief Horizontal min of 32 signed i16s in a ZMM register. */
NK_INTERNAL nk_i16_t nk_reduce_min_i16x32_skylake_(__m512i min_i16x32) {
    __m256i lo_i16x16 = _mm512_castsi512_si256(min_i16x32);
    __m256i hi_i16x16 = _mm512_extracti64x4_epi64(min_i16x32, 1);
    __m256i min_i16x16 = _mm256_min_epi16(lo_i16x16, hi_i16x16);
    __m128i lo_i16x8 = _mm256_castsi256_si128(min_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(min_i16x16, 1);
    __m128i min_i16x8 = _mm_min_epi16(lo_i16x8, hi_i16x8);
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shufflelo_epi16(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(min_i16x8);
}

/** @brief Horizontal max of 32 signed i16s in a ZMM register. */
NK_INTERNAL nk_i16_t nk_reduce_max_i16x32_skylake_(__m512i max_i16x32) {
    __m256i lo_i16x16 = _mm512_castsi512_si256(max_i16x32);
    __m256i hi_i16x16 = _mm512_extracti64x4_epi64(max_i16x32, 1);
    __m256i max_i16x16 = _mm256_max_epi16(lo_i16x16, hi_i16x16);
    __m128i lo_i16x8 = _mm256_castsi256_si128(max_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(max_i16x16, 1);
    __m128i max_i16x8 = _mm_max_epi16(lo_i16x8, hi_i16x8);
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shufflelo_epi16(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(max_i16x8);
}

/** @brief Horizontal min of 32 unsigned u16s in a ZMM register. */
NK_INTERNAL nk_u16_t nk_reduce_min_u16x32_skylake_(__m512i min_u16x32) {
    __m256i lo_u16x16 = _mm512_castsi512_si256(min_u16x32);
    __m256i hi_u16x16 = _mm512_extracti64x4_epi64(min_u16x32, 1);
    __m256i min_u16x16 = _mm256_min_epu16(lo_u16x16, hi_u16x16);
    __m128i lo_u16x8 = _mm256_castsi256_si128(min_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(min_u16x16, 1);
    __m128i min_u16x8 = _mm_min_epu16(lo_u16x8, hi_u16x8);
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shufflelo_epi16(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(min_u16x8);
}

/** @brief Horizontal max of 32 unsigned u16s in a ZMM register. */
NK_INTERNAL nk_u16_t nk_reduce_max_u16x32_skylake_(__m512i max_u16x32) {
    __m256i lo_u16x16 = _mm512_castsi512_si256(max_u16x32);
    __m256i hi_u16x16 = _mm512_extracti64x4_epi64(max_u16x32, 1);
    __m256i max_u16x16 = _mm256_max_epu16(lo_u16x16, hi_u16x16);
    __m128i lo_u16x8 = _mm256_castsi256_si128(max_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(max_u16x16, 1);
    __m128i max_u16x8 = _mm_max_epu16(lo_u16x8, hi_u16x8);
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shufflelo_epi16(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(max_u16x8);
}

/** @brief Horizontal min of 16 signed i32s in a ZMM register. */
NK_INTERNAL nk_i32_t nk_reduce_min_i32x16_skylake_(__m512i min_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(min_i32x16);
    __m256i hi_i32x8 = _mm512_extracti64x4_epi64(min_i32x16, 1);
    __m256i min_i32x8 = _mm256_min_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(min_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(min_i32x8, 1);
    __m128i min_i32x4 = _mm_min_epi32(lo_i32x4, hi_i32x4);
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(min_i32x4);
}

/** @brief Horizontal max of 16 signed i32s in a ZMM register. */
NK_INTERNAL nk_i32_t nk_reduce_max_i32x16_skylake_(__m512i max_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(max_i32x16);
    __m256i hi_i32x8 = _mm512_extracti64x4_epi64(max_i32x16, 1);
    __m256i max_i32x8 = _mm256_max_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(max_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(max_i32x8, 1);
    __m128i max_i32x4 = _mm_max_epi32(lo_i32x4, hi_i32x4);
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(max_i32x4);
}

/** @brief Horizontal min of 16 unsigned u32s in a ZMM register. */
NK_INTERNAL nk_u32_t nk_reduce_min_u32x16_skylake_(__m512i min_u32x16) {
    __m256i lo_u32x8 = _mm512_castsi512_si256(min_u32x16);
    __m256i hi_u32x8 = _mm512_extracti64x4_epi64(min_u32x16, 1);
    __m256i min_u32x8 = _mm256_min_epu32(lo_u32x8, hi_u32x8);
    __m128i lo_u32x4 = _mm256_castsi256_si128(min_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(min_u32x8, 1);
    __m128i min_u32x4 = _mm_min_epu32(lo_u32x4, hi_u32x4);
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(min_u32x4);
}

/** @brief Horizontal max of 16 unsigned u32s in a ZMM register. */
NK_INTERNAL nk_u32_t nk_reduce_max_u32x16_skylake_(__m512i max_u32x16) {
    __m256i lo_u32x8 = _mm512_castsi512_si256(max_u32x16);
    __m256i hi_u32x8 = _mm512_extracti64x4_epi64(max_u32x16, 1);
    __m256i max_u32x8 = _mm256_max_epu32(lo_u32x8, hi_u32x8);
    __m128i lo_u32x4 = _mm256_castsi256_si128(max_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(max_u32x8, 1);
    __m128i max_u32x4 = _mm_max_epu32(lo_u32x4, hi_u32x4);
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(max_u32x4);
}

/** @brief Horizontal min of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t nk_reduce_min_i64x8_skylake_(__m512i min_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(min_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(min_i64x8, 1);
    __m256i min_i64x4 = _mm256_min_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(min_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
    __m128i min_i64x2 = _mm_min_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(min_i64x2, min_i64x2);
    __m128i final_i64 = _mm_min_epi64(min_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal max of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t nk_reduce_max_i64x8_skylake_(__m512i max_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(max_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(max_i64x8, 1);
    __m256i max_i64x4 = _mm256_max_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(max_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
    __m128i max_i64x2 = _mm_max_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(max_i64x2, max_i64x2);
    __m128i final_i64 = _mm_max_epi64(max_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal min of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t nk_reduce_min_u64x8_skylake_(__m512i min_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(min_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(min_u64x8, 1);
    __m256i min_u64x4 = _mm256_min_epu64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(min_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
    __m128i min_u64x2 = _mm_min_epu64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(min_u64x2, min_u64x2);
    __m128i final_u64 = _mm_min_epu64(min_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

/** @brief Horizontal max of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t nk_reduce_max_u64x8_skylake_(__m512i max_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(max_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(max_u64x8, 1);
    __m256i max_u64x4 = _mm256_max_epu64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(max_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
    __m128i max_u64x2 = _mm_max_epu64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(max_u64x2, max_u64x2);
    __m128i final_u64 = _mm_max_epu64(max_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

/** @brief Horizontal sum of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t nk_reduce_add_u64x8_skylake_(__m512i sum_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(sum_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(sum_u64x8, 1);
    __m256i sum_u64x4 = _mm256_add_epi64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(sum_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(sum_u64x4, 1);
    __m128i sum_u64x2 = _mm_add_epi64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(sum_u64x2, sum_u64x2);
    __m128i final_u64 = _mm_add_epi64(sum_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

NK_INTERNAL void nk_reduce_add_f32_skylake_contiguous_( //
    nk_f32_t const *data, nk_size_t count, nk_f64_t *result) {
    // Use dual accumulators to hide VADDPD latency (4 cycles) and two 256-bit
    // loads instead of 512-bit load + VEXTRACTF32X8 to reduce Port 5 pressure.
    __m512d sum_low_f64x8 = _mm512_setzero_pd();
    __m512d sum_high_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m256 low_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        __m256 high_f32x8 = _mm256_loadu_ps(data + idx_scalars + 8);
        sum_low_f64x8 = _mm512_add_pd(sum_low_f64x8, _mm512_cvtps_pd(low_f32x8));
        sum_high_f64x8 = _mm512_add_pd(sum_high_f64x8, _mm512_cvtps_pd(high_f32x8));
    }
    __m512d sum_f64x8 = _mm512_add_pd(sum_low_f64x8, sum_high_f64x8);
    // Handle tail with masked load (keep 512-bit + extract for correct masking)
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        __m256 low_f32x8 = _mm512_castps512_ps256(tail_f32x16);
        __m256 high_f32x8 = _mm512_extractf32x8_ps(tail_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(low_f32x8));
        if (remaining > 8) sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(high_f32x8));
    }
    *result = nk_reduce_add_f64x8_skylake_(sum_f64x8);
}

NK_INTERNAL void nk_reduce_add_f32_skylake_gather_(                //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m512i indices_i32x16 = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                _mm512_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 gathered_f32x16 = _mm512_i32gather_ps(indices_i32x16, data + idx_scalars * stride_elements,
                                                     sizeof(nk_f32_t));
        __m256 lo_f32x8 = _mm512_castps512_ps256(gathered_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(gathered_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    nk_f64_t sum = nk_reduce_add_f64x8_skylake_(sum_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(nk_f32_t const *)ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f32_skylake_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Masked load zeros out non-stride elements; zeros don't affect the sum
    __mmask16 stride_mask_m16 = nk_stride_mask_b32x16_(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(stride_mask_m16, data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(data_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(data_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    // Masked tail: combine stride mask with tail mask
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(load_mask_m16, data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(data_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(data_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        if (remaining > 8) sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    *result = nk_reduce_add_f64x8_skylake_(sum_f64x8);
}

NK_PUBLIC void nk_reduce_add_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_add_f32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f32_skylake_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_f32_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f32_skylake_gather_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_f64_skylake_contiguous_( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();
    __m512d absolute_mask_f64x8 = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d term_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, term_f64x8);
        __m512d absolute_sum_f64x8 = _mm512_and_pd(sum_f64x8, absolute_mask_f64x8);
        __m512d absolute_term_f64x8 = _mm512_and_pd(term_f64x8, absolute_mask_f64x8);
        __mmask8 term_bigger_m8 = _mm512_cmp_pd_mask(absolute_term_f64x8, absolute_sum_f64x8, _CMP_GT_OQ);
        // Default: (sum - tentative) + term; where term bigger: (term - tentative) + sum
        __m512d difference_f64x8 = _mm512_sub_pd(sum_f64x8, tentative_f64x8);
        __m512d correction_f64x8 = _mm512_add_pd(difference_f64x8, term_f64x8);
        difference_f64x8 = _mm512_mask_sub_pd(difference_f64x8, term_bigger_m8, term_f64x8, tentative_f64x8);
        correction_f64x8 = _mm512_mask_add_pd(correction_f64x8, term_bigger_m8, difference_f64x8, sum_f64x8);
        compensation_f64x8 = _mm512_add_pd(compensation_f64x8, correction_f64x8);
        sum_f64x8 = tentative_f64x8;
    }
    // Handle tail with masked load + Neumaier (zeros in invalid lanes produce zero corrections)
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d term_f64x8 = _mm512_maskz_loadu_pd(tail_mask_m8, data + idx_scalars);
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, term_f64x8);
        __m512d absolute_sum_f64x8 = _mm512_and_pd(sum_f64x8, absolute_mask_f64x8);
        __m512d absolute_term_f64x8 = _mm512_and_pd(term_f64x8, absolute_mask_f64x8);
        __mmask8 term_bigger_m8 = _mm512_cmp_pd_mask(absolute_term_f64x8, absolute_sum_f64x8, _CMP_GT_OQ);
        __m512d difference_f64x8 = _mm512_sub_pd(sum_f64x8, tentative_f64x8);
        __m512d correction_f64x8 = _mm512_add_pd(difference_f64x8, term_f64x8);
        difference_f64x8 = _mm512_mask_sub_pd(difference_f64x8, term_bigger_m8, term_f64x8, tentative_f64x8);
        correction_f64x8 = _mm512_mask_add_pd(correction_f64x8, term_bigger_m8, difference_f64x8, sum_f64x8);
        compensation_f64x8 = _mm512_add_pd(compensation_f64x8, correction_f64x8);
        sum_f64x8 = tentative_f64x8;
    }
    __m512d total_f64x8 = _mm512_add_pd(sum_f64x8, compensation_f64x8);
    *result = nk_reduce_add_f64x8_skylake_(total_f64x8);
}

NK_INTERNAL void nk_reduce_add_f64_skylake_gather_(                //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f64_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();
    __m512d absolute_mask_f64x8 = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d term_f64x8 = _mm512_i32gather_pd(indices_i32x8, data + idx_scalars * stride_elements, sizeof(nk_f64_t));
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, term_f64x8);
        __m512d absolute_sum_f64x8 = _mm512_and_pd(sum_f64x8, absolute_mask_f64x8);
        __m512d absolute_term_f64x8 = _mm512_and_pd(term_f64x8, absolute_mask_f64x8);
        __mmask8 term_bigger_m8 = _mm512_cmp_pd_mask(absolute_term_f64x8, absolute_sum_f64x8, _CMP_GT_OQ);
        __m512d difference_f64x8 = _mm512_sub_pd(sum_f64x8, tentative_f64x8);
        __m512d correction_f64x8 = _mm512_add_pd(difference_f64x8, term_f64x8);
        difference_f64x8 = _mm512_mask_sub_pd(difference_f64x8, term_bigger_m8, term_f64x8, tentative_f64x8);
        correction_f64x8 = _mm512_mask_add_pd(correction_f64x8, term_bigger_m8, difference_f64x8, sum_f64x8);
        compensation_f64x8 = _mm512_add_pd(compensation_f64x8, correction_f64x8);
        sum_f64x8 = tentative_f64x8;
    }
    __m512d total_f64x8 = _mm512_add_pd(sum_f64x8, compensation_f64x8);
    nk_f64_t sum = nk_reduce_add_f64x8_skylake_(total_f64x8);
    nk_f64_t compensation = 0;
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) {
        nk_f64_t term = *(nk_f64_t const *)ptr, tentative = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - tentative) + term)
                                                                : ((term - tentative) + sum);
        sum = tentative;
    }
    *result = sum + compensation;
}

NK_INTERNAL void nk_reduce_add_f64_skylake_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Masked load zeros out non-stride elements; zeros produce zero corrections
    __mmask8 stride_mask_m8 = nk_stride_mask_b64x8_(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    __m512d compensation_f64x8 = _mm512_setzero_pd();
    __m512d absolute_mask_f64x8 = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d term_f64x8 = _mm512_maskz_loadu_pd(stride_mask_m8, data + idx_scalars);
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, term_f64x8);
        __m512d absolute_sum_f64x8 = _mm512_and_pd(sum_f64x8, absolute_mask_f64x8);
        __m512d absolute_term_f64x8 = _mm512_and_pd(term_f64x8, absolute_mask_f64x8);
        __mmask8 term_bigger_m8 = _mm512_cmp_pd_mask(absolute_term_f64x8, absolute_sum_f64x8, _CMP_GT_OQ);
        __m512d difference_f64x8 = _mm512_sub_pd(sum_f64x8, tentative_f64x8);
        __m512d correction_f64x8 = _mm512_add_pd(difference_f64x8, term_f64x8);
        difference_f64x8 = _mm512_mask_sub_pd(difference_f64x8, term_bigger_m8, term_f64x8, tentative_f64x8);
        correction_f64x8 = _mm512_mask_add_pd(correction_f64x8, term_bigger_m8, difference_f64x8, sum_f64x8);
        compensation_f64x8 = _mm512_add_pd(compensation_f64x8, correction_f64x8);
        sum_f64x8 = tentative_f64x8;
    }
    // Masked tail: combine stride mask with tail mask
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d term_f64x8 = _mm512_maskz_loadu_pd(load_mask_m8, data + idx_scalars);
        __m512d tentative_f64x8 = _mm512_add_pd(sum_f64x8, term_f64x8);
        __m512d absolute_sum_f64x8 = _mm512_and_pd(sum_f64x8, absolute_mask_f64x8);
        __m512d absolute_term_f64x8 = _mm512_and_pd(term_f64x8, absolute_mask_f64x8);
        __mmask8 term_bigger_m8 = _mm512_cmp_pd_mask(absolute_term_f64x8, absolute_sum_f64x8, _CMP_GT_OQ);
        __m512d difference_f64x8 = _mm512_sub_pd(sum_f64x8, tentative_f64x8);
        __m512d correction_f64x8 = _mm512_add_pd(difference_f64x8, term_f64x8);
        difference_f64x8 = _mm512_mask_sub_pd(difference_f64x8, term_bigger_m8, term_f64x8, tentative_f64x8);
        correction_f64x8 = _mm512_mask_add_pd(correction_f64x8, term_bigger_m8, difference_f64x8, sum_f64x8);
        compensation_f64x8 = _mm512_add_pd(compensation_f64x8, correction_f64x8);
        sum_f64x8 = tentative_f64x8;
    }
    __m512d total_f64x8 = _mm512_add_pd(sum_f64x8, compensation_f64x8);
    *result = nk_reduce_add_f64x8_skylake_(total_f64x8);
}

NK_PUBLIC void nk_reduce_add_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_add_f64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f64_skylake_contiguous_(data, count, result);
    else if (stride_elements <= 8) nk_reduce_add_f64_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f64_skylake_gather_(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_f32_skylake_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512 min_f32x16 = _mm512_loadu_ps(data);
    __m512i min_idx_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_idx_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
        current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        // Set masked-out lanes to +inf so they don't affect the min
        __m512 inf_f32x16 = _mm512_set1_ps(__builtin_huge_valf());
        tail_f32x16 = _mm512_mask_mov_ps(inf_f32x16, tail_mask, tail_f32x16);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(tail_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, tail_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
    }

    // Horizontal reduction to find lane with minimum
    nk_f32_t min_val = nk_reduce_min_f32x16_skylake_(min_f32x16);

    // Find the first lane that matches the minimum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_idx_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_f32_skylake_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Masked load with +inf for non-stride elements; track logical indices
    __mmask16 stride_mask_m16 = nk_stride_mask_b32x16_(stride_elements);
    __m512 pos_inf_f32x16 = _mm512_set1_ps(__builtin_huge_valf());
    __m512 min_f32x16 = pos_inf_f32x16;
    __m512i min_idx_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i32x16 = nk_stride_logidx_i32x16_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b32x16_(stride_elements);
    __m512i step_i32x16 = _mm512_set1_epi32((nk_i32_t)elems_per_chunk);

    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_mask_loadu_ps(pos_inf_f32x16, stride_mask_m16, data + idx_scalars);
        __mmask16 lt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask_m16, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask_m16, logical_idx_i32x16);
        logical_idx_i32x16 = _mm512_add_epi32(logical_idx_i32x16, step_i32x16);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_mask_loadu_ps(pos_inf_f32x16, load_mask_m16, data + idx_scalars);
        __mmask16 lt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask_m16, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask_m16, logical_idx_i32x16);
    }

    // Horizontal reduction
    nk_f32_t min_val = nk_reduce_min_f32x16_skylake_(min_f32x16);
    __mmask16 eq_mask_m16 = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m16);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_idx_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_min_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 16)
        nk_reduce_min_f32_skylake_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_min_f32_skylake_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f32_skylake_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512 max_f32x16 = _mm512_loadu_ps(data);
    __m512i max_idx_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_idx_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask, current_idx_i32x16);
        current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        // Set masked-out lanes to -inf so they don't affect the max
        __m512 neg_inf_f32x16 = _mm512_set1_ps(-__builtin_huge_valf());
        tail_f32x16 = _mm512_mask_mov_ps(neg_inf_f32x16, tail_mask, tail_f32x16);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(tail_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, tail_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask, current_idx_i32x16);
    }

    // Horizontal reduction to find lane with maximum
    nk_f32_t max_val = nk_reduce_max_f32x16_skylake_(max_f32x16);

    // Find the first lane that matches the maximum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_idx_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_f32_skylake_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Masked load with -inf for non-stride elements; track logical indices
    __mmask16 stride_mask_m16 = nk_stride_mask_b32x16_(stride_elements);
    __m512 neg_inf_f32x16 = _mm512_set1_ps(-__builtin_huge_valf());
    __m512 max_f32x16 = neg_inf_f32x16;
    __m512i max_idx_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i32x16 = nk_stride_logidx_i32x16_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b32x16_(stride_elements);
    __m512i step_i32x16 = _mm512_set1_epi32((nk_i32_t)elems_per_chunk);

    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_mask_loadu_ps(neg_inf_f32x16, stride_mask_m16, data + idx_scalars);
        __mmask16 gt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask_m16, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask_m16, logical_idx_i32x16);
        logical_idx_i32x16 = _mm512_add_epi32(logical_idx_i32x16, step_i32x16);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_mask_loadu_ps(neg_inf_f32x16, load_mask_m16, data + idx_scalars);
        __mmask16 gt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask_m16, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask_m16, logical_idx_i32x16);
    }

    // Horizontal reduction
    nk_f32_t max_val = nk_reduce_max_f32x16_skylake_(max_f32x16);
    __mmask16 eq_mask_m16 = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m16);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_idx_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_max_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 16)
        nk_reduce_max_f32_skylake_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_max_f32_skylake_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_f64_skylake_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512d min_f64x8 = _mm512_loadu_pd(data);
    __m512i min_idx_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_idx_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask, current_idx_i64x8);
        current_idx_i64x8 = _mm512_add_epi64(current_idx_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        __m512d inf_f64x8 = _mm512_set1_pd(__builtin_huge_val());
        tail_f64x8 = _mm512_mask_mov_pd(inf_f64x8, tail_mask, tail_f64x8);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(tail_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, tail_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask, current_idx_i64x8);
    }

    // Horizontal reduction
    nk_f64_t min_val = nk_reduce_min_f64x8_skylake_(min_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_idx_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_f64_skylake_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Masked load with +inf for non-stride elements; track logical indices
    __mmask8 stride_mask_m8 = nk_stride_mask_b64x8_(stride_elements);
    __m512d pos_inf_f64x8 = _mm512_set1_pd(__builtin_huge_val());
    __m512d min_f64x8 = pos_inf_f64x8;
    __m512i min_idx_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i64x8 = nk_stride_logidx_i64x8_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b64x8_(stride_elements);
    __m512i step_i64x8 = _mm512_set1_epi64((nk_i64_t)elems_per_chunk);

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_mask_loadu_pd(pos_inf_f64x8, stride_mask_m8, data + idx_scalars);
        __mmask8 lt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask_m8, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask_m8, logical_idx_i64x8);
        logical_idx_i64x8 = _mm512_add_epi64(logical_idx_i64x8, step_i64x8);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_mask_loadu_pd(pos_inf_f64x8, load_mask_m8, data + idx_scalars);
        __mmask8 lt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask_m8, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask_m8, logical_idx_i64x8);
    }

    // Horizontal reduction
    nk_f64_t min_val = nk_reduce_min_f64x8_skylake_(min_f64x8);
    __mmask8 eq_mask_m8 = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m8);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_idx_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_min_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_min_f64_skylake_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_min_f64_skylake_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f64_skylake_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512d max_f64x8 = _mm512_loadu_pd(data);
    __m512i max_idx_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_idx_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask, current_idx_i64x8);
        current_idx_i64x8 = _mm512_add_epi64(current_idx_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        __m512d neg_inf_f64x8 = _mm512_set1_pd(-__builtin_huge_val());
        tail_f64x8 = _mm512_mask_mov_pd(neg_inf_f64x8, tail_mask, tail_f64x8);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(tail_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, tail_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask, current_idx_i64x8);
    }

    // Horizontal reduction
    nk_f64_t max_val = nk_reduce_max_f64x8_skylake_(max_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_idx_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_f64_skylake_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Masked load with -inf for non-stride elements; track logical indices
    __mmask8 stride_mask_m8 = nk_stride_mask_b64x8_(stride_elements);
    __m512d neg_inf_f64x8 = _mm512_set1_pd(-__builtin_huge_val());
    __m512d max_f64x8 = neg_inf_f64x8;
    __m512i max_idx_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i64x8 = nk_stride_logidx_i64x8_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b64x8_(stride_elements);
    __m512i step_i64x8 = _mm512_set1_epi64((nk_i64_t)elems_per_chunk);

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_mask_loadu_pd(neg_inf_f64x8, stride_mask_m8, data + idx_scalars);
        __mmask8 gt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask_m8, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask_m8, logical_idx_i64x8);
        logical_idx_i64x8 = _mm512_add_epi64(logical_idx_i64x8, step_i64x8);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_mask_loadu_pd(neg_inf_f64x8, load_mask_m8, data + idx_scalars);
        __mmask8 gt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask_m8, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask_m8, logical_idx_i64x8);
    }

    // Horizontal reduction
    nk_f64_t max_val = nk_reduce_max_f64x8_skylake_(max_f64x8);
    __mmask8 eq_mask_m8 = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m8);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_idx_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_max_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_max_f64_skylake_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_max_f64_skylake_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        // Widen lower 32 bytes: i8 → i16 → i32 → i64
        __m256i lo_i8x32 = _mm512_castsi512_si256(data_i8x64);
        __m256i hi_i8x32 = _mm512_extracti64x4_epi64(data_i8x64, 1);
        // Process lo_i8x32
        __m512i lo_i16x32 = _mm512_cvtepi8_epi16(lo_i8x32);
        __m256i lo_lo_i16x16 = _mm512_castsi512_si256(lo_i16x32);
        __m256i lo_hi_i16x16 = _mm512_extracti64x4_epi64(lo_i16x32, 1);
        __m512i lo_lo_i32x16 = _mm512_cvtepi16_epi32(lo_lo_i16x16);
        __m512i lo_hi_i32x16 = _mm512_cvtepi16_epi32(lo_hi_i16x16);
        __m256i a_i32x8 = _mm512_castsi512_si256(lo_lo_i32x16);
        __m256i b_i32x8 = _mm512_extracti64x4_epi64(lo_lo_i32x16, 1);
        __m256i c_i32x8 = _mm512_castsi512_si256(lo_hi_i32x16);
        __m256i d_i32x8 = _mm512_extracti64x4_epi64(lo_hi_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(a_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(b_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(c_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(d_i32x8));
        // Process hi_i8x32
        __m512i hi_i16x32 = _mm512_cvtepi8_epi16(hi_i8x32);
        __m256i hi_lo_i16x16 = _mm512_castsi512_si256(hi_i16x32);
        __m256i hi_hi_i16x16 = _mm512_extracti64x4_epi64(hi_i16x32, 1);
        __m512i hi_lo_i32x16 = _mm512_cvtepi16_epi32(hi_lo_i16x16);
        __m512i hi_hi_i32x16 = _mm512_cvtepi16_epi32(hi_hi_i16x16);
        __m256i e_i32x8 = _mm512_castsi512_si256(hi_lo_i32x16);
        __m256i f_i32x8 = _mm512_extracti64x4_epi64(hi_lo_i32x16, 1);
        __m256i g_i32x8 = _mm512_castsi512_si256(hi_hi_i32x16);
        __m256i h_i32x8 = _mm512_extracti64x4_epi64(hi_hi_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(e_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(f_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(g_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(h_i32x8));
    }
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        __m256i lo_u8x32 = _mm512_castsi512_si256(data_u8x64);
        __m256i hi_u8x32 = _mm512_extracti64x4_epi64(data_u8x64, 1);
        // Process lo_u8x32
        __m512i lo_u16x32 = _mm512_cvtepu8_epi16(lo_u8x32);
        __m256i lo_lo_u16x16 = _mm512_castsi512_si256(lo_u16x32);
        __m256i lo_hi_u16x16 = _mm512_extracti64x4_epi64(lo_u16x32, 1);
        __m512i lo_lo_u32x16 = _mm512_cvtepu16_epi32(lo_lo_u16x16);
        __m512i lo_hi_u32x16 = _mm512_cvtepu16_epi32(lo_hi_u16x16);
        __m256i a_u32x8 = _mm512_castsi512_si256(lo_lo_u32x16);
        __m256i b_u32x8 = _mm512_extracti64x4_epi64(lo_lo_u32x16, 1);
        __m256i c_u32x8 = _mm512_castsi512_si256(lo_hi_u32x16);
        __m256i d_u32x8 = _mm512_extracti64x4_epi64(lo_hi_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(a_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(b_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(c_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(d_u32x8));
        // Process hi_u8x32
        __m512i hi_u16x32 = _mm512_cvtepu8_epi16(hi_u8x32);
        __m256i hi_lo_u16x16 = _mm512_castsi512_si256(hi_u16x32);
        __m256i hi_hi_u16x16 = _mm512_extracti64x4_epi64(hi_u16x32, 1);
        __m512i hi_lo_u32x16 = _mm512_cvtepu16_epi32(hi_lo_u16x16);
        __m512i hi_hi_u32x16 = _mm512_cvtepu16_epi32(hi_hi_u16x16);
        __m256i e_u32x8 = _mm512_castsi512_si256(hi_lo_u32x16);
        __m256i f_u32x8 = _mm512_extracti64x4_epi64(hi_lo_u32x16, 1);
        __m256i g_u32x8 = _mm512_castsi512_si256(hi_hi_u32x16);
        __m256i h_u32x8 = _mm512_extracti64x4_epi64(hi_hi_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(e_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(f_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(g_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(h_u32x8));
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        __m256i lo_i16x16 = _mm512_castsi512_si256(data_i16x32);
        __m256i hi_i16x16 = _mm512_extracti64x4_epi64(data_i16x32, 1);
        __m512i lo_i32x16 = _mm512_cvtepi16_epi32(lo_i16x16);
        __m512i hi_i32x16 = _mm512_cvtepi16_epi32(hi_i16x16);
        __m256i a_i32x8 = _mm512_castsi512_si256(lo_i32x16);
        __m256i b_i32x8 = _mm512_extracti64x4_epi64(lo_i32x16, 1);
        __m256i c_i32x8 = _mm512_castsi512_si256(hi_i32x16);
        __m256i d_i32x8 = _mm512_extracti64x4_epi64(hi_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(a_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(b_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(c_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(d_i32x8));
    }
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        __m256i lo_u16x16 = _mm512_castsi512_si256(data_u16x32);
        __m256i hi_u16x16 = _mm512_extracti64x4_epi64(data_u16x32, 1);
        __m512i lo_u32x16 = _mm512_cvtepu16_epi32(lo_u16x16);
        __m512i hi_u32x16 = _mm512_cvtepu16_epi32(hi_u16x16);
        __m256i a_u32x8 = _mm512_castsi512_si256(lo_u32x16);
        __m256i b_u32x8 = _mm512_extracti64x4_epi64(lo_u32x16, 1);
        __m256i c_u32x8 = _mm512_castsi512_si256(hi_u32x16);
        __m256i d_u32x8 = _mm512_extracti64x4_epi64(hi_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(a_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(b_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(c_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(d_u32x8));
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i8_skylake_strided_(                  //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    // Masked load zeros out non-stride elements; zeros don't affect sum
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        // Widen with sign extension: i8 → i16 → i32 → i64
        __m256i lo_i8x32 = _mm512_castsi512_si256(data_i8x64);
        __m256i hi_i8x32 = _mm512_extracti64x4_epi64(data_i8x64, 1);
        __m512i lo_i16x32 = _mm512_cvtepi8_epi16(lo_i8x32);
        __m512i hi_i16x32 = _mm512_cvtepi8_epi16(hi_i8x32);
        // Sum all 64 values via widening to i64
        __m256i a_i16x16 = _mm512_castsi512_si256(lo_i16x32);
        __m256i b_i16x16 = _mm512_extracti64x4_epi64(lo_i16x32, 1);
        __m256i c_i16x16 = _mm512_castsi512_si256(hi_i16x32);
        __m256i d_i16x16 = _mm512_extracti64x4_epi64(hi_i16x32, 1);
        __m512i a_i32x16 = _mm512_cvtepi16_epi32(a_i16x16);
        __m512i b_i32x16 = _mm512_cvtepi16_epi32(b_i16x16);
        __m512i c_i32x16 = _mm512_cvtepi16_epi32(c_i16x16);
        __m512i d_i32x16 = _mm512_cvtepi16_epi32(d_i16x16);
        // Pairwise add i32x16 → i32x16 (horizontal), then widen to i64
        __m512i ab_i32x16 = _mm512_add_epi32(a_i32x16, b_i32x16);
        __m512i cd_i32x16 = _mm512_add_epi32(c_i32x16, d_i32x16);
        __m512i abcd_i32x16 = _mm512_add_epi32(ab_i32x16, cd_i32x16);
        __m256i lo_i32x8 = _mm512_castsi512_si256(abcd_i32x16);
        __m256i hi_i32x8 = _mm512_extracti64x4_epi64(abcd_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(lo_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(hi_i32x8));
    }
    // Scalar tail for remaining elements
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_i8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u8_skylake_strided_(                  //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        // Widen with zero extension: u8 → u16 → u32 → u64
        __m256i lo_u8x32 = _mm512_castsi512_si256(data_u8x64);
        __m256i hi_u8x32 = _mm512_extracti64x4_epi64(data_u8x64, 1);
        __m512i lo_u16x32 = _mm512_cvtepu8_epi16(lo_u8x32);
        __m512i hi_u16x32 = _mm512_cvtepu8_epi16(hi_u8x32);
        __m256i a_u16x16 = _mm512_castsi512_si256(lo_u16x32);
        __m256i b_u16x16 = _mm512_extracti64x4_epi64(lo_u16x32, 1);
        __m256i c_u16x16 = _mm512_castsi512_si256(hi_u16x32);
        __m256i d_u16x16 = _mm512_extracti64x4_epi64(hi_u16x32, 1);
        __m512i a_u32x16 = _mm512_cvtepu16_epi32(a_u16x16);
        __m512i b_u32x16 = _mm512_cvtepu16_epi32(b_u16x16);
        __m512i c_u32x16 = _mm512_cvtepu16_epi32(c_u16x16);
        __m512i d_u32x16 = _mm512_cvtepu16_epi32(d_u16x16);
        __m512i ab_u32x16 = _mm512_add_epi32(a_u32x16, b_u32x16);
        __m512i cd_u32x16 = _mm512_add_epi32(c_u32x16, d_u32x16);
        __m512i abcd_u32x16 = _mm512_add_epi32(ab_u32x16, cd_u32x16);
        __m256i lo_u32x8 = _mm512_castsi512_si256(abcd_u32x16);
        __m256i hi_u32x8 = _mm512_extracti64x4_epi64(abcd_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(lo_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(hi_u32x8));
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i16_skylake_strided_(                  //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
        __m256i lo_i16x16 = _mm512_castsi512_si256(data_i16x32);
        __m256i hi_i16x16 = _mm512_extracti64x4_epi64(data_i16x32, 1);
        __m512i lo_i32x16 = _mm512_cvtepi16_epi32(lo_i16x16);
        __m512i hi_i32x16 = _mm512_cvtepi16_epi32(hi_i16x16);
        __m512i sum_i32x16 = _mm512_add_epi32(lo_i32x16, hi_i32x16);
        __m256i lo_i32x8 = _mm512_castsi512_si256(sum_i32x16);
        __m256i hi_i32x8 = _mm512_extracti64x4_epi64(sum_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(lo_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(hi_i32x8));
    }
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_i16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u16_skylake_strided_(                  //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 32 <= total_scalars; idx_scalars += 32) {
        __m512i data_u16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
        __m256i lo_u16x16 = _mm512_castsi512_si256(data_u16x32);
        __m256i hi_u16x16 = _mm512_extracti64x4_epi64(data_u16x32, 1);
        __m512i lo_u32x16 = _mm512_cvtepu16_epi32(lo_u16x16);
        __m512i hi_u32x16 = _mm512_cvtepu16_epi32(hi_u16x16);
        __m512i sum_u32x16 = _mm512_add_epi32(lo_u32x16, hi_u32x16);
        __m256i lo_u32x8 = _mm512_castsi512_si256(sum_u32x16);
        __m256i hi_u32x8 = _mm512_extracti64x4_epi64(sum_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(lo_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(hi_u32x8));
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    nk_u16_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        __m256i lo_i32x8 = _mm512_castsi512_si256(data_i32x16);
        __m256i hi_i32x8 = _mm512_extracti64x4_epi64(data_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(lo_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(hi_i32x8));
    }
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        __m256i lo_u32x8 = _mm512_castsi512_si256(data_u32x16);
        __m256i hi_u32x8 = _mm512_extracti64x4_epi64(data_u32x16, 1);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(lo_u32x8));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_cvtepu32_epi64(hi_u32x8));
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, data_i64x8);
    }
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, data_u64x8);
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_min_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *min_value, nk_size_t *min_index) {
    __m512i min_i8x64 = _mm512_set1_epi8(127);
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        min_i8x64 = _mm512_min_epi8(min_i8x64, data_i8x64);
    }
    nk_i8_t min_val = nk_reduce_min_i8x64_skylake_(min_i8x64);
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

NK_INTERNAL void nk_reduce_max_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *max_value, nk_size_t *max_index) {
    __m512i max_i8x64 = _mm512_set1_epi8(-128);
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        max_i8x64 = _mm512_max_epi8(max_i8x64, data_i8x64);
    }
    nk_i8_t max_val = nk_reduce_max_i8x64_skylake_(max_i8x64);
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

NK_INTERNAL void nk_reduce_min_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *min_value, nk_size_t *min_index) {
    __m512i min_u8x64 = _mm512_set1_epi8((char)255);
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        min_u8x64 = _mm512_min_epu8(min_u8x64, data_u8x64);
    }
    nk_u8_t min_val = nk_reduce_min_u8x64_skylake_(min_u8x64);
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

NK_INTERNAL void nk_reduce_max_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *max_value, nk_size_t *max_index) {
    __m512i max_u8x64 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        max_u8x64 = _mm512_max_epu8(max_u8x64, data_u8x64);
    }
    nk_u8_t max_val = nk_reduce_max_u8x64_skylake_(max_u8x64);
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

NK_INTERNAL void nk_reduce_min_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *min_value, nk_size_t *min_index) {
    __m512i min_i16x32 = _mm512_set1_epi16(32767);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        min_i16x32 = _mm512_min_epi16(min_i16x32, data_i16x32);
    }
    nk_i16_t min_val = nk_reduce_min_i16x32_skylake_(min_i16x32);
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

NK_INTERNAL void nk_reduce_max_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *max_value, nk_size_t *max_index) {
    __m512i max_i16x32 = _mm512_set1_epi16(-32768);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        max_i16x32 = _mm512_max_epi16(max_i16x32, data_i16x32);
    }
    nk_i16_t max_val = nk_reduce_max_i16x32_skylake_(max_i16x32);
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

NK_INTERNAL void nk_reduce_min_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *min_value, nk_size_t *min_index) {
    __m512i min_u16x32 = _mm512_set1_epi16((short)65535);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        min_u16x32 = _mm512_min_epu16(min_u16x32, data_u16x32);
    }
    nk_u16_t min_val = nk_reduce_min_u16x32_skylake_(min_u16x32);
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

NK_INTERNAL void nk_reduce_max_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *max_value, nk_size_t *max_index) {
    __m512i max_u16x32 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        max_u16x32 = _mm512_max_epu16(max_u16x32, data_u16x32);
    }
    nk_u16_t max_val = nk_reduce_max_u16x32_skylake_(max_u16x32);
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

NK_INTERNAL void nk_reduce_min_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *min_value, nk_size_t *min_index) {
    __m512i min_i32x16 = _mm512_set1_epi32(NK_I32_MAX);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        min_i32x16 = _mm512_min_epi32(min_i32x16, data_i32x16);
    }
    nk_i32_t min_val = nk_reduce_min_i32x16_skylake_(min_i32x16);
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

NK_INTERNAL void nk_reduce_max_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *max_value, nk_size_t *max_index) {
    __m512i max_i32x16 = _mm512_set1_epi32(NK_I32_MIN);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        max_i32x16 = _mm512_max_epi32(max_i32x16, data_i32x16);
    }
    nk_i32_t max_val = nk_reduce_max_i32x16_skylake_(max_i32x16);
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

NK_INTERNAL void nk_reduce_min_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *min_value, nk_size_t *min_index) {
    __m512i min_u32x16 = _mm512_set1_epi32((nk_i32_t)NK_U32_MAX);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        min_u32x16 = _mm512_min_epu32(min_u32x16, data_u32x16);
    }
    nk_u32_t min_val = nk_reduce_min_u32x16_skylake_(min_u32x16);
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

NK_INTERNAL void nk_reduce_max_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *max_value, nk_size_t *max_index) {
    __m512i max_u32x16 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        max_u32x16 = _mm512_max_epu32(max_u32x16, data_u32x16);
    }
    nk_u32_t max_val = nk_reduce_max_u32x16_skylake_(max_u32x16);
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

NK_INTERNAL void nk_reduce_min_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *min_value, nk_size_t *min_index) {
    __m512i min_i64x8 = _mm512_set1_epi64(NK_I64_MAX);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        min_i64x8 = _mm512_min_epi64(min_i64x8, data_i64x8);
    }
    nk_i64_t min_val = nk_reduce_min_i64x8_skylake_(min_i64x8);
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

NK_INTERNAL void nk_reduce_max_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *max_value, nk_size_t *max_index) {
    __m512i max_i64x8 = _mm512_set1_epi64(NK_I64_MIN);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        max_i64x8 = _mm512_max_epi64(max_i64x8, data_i64x8);
    }
    nk_i64_t max_val = nk_reduce_max_i64x8_skylake_(max_i64x8);
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

NK_INTERNAL void nk_reduce_min_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *min_value, nk_size_t *min_index) {
    __m512i min_u64x8 = _mm512_set1_epi64((nk_i64_t)NK_U64_MAX);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        min_u64x8 = _mm512_min_epu64(min_u64x8, data_u64x8);
    }
    nk_u64_t min_val = nk_reduce_min_u64x8_skylake_(min_u64x8);
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

NK_INTERNAL void nk_reduce_max_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *max_value, nk_size_t *max_index) {
    __m512i max_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        max_u64x8 = _mm512_max_epu64(max_u64x8, data_u64x8);
    }
    nk_u64_t max_val = nk_reduce_max_u64x8_skylake_(max_u64x8);
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

NK_PUBLIC void nk_reduce_add_i8_skylake(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_add_i8_skylake_contiguous_(data, count, result);
    else if (stride_bytes % sizeof(nk_i8_t) == 0 && stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_add_i8_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u8_skylake(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_add_u8_skylake_contiguous_(data, count, result);
    else if (stride_bytes % sizeof(nk_u8_t) == 0 && stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_add_u8_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u8_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i16_skylake(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_add_i16_skylake_contiguous_(data, count, result);
    else if (stride_bytes % sizeof(nk_i16_t) == 0 && stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_add_i16_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i16_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u16_skylake(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_add_u16_skylake_contiguous_(data, count, result);
    else if (stride_bytes % sizeof(nk_u16_t) == 0 && stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_add_u16_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u16_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i32_skylake(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_add_i32_skylake_contiguous_(data, count, result);
    else nk_reduce_add_i32_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u32_skylake(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_add_u32_skylake_contiguous_(data, count, result);
    else nk_reduce_add_u32_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i64_skylake(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_add_i64_skylake_contiguous_(data, count, result);
    else nk_reduce_add_i64_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u64_skylake(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_add_u64_skylake_contiguous_(data, count, result);
    else nk_reduce_add_u64_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_min_i8_skylake(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_min_i8_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i8_skylake(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i8_t)) nk_reduce_max_i8_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u8_skylake(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_min_u8_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u8_skylake(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u8_t)) nk_reduce_max_u8_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i16_skylake(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_min_i16_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i16_skylake(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i16_t)) nk_reduce_max_i16_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u16_skylake(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_min_u16_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u16_skylake(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u16_t)) nk_reduce_max_u16_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i32_skylake(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_min_i32_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i32_skylake(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i32_t)) nk_reduce_max_i32_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u32_skylake(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_min_u32_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u32_skylake(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u32_t)) nk_reduce_max_u32_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i64_skylake(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_min_i64_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i64_skylake(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i64_t)) nk_reduce_max_i64_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u64_skylake(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_min_u64_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u64_skylake(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u64_t)) nk_reduce_max_u64_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_REDUCE_SKYLAKE_H
