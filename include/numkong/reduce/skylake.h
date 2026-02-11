/**
 *  @brief SIMD-accelerated Vector Reductions for Skylake.
 *  @file include/numkong/reduce/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section float_as_int Float-as-Int Comparison Trick
 *
 *  FP8 min/max operations use direct byte comparison instead of F32 upcasting.
 *  IEEE-like floats have the property that for same-sign values, the unsigned
 *  bit pattern ordering matches the numeric ordering. To extend this to mixed
 *  signs, we transform to "comparable form":
 *
 *  - Positive FP8 (sign=0): XOR with 0x80 → maps [0x00, 0x7F] to [0x80, 0xFF]
 *  - Negative FP8 (sign=1): XOR with 0xFF → maps [0x80, 0xFF] to [0x7F, 0x00]
 *
 *  After transformation, unsigned byte comparison preserves FP8 numeric order:
 *
 *      max_positive → 0xFE (highest)    [E4M3: 0x7E → 0xFE, E5M2: 0x7B → 0xFB]
 *      +0           → 0x80
 *      -0           → 0x7F
 *      max_negative → 0x01 (lowest)
 *      NaN (E4M3)   → 0xFF              [0x7F transforms to 0xFF]
 *
 *  NaN Handling:
 *
 *  - E4M3: Single NaN encoding per sign (0x7F, 0xFF) → comparable form 0xFF, 0x00
 *  - E5M2: Multiple NaNs (0x7D-0x7F, 0xFD-0xFF) → use threshold comparison
 *
 *  @section skylake_reduce_instructions Key AVX-512 Reduction Instructions
 *
 *      Intrinsic                       Instruction                     SKX Lat  SKX Ports   Zen4 Lat  Zen4 Ports
 *      _mm512_sad_epu8                 VPSADBW (ZMM, ZMM, ZMM)         3cy     p5           3cy      p01
 *      _mm512_madd_epi16               VPMADDWD (ZMM, ZMM, ZMM)        5cy     p05          3cy      p01
 *      _mm512_cvtepi8_epi16            VPMOVSXBW (ZMM, YMM)            3cy     p5           4cy      p12
 *      _mm512_cvtepi16_epi32           VPMOVSXWD (ZMM, YMM)            3cy     p5           4cy      p12
 *      _mm512_cvtepi32_epi64           VPMOVSXDQ (ZMM, YMM)            3cy     p5           4cy      p12
 *      _mm512_extracti64x4_epi64       VEXTRACTI64X4 (YMM, ZMM, I8)    3cy     p5           1cy      p0123
 *      _mm512_extractf32x8_ps          VEXTRACTF32X8 (YMM, ZMM, I8)    3cy     p5           1cy      p0123
 *      _mm256_extractf128_ps           VEXTRACTF128 (XMM, YMM, I8)     3cy     p5           1cy      p0123
 *      _mm256_hadd_ps                  VHADDPS (YMM, YMM, YMM)         7cy     p01+p5       -        -
 *      _mm512_cmpneq_epi64_mask        VPCMPQ (K, ZMM, ZMM, I8)        3cy     p5           1cy      p01
 *
 *  Horizontal reductions require cross-lane shuffles that bottleneck on port 5. The full ZMM to scalar
 *  reduction takes 15-18 cycles via extract-and-add sequences. Using dual accumulators amortizes this
 *  cost over larger input batches. Skylake-X server chips benefit from wider execution resources.
 */
#ifndef NK_REDUCE_SKYLAKE_H
#define NK_REDUCE_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_e4m3_to_f32_serial`
#include "numkong/cast/skylake.h"  // `nk_e4m3x8_to_f32x8_skylake_`

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
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_si128(min_i8x16, 2));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_si128(min_i8x16, 1));
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
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_si128(max_i8x16, 2));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_si128(max_i8x16, 1));
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
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_si128(min_u8x16, 2));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_si128(min_u8x16, 1));
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
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_si128(max_u8x16, 2));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_si128(max_u8x16, 1));
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
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_srli_si128(min_i16x8, 2));
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
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_srli_si128(max_i16x8, 2));
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
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_srli_si128(min_u16x8, 2));
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
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_srli_si128(max_u16x8, 2));
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

/**
 *  @brief Convert 64× FP8 bytes to unsigned-comparable form for ordering.
 *
 *  Transforms FP8 bit patterns so that unsigned byte comparison (VPCMPUB)
 *  preserves the correct FP8 numeric ordering across positive and negative values.
 *
 *  @param raw_i8x64 Raw FP8 bytes (E4M3 or E5M2 - same transformation)
 *  @return Transformed bytes suitable for unsigned comparison
 *
 *  @note Same function works for both E4M3 and E5M2 (sign bit position identical)
 *  @note Port usage: 1× VPTESTMB (p5), 1× mask_mov (p05), 1× VPXORD (p05) = 3 uops
 */
NK_INTERNAL __m512i nk_fp8x64_to_u8x64_comparable_skylake_(__m512i raw_i8x64) {
    __mmask64 neg_m64 = _mm512_test_epi8_mask(raw_i8x64, _mm512_set1_epi8((char)0x80));
    __m512i pos_xor_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i neg_xor_i8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(raw_i8x64, xor_i8x64);
}

/**
 *  @brief Convert 64× comparable bytes back to FP8 format.
 *
 *  Reverses the transformation applied by nk_fp8x64_to_u8x64_comparable_skylake_.
 *  Values < 0x80 in comparable form were originally negative FP8.
 *
 *  @param cmp_i8x64 Bytes in comparable form
 *  @return Original FP8 bytes (E4M3 or E5M2)
 *
 *  @note Port usage: 1× VPCMPUB (p5), 1× mask_mov (p05), 1× VPXORD (p05) = 3 uops
 */
NK_INTERNAL __m512i nk_u8x64_comparable_to_fp8x64_skylake_(__m512i cmp_i8x64) {
    __mmask64 was_neg_m64 = _mm512_cmplt_epu8_mask(cmp_i8x64, _mm512_set1_epi8((char)0x80));
    __m512i neg_xor_i8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i pos_xor_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, was_neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(cmp_i8x64, xor_i8x64);
}

/**
 *  @brief IEEE-compliant min selection mask for E4M3 vectors in comparable form.
 *
 *  Returns mask indicating which lanes should select from 'a' to get element-wise
 *  minimum with IEEE NaN semantics: min(x, NaN) = x, min(NaN, NaN) = NaN.
 *
 *  @param a_cmp_u8x64 First operand in comparable form
 *  @param b_cmp_u8x64 Second operand in comparable form
 *  @param nan_cmp_u8x64 NaN value in comparable form (0xFF for E4M3)
 *  @return Mask where 1 = select a, 0 = select b
 *
 *  Usage: min = _mm512_mask_mov_epi8(b_cmp, mask, a_cmp);
 *
 *  @note Port usage: 2× VPCMPEQB (p5), 1× VPCMPUB (p5), 4× mask ops (p05) = ~5 uops
 */
NK_INTERNAL __mmask64 nk_min_mask_e4m3x64_skylake_( //
    __m512i a_cmp_u8x64, __m512i b_cmp_u8x64, __m512i nan_cmp_u8x64) {
    __mmask64 a_nan_m64 = _mm512_cmpeq_epi8_mask(a_cmp_u8x64, nan_cmp_u8x64);
    __mmask64 b_nan_m64 = _mm512_cmpeq_epi8_mask(b_cmp_u8x64, nan_cmp_u8x64);
    __mmask64 lt_m64 = _mm512_cmplt_epu8_mask(a_cmp_u8x64, b_cmp_u8x64);
    // Select a if: a is not NaN AND (a < b OR b is NaN)
    return _kand_mask64(_knot_mask64(a_nan_m64), _kor_mask64(lt_m64, b_nan_m64));
}

/**
 *  @brief IEEE-compliant max selection mask for E4M3 vectors in comparable form.
 *
 *  Returns mask indicating which lanes should select from 'a' to get element-wise
 *  maximum with IEEE NaN semantics: max(x, NaN) = x, max(NaN, NaN) = NaN.
 *
 *  @param a_cmp_u8x64 First operand in comparable form
 *  @param b_cmp_u8x64 Second operand in comparable form
 *  @param nan_cmp_u8x64 NaN value in comparable form (0xFF for E4M3)
 *  @return Mask where 1 = select a, 0 = select b
 *
 *  Usage: max = _mm512_mask_mov_epi8(b_cmp, mask, a_cmp);
 */
NK_INTERNAL __mmask64 nk_max_mask_e4m3x64_skylake_( //
    __m512i a_cmp_u8x64, __m512i b_cmp_u8x64, __m512i nan_cmp_u8x64) {
    __mmask64 a_nan_m64 = _mm512_cmpeq_epi8_mask(a_cmp_u8x64, nan_cmp_u8x64);
    __mmask64 b_nan_m64 = _mm512_cmpeq_epi8_mask(b_cmp_u8x64, nan_cmp_u8x64);
    __mmask64 gt_m64 = _mm512_cmpgt_epu8_mask(a_cmp_u8x64, b_cmp_u8x64);
    // Select a if: a is not NaN AND (a > b OR b is NaN)
    return _kand_mask64(_knot_mask64(a_nan_m64), _kor_mask64(gt_m64, b_nan_m64));
}

/**
 *  @brief IEEE-compliant min selection mask for E5M2 vectors in comparable form.
 *
 *  E5M2 has multiple NaN encodings (0x7D-0x7F per sign), requiring threshold
 *  comparison instead of equality. In comparable form, NaNs map to >= 0xFD.
 *
 *  @param a_cmp_u8x64 First operand in comparable form
 *  @param b_cmp_u8x64 Second operand in comparable form
 *  @param nan_threshold_cmp_u8x64 NaN threshold in comparable form (0xFD for E5M2)
 *  @return Mask where 1 = select a, 0 = select b
 */
NK_INTERNAL __mmask64 nk_min_mask_e5m2x64_skylake_( //
    __m512i a_cmp_u8x64, __m512i b_cmp_u8x64, __m512i nan_threshold_cmp_u8x64) {
    __mmask64 a_nan_m64 = _mm512_cmpge_epu8_mask(a_cmp_u8x64, nan_threshold_cmp_u8x64);
    __mmask64 b_nan_m64 = _mm512_cmpge_epu8_mask(b_cmp_u8x64, nan_threshold_cmp_u8x64);
    __mmask64 lt_m64 = _mm512_cmplt_epu8_mask(a_cmp_u8x64, b_cmp_u8x64);
    return _kand_mask64(_knot_mask64(a_nan_m64), _kor_mask64(lt_m64, b_nan_m64));
}

/**
 *  @brief IEEE-compliant max selection mask for E5M2 vectors in comparable form.
 *
 *  @param a_cmp_u8x64 First operand in comparable form
 *  @param b_cmp_u8x64 Second operand in comparable form
 *  @param nan_threshold_cmp_u8x64 NaN threshold in comparable form (0xFD for E5M2)
 *  @return Mask where 1 = select a, 0 = select b
 */
NK_INTERNAL __mmask64 nk_max_mask_e5m2x64_skylake_( //
    __m512i a_cmp_u8x64, __m512i b_cmp_u8x64, __m512i nan_threshold_cmp_u8x64) {
    __mmask64 a_nan_m64 = _mm512_cmpge_epu8_mask(a_cmp_u8x64, nan_threshold_cmp_u8x64);
    __mmask64 b_nan_m64 = _mm512_cmpge_epu8_mask(b_cmp_u8x64, nan_threshold_cmp_u8x64);
    __mmask64 gt_m64 = _mm512_cmpgt_epu8_mask(a_cmp_u8x64, b_cmp_u8x64);
    return _kand_mask64(_knot_mask64(a_nan_m64), _kor_mask64(gt_m64, b_nan_m64));
}

/** @brief Horizontal argmin: returns index of first minimum unsigned byte in ZMM register. */
NK_INTERNAL nk_size_t nk_argmin_u8x64_skylake_(__m512i data_u8x64) {
    nk_u8_t min_val = nk_reduce_min_u8x64_skylake_(data_u8x64);
    __mmask64 eq_m64 = _mm512_cmpeq_epi8_mask(data_u8x64, _mm512_set1_epi8((char)min_val));
    return (nk_size_t)_tzcnt_u64(eq_m64);
}

/** @brief Horizontal argmax: returns index of first maximum unsigned byte in ZMM register. */
NK_INTERNAL nk_size_t nk_argmax_u8x64_skylake_(__m512i data_u8x64) {
    nk_u8_t max_val = nk_reduce_max_u8x64_skylake_(data_u8x64);
    __mmask64 eq_m64 = _mm512_cmpeq_epi8_mask(data_u8x64, _mm512_set1_epi8((char)max_val));
    return (nk_size_t)_tzcnt_u64(eq_m64);
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
    __m512i min_index_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_index_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask, current_index_i32x16);
        current_index_i32x16 = _mm512_add_epi32(current_index_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        // Set masked-out lanes to +inf so they don't affect the min
        __m512 inf_f32x16 = _mm512_set1_ps(NK_F32_MAX);
        tail_f32x16 = _mm512_mask_mov_ps(inf_f32x16, tail_mask, tail_f32x16);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(tail_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, tail_f32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask, current_index_i32x16);
    }

    // Horizontal reduction to find lane with minimum
    nk_f32_t min_val = nk_reduce_min_f32x16_skylake_(min_f32x16);

    // Find the first lane that matches the minimum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_index_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_f32_skylake_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Masked load with +inf for non-stride elements; track logical indices
    __mmask16 stride_mask_m16 = nk_stride_mask_b32x16_(stride_elements);
    __m512 pos_inf_f32x16 = _mm512_set1_ps(NK_F32_MAX);
    __m512 min_f32x16 = pos_inf_f32x16;
    __m512i min_index_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_index_i32x16 = nk_stride_logidx_i32x16_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b32x16_(stride_elements);
    __m512i step_i32x16 = _mm512_set1_epi32((nk_i32_t)elems_per_chunk);

    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_mask_loadu_ps(pos_inf_f32x16, stride_mask_m16, data + idx_scalars);
        __mmask16 lt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask_m16, data_f32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask_m16, logical_index_i32x16);
        logical_index_i32x16 = _mm512_add_epi32(logical_index_i32x16, step_i32x16);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_mask_loadu_ps(pos_inf_f32x16, load_mask_m16, data + idx_scalars);
        __mmask16 lt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask_m16, data_f32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask_m16, logical_index_i32x16);
    }

    // Horizontal reduction
    nk_f32_t min_val = nk_reduce_min_f32x16_skylake_(min_f32x16);
    __mmask16 eq_mask_m16 = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m16);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_index_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_min_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
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
    __m512i max_index_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_index_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, data_f32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask, current_index_i32x16);
        current_index_i32x16 = _mm512_add_epi32(current_index_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        // Set masked-out lanes to -inf so they don't affect the max
        __m512 neg_inf_f32x16 = _mm512_set1_ps(NK_F32_MIN);
        tail_f32x16 = _mm512_mask_mov_ps(neg_inf_f32x16, tail_mask, tail_f32x16);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(tail_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, tail_f32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask, current_index_i32x16);
    }

    // Horizontal reduction to find lane with maximum
    nk_f32_t max_val = nk_reduce_max_f32x16_skylake_(max_f32x16);

    // Find the first lane that matches the maximum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_index_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_f32_skylake_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Masked load with -inf for non-stride elements; track logical indices
    __mmask16 stride_mask_m16 = nk_stride_mask_b32x16_(stride_elements);
    __m512 neg_inf_f32x16 = _mm512_set1_ps(NK_F32_MIN);
    __m512 max_f32x16 = neg_inf_f32x16;
    __m512i max_index_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_index_i32x16 = nk_stride_logidx_i32x16_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b32x16_(stride_elements);
    __m512i step_i32x16 = _mm512_set1_epi32((nk_i32_t)elems_per_chunk);

    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_mask_loadu_ps(neg_inf_f32x16, stride_mask_m16, data + idx_scalars);
        __mmask16 gt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask_m16, data_f32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask_m16, logical_index_i32x16);
        logical_index_i32x16 = _mm512_add_epi32(logical_index_i32x16, step_i32x16);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_mask_loadu_ps(neg_inf_f32x16, load_mask_m16, data + idx_scalars);
        __mmask16 gt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask_m16, data_f32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask_m16, logical_index_i32x16);
    }

    // Horizontal reduction
    nk_f32_t max_val = nk_reduce_max_f32x16_skylake_(max_f32x16);
    __mmask16 eq_mask_m16 = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m16);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_index_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_max_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
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
    __m512i min_index_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_index_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, data_f64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask, current_index_i64x8);
        current_index_i64x8 = _mm512_add_epi64(current_index_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        __m512d inf_f64x8 = _mm512_set1_pd(NK_F64_MAX);
        tail_f64x8 = _mm512_mask_mov_pd(inf_f64x8, tail_mask, tail_f64x8);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(tail_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, tail_f64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask, current_index_i64x8);
    }

    // Horizontal reduction
    nk_f64_t min_val = nk_reduce_min_f64x8_skylake_(min_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_index_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_f64_skylake_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Masked load with +inf for non-stride elements; track logical indices
    __mmask8 stride_mask_m8 = nk_stride_mask_b64x8_(stride_elements);
    __m512d pos_inf_f64x8 = _mm512_set1_pd(NK_F64_MAX);
    __m512d min_f64x8 = pos_inf_f64x8;
    __m512i min_index_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_index_i64x8 = nk_stride_logidx_i64x8_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b64x8_(stride_elements);
    __m512i step_i64x8 = _mm512_set1_epi64((nk_i64_t)elems_per_chunk);

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_mask_loadu_pd(pos_inf_f64x8, stride_mask_m8, data + idx_scalars);
        __mmask8 lt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask_m8, data_f64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask_m8, logical_index_i64x8);
        logical_index_i64x8 = _mm512_add_epi64(logical_index_i64x8, step_i64x8);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_mask_loadu_pd(pos_inf_f64x8, load_mask_m8, data + idx_scalars);
        __mmask8 lt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask_m8, data_f64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask_m8, logical_index_i64x8);
    }

    // Horizontal reduction
    nk_f64_t min_val = nk_reduce_min_f64x8_skylake_(min_f64x8);
    __mmask8 eq_mask_m8 = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m8);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_index_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_min_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *min_value = NK_F64_MAX, *min_index = count;
    else if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
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
    __m512i max_index_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_index_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, data_f64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask, current_index_i64x8);
        current_index_i64x8 = _mm512_add_epi64(current_index_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        __m512d neg_inf_f64x8 = _mm512_set1_pd(NK_F64_MIN);
        tail_f64x8 = _mm512_mask_mov_pd(neg_inf_f64x8, tail_mask, tail_f64x8);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(tail_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, tail_f64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask, current_index_i64x8);
    }

    // Horizontal reduction
    nk_f64_t max_val = nk_reduce_max_f64x8_skylake_(max_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_index_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_f64_skylake_strided_(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Masked load with -inf for non-stride elements; track logical indices
    __mmask8 stride_mask_m8 = nk_stride_mask_b64x8_(stride_elements);
    __m512d neg_inf_f64x8 = _mm512_set1_pd(NK_F64_MIN);
    __m512d max_f64x8 = neg_inf_f64x8;
    __m512i max_index_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_index_i64x8 = nk_stride_logidx_i64x8_(stride_elements);
    nk_size_t elems_per_chunk = nk_stride_elems_b64x8_(stride_elements);
    __m512i step_i64x8 = _mm512_set1_epi64((nk_i64_t)elems_per_chunk);

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_mask_loadu_pd(neg_inf_f64x8, stride_mask_m8, data + idx_scalars);
        __mmask8 gt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask_m8, data_f64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask_m8, logical_index_i64x8);
        logical_index_i64x8 = _mm512_add_epi64(logical_index_i64x8, step_i64x8);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_mask_loadu_pd(neg_inf_f64x8, load_mask_m8, data + idx_scalars);
        __mmask8 gt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask_m8, data_f64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask_m8, logical_index_i64x8);
    }

    // Horizontal reduction
    nk_f64_t max_val = nk_reduce_max_f64x8_skylake_(max_f64x8);
    __mmask8 eq_mask_m8 = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m8);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_index_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_max_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *max_value = NK_F64_MIN, *max_index = count;
    else if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8)
        nk_reduce_max_f64_skylake_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        nk_reduce_max_f64_skylake_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    // XOR with 0x80 to convert signed i8 to unsigned u8 for VPSADBW, then correct bias after loop.
    // Each i8 value v becomes (v ^ 0x80) = v + 128 (as unsigned), so SAD sums are biased by +128 per element.
    __m512i bias_i8x64 = _mm512_set1_epi8((char)0x80);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
    }
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8);
    sum -= (nk_i64_t)128 * (nk_i64_t)(idx); // correct for bias: each of `idx` elements was biased by +128
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    // VPSADBW sums 8 unsigned bytes → 1 u64 per 64-bit lane (8 lanes per ZMM).
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
    }
    nk_u64_t sum = nk_reduce_add_u64x8_skylake_(sum_u64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    // VPMADDWD with ones pairwise-sums adjacent i16 pairs into i32: result[i] = data[2i] + data[2i+1].
    // Accumulate in i32 (2 ops/iter: madd + add_epi32), flush to i64 every 32767 iterations.
    // VPMADDWD pair max is ±65536; i32 holds ±2.1B → safe for ~32K iterations per lane.
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    while (idx + 32 <= count) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        nk_size_t inner_end = idx + (nk_size_t)32 * 32767;
        if (inner_end > count) inner_end = count;
        for (; idx + 32 <= inner_end; idx += 32) {
            __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
            sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(data_i16x32, ones_i16x32));
        }
        // Flush i32 → i64
        __m256i low_i32x8 = _mm512_castsi512_si256(sum_i32x16);
        __m256i high_i32x8 = _mm512_extracti64x4_epi64(sum_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(low_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(high_i32x8));
    }
    nk_i64_t sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    // VPMADDWD treats operands as signed i16. XOR with 0x8000 converts each u16 v to signed
    // (v - 32768), making the bias exactly +32768 per element (predictable, independent of data).
    // Accumulate in i32 (3 ops/iter: xor + madd + add_epi32), flush to i64 every 32767 iterations.
    __m512i bias_u16x32 = _mm512_set1_epi16((short)0x8000);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx = 0;
    while (idx + 32 <= count) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        nk_size_t inner_end = idx + (nk_size_t)32 * 32767;
        if (inner_end > count) inner_end = count;
        for (; idx + 32 <= inner_end; idx += 32) {
            __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
            __m512i biased_i16x32 = _mm512_xor_si512(data_u16x32, bias_u16x32);
            sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(biased_i16x32, ones_i16x32));
        }
        // Flush i32 → i64
        __m256i low_i32x8 = _mm512_castsi512_si256(sum_i32x16);
        __m256i high_i32x8 = _mm512_extracti64x4_epi64(sum_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(low_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(high_i32x8));
    }
    nk_i64_t signed_sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_u64_t sum = (nk_u64_t)(signed_sum + (nk_i64_t)32768 * (nk_i64_t)idx);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i8_skylake_strided_(                  //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    // Masked load zeros out non-stride elements; zeros don't affect SAD results.
    // XOR with masked bias converts signed → unsigned for VPSADBW only at stride positions.
    // Pre-masked bias: non-stride positions are 0, so 0 XOR 0 = 0 (no extra maskz_mov needed).
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i masked_bias_i8x64 = _mm512_maskz_mov_epi8(stride_mask_m64, _mm512_set1_epi8((char)0x80));
    __m512i zero_i8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t elements_per_vector = 64 / stride_elements; // stride-selected elements per ZMM load
    nk_size_t vector_element_count = 0;
    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_i8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        __m512i unsigned_i8x64 = _mm512_xor_si512(data_i8x64, masked_bias_i8x64);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(unsigned_i8x64, zero_i8x64));
        vector_element_count += elements_per_vector;
    }
    nk_i64_t sum = (nk_i64_t)nk_reduce_add_u64x8_skylake_(sum_u64x8);
    sum -= (nk_i64_t)128 * (nk_i64_t)vector_element_count;
    // Scalar tail for remaining elements
    nk_i8_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u8_skylake_strided_(                  //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    // Masked load zeros non-stride elements; zeros don't affect SAD results.
    __mmask64 stride_mask_m64 = nk_stride_mask_u1x64_(stride_elements);
    __m512i zero_u8x64 = _mm512_setzero_si512();
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 64 <= total_scalars; idx_scalars += 64) {
        __m512i data_u8x64 = _mm512_maskz_loadu_epi8(stride_mask_m64, data + idx_scalars);
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(data_u8x64, zero_u8x64));
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
    // VPMADDWD with ones pairwise-sums adjacent i16 pairs into i32. Masked load zeros non-stride
    // elements, so VPMADDWD pair sums (zero + value) or (zero + zero) remain correct.
    // Accumulate in i32 (2 ops/iter: madd + add_epi32), flush to i64 every 32767 iterations.
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    while (idx_scalars + 32 <= total_scalars) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        nk_size_t inner_end = idx_scalars + (nk_size_t)32 * 32767;
        if (inner_end > total_scalars) inner_end = total_scalars;
        for (; idx_scalars + 32 <= inner_end; idx_scalars += 32) {
            __m512i data_i16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
            sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(data_i16x32, ones_i16x32));
        }
        // Flush i32 → i64
        __m256i low_i32x8 = _mm512_castsi512_si256(sum_i32x16);
        __m256i high_i32x8 = _mm512_extracti64x4_epi64(sum_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(low_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(high_i32x8));
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
    // VPMADDWD treats operands as signed. XOR with 0x8000 at stride positions only converts u16 to
    // signed (v - 32768), making bias exactly +32768 per stride-selected element.
    // Pre-masked bias: non-stride positions are 0, so 0 XOR 0 = 0 (zeros stay zero for VPMADDWD).
    // Accumulate in i32 (3 ops/iter: xor + madd + add_epi32), flush to i64 every 32767 iterations.
    __mmask32 stride_mask_m32 = nk_stride_mask_b16x32_(stride_elements);
    __m512i masked_bias_u16x32 = _mm512_maskz_mov_epi16(stride_mask_m32, _mm512_set1_epi16((short)0x8000));
    __m512i ones_i16x32 = _mm512_set1_epi16(1);
    __m512i sum_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    nk_size_t elements_per_vector = 32 / stride_elements;
    nk_size_t vector_element_count = 0;
    while (idx_scalars + 32 <= total_scalars) {
        __m512i sum_i32x16 = _mm512_setzero_si512();
        nk_size_t inner_end = idx_scalars + (nk_size_t)32 * 32767;
        if (inner_end > total_scalars) inner_end = total_scalars;
        for (; idx_scalars + 32 <= inner_end; idx_scalars += 32) {
            __m512i data_u16x32 = _mm512_maskz_loadu_epi16(stride_mask_m32, data + idx_scalars);
            __m512i biased_i16x32 = _mm512_xor_si512(data_u16x32, masked_bias_u16x32);
            sum_i32x16 = _mm512_add_epi32(sum_i32x16, _mm512_madd_epi16(biased_i16x32, ones_i16x32));
            vector_element_count += elements_per_vector;
        }
        // Flush i32 → i64
        __m256i low_i32x8 = _mm512_castsi512_si256(sum_i32x16);
        __m256i high_i32x8 = _mm512_extracti64x4_epi64(sum_i32x16, 1);
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(low_i32x8));
        sum_i64x8 = _mm512_add_epi64(sum_i64x8, _mm512_cvtepi32_epi64(high_i32x8));
    }
    nk_i64_t signed_sum = nk_reduce_add_i64x8_skylake_(sum_i64x8);
    nk_u64_t sum = (nk_u64_t)(signed_sum + (nk_i64_t)32768 * (nk_i64_t)vector_element_count);
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
    // Single-pass: track value in i8x64, indices in 4 × i32x16 (quarters of the 64 lanes)
    __m512i min_i8x64 = _mm512_loadu_si512(data);
    __m512i min_index_quarter0_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i min_index_quarter1_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                          31);
    __m512i min_index_quarter2_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                          47);
    __m512i min_index_quarter3_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i current_index_quarter0_i32x16 = _mm512_setr_epi32(64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                              78, 79);
    __m512i current_index_quarter1_i32x16 = _mm512_setr_epi32(80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                                                              94, 95);
    __m512i current_index_quarter2_i32x16 = _mm512_setr_epi32(96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                                                              108, 109, 110, 111);
    __m512i current_index_quarter3_i32x16 = _mm512_setr_epi32(112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                                                              123, 124, 125, 126, 127);
    __m512i step_i32x16 = _mm512_set1_epi32(64);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __mmask64 lt_mask = _mm512_cmp_epi8_mask(data_i8x64, min_i8x64, _MM_CMPINT_LT);
        min_i8x64 = _mm512_mask_mov_epi8(min_i8x64, lt_mask, data_i8x64);
        __mmask16 quarter0_mask = (__mmask16)(lt_mask);
        __mmask16 quarter1_mask = (__mmask16)(lt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(lt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(lt_mask >> 48);
        min_index_quarter0_i32x16 = _mm512_mask_mov_epi32(min_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        min_index_quarter1_i32x16 = _mm512_mask_mov_epi32(min_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        min_index_quarter2_i32x16 = _mm512_mask_mov_epi32(min_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        min_index_quarter3_i32x16 = _mm512_mask_mov_epi32(min_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
        current_index_quarter0_i32x16 = _mm512_add_epi32(current_index_quarter0_i32x16, step_i32x16);
        current_index_quarter1_i32x16 = _mm512_add_epi32(current_index_quarter1_i32x16, step_i32x16);
        current_index_quarter2_i32x16 = _mm512_add_epi32(current_index_quarter2_i32x16, step_i32x16);
        current_index_quarter3_i32x16 = _mm512_add_epi32(current_index_quarter3_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i sentinel_i8x64 = _mm512_set1_epi8(NK_I8_MAX);
        __m512i tail_i8x64 = _mm512_mask_loadu_epi8(sentinel_i8x64, tail_mask, data + idx);
        __mmask64 lt_mask = _mm512_cmp_epi8_mask(tail_i8x64, min_i8x64, _MM_CMPINT_LT);
        min_i8x64 = _mm512_mask_mov_epi8(min_i8x64, lt_mask, tail_i8x64);
        __mmask16 quarter0_mask = (__mmask16)(lt_mask);
        __mmask16 quarter1_mask = (__mmask16)(lt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(lt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(lt_mask >> 48);
        min_index_quarter0_i32x16 = _mm512_mask_mov_epi32(min_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        min_index_quarter1_i32x16 = _mm512_mask_mov_epi32(min_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        min_index_quarter2_i32x16 = _mm512_mask_mov_epi32(min_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        min_index_quarter3_i32x16 = _mm512_mask_mov_epi32(min_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
    }

    // Horizontal reduction: find scalar min, then locate first matching lane
    nk_i8_t min_val = nk_reduce_min_i8x64_skylake_(min_i8x64);
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(min_i8x64, _mm512_set1_epi8(min_val));
    unsigned int first_lane = (unsigned int)_tzcnt_u64(eq_mask);

    // Extract index from the correct quarter
    nk_i32_t indices[16];
    unsigned int quarter = first_lane / 16;
    unsigned int lane_in_quarter = first_lane % 16;
    switch (quarter) {
    case 0: _mm512_storeu_si512(indices, min_index_quarter0_i32x16); break;
    case 1: _mm512_storeu_si512(indices, min_index_quarter1_i32x16); break;
    case 2: _mm512_storeu_si512(indices, min_index_quarter2_i32x16); break;
    default: _mm512_storeu_si512(indices, min_index_quarter3_i32x16); break;
    }
    *min_value = min_val;
    *min_index = (nk_size_t)indices[lane_in_quarter];
}

NK_INTERNAL void nk_reduce_max_i8_skylake_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *max_value, nk_size_t *max_index) {
    // Single-pass: track value in i8x64, indices in 4 × i32x16 (quarters of the 64 lanes)
    __m512i max_i8x64 = _mm512_loadu_si512(data);
    __m512i max_index_quarter0_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i max_index_quarter1_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                          31);
    __m512i max_index_quarter2_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                          47);
    __m512i max_index_quarter3_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i current_index_quarter0_i32x16 = _mm512_setr_epi32(64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                              78, 79);
    __m512i current_index_quarter1_i32x16 = _mm512_setr_epi32(80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                                                              94, 95);
    __m512i current_index_quarter2_i32x16 = _mm512_setr_epi32(96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                                                              108, 109, 110, 111);
    __m512i current_index_quarter3_i32x16 = _mm512_setr_epi32(112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                                                              123, 124, 125, 126, 127);
    __m512i step_i32x16 = _mm512_set1_epi32(64);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_i8x64 = _mm512_loadu_si512(data + idx);
        __mmask64 gt_mask = _mm512_cmp_epi8_mask(data_i8x64, max_i8x64, _MM_CMPINT_NLE);
        max_i8x64 = _mm512_mask_mov_epi8(max_i8x64, gt_mask, data_i8x64);
        __mmask16 quarter0_mask = (__mmask16)(gt_mask);
        __mmask16 quarter1_mask = (__mmask16)(gt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(gt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(gt_mask >> 48);
        max_index_quarter0_i32x16 = _mm512_mask_mov_epi32(max_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        max_index_quarter1_i32x16 = _mm512_mask_mov_epi32(max_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        max_index_quarter2_i32x16 = _mm512_mask_mov_epi32(max_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        max_index_quarter3_i32x16 = _mm512_mask_mov_epi32(max_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
        current_index_quarter0_i32x16 = _mm512_add_epi32(current_index_quarter0_i32x16, step_i32x16);
        current_index_quarter1_i32x16 = _mm512_add_epi32(current_index_quarter1_i32x16, step_i32x16);
        current_index_quarter2_i32x16 = _mm512_add_epi32(current_index_quarter2_i32x16, step_i32x16);
        current_index_quarter3_i32x16 = _mm512_add_epi32(current_index_quarter3_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i sentinel_i8x64 = _mm512_set1_epi8(NK_I8_MIN);
        __m512i tail_i8x64 = _mm512_mask_loadu_epi8(sentinel_i8x64, tail_mask, data + idx);
        __mmask64 gt_mask = _mm512_cmp_epi8_mask(tail_i8x64, max_i8x64, _MM_CMPINT_NLE);
        max_i8x64 = _mm512_mask_mov_epi8(max_i8x64, gt_mask, tail_i8x64);
        __mmask16 quarter0_mask = (__mmask16)(gt_mask);
        __mmask16 quarter1_mask = (__mmask16)(gt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(gt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(gt_mask >> 48);
        max_index_quarter0_i32x16 = _mm512_mask_mov_epi32(max_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        max_index_quarter1_i32x16 = _mm512_mask_mov_epi32(max_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        max_index_quarter2_i32x16 = _mm512_mask_mov_epi32(max_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        max_index_quarter3_i32x16 = _mm512_mask_mov_epi32(max_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
    }

    // Horizontal reduction: find scalar max, then locate first matching lane
    nk_i8_t max_val = nk_reduce_max_i8x64_skylake_(max_i8x64);
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(max_i8x64, _mm512_set1_epi8(max_val));
    unsigned int first_lane = (unsigned int)_tzcnt_u64(eq_mask);

    // Extract index from the correct quarter
    nk_i32_t indices[16];
    unsigned int quarter = first_lane / 16;
    unsigned int lane_in_quarter = first_lane % 16;
    switch (quarter) {
    case 0: _mm512_storeu_si512(indices, max_index_quarter0_i32x16); break;
    case 1: _mm512_storeu_si512(indices, max_index_quarter1_i32x16); break;
    case 2: _mm512_storeu_si512(indices, max_index_quarter2_i32x16); break;
    default: _mm512_storeu_si512(indices, max_index_quarter3_i32x16); break;
    }
    *max_value = max_val;
    *max_index = (nk_size_t)indices[lane_in_quarter];
}

NK_INTERNAL void nk_reduce_min_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *min_value, nk_size_t *min_index) {
    // Single-pass: track value in u8x64, indices in 4 × i32x16 (quarters of the 64 lanes)
    __m512i min_u8x64 = _mm512_loadu_si512(data);
    __m512i min_index_quarter0_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i min_index_quarter1_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                          31);
    __m512i min_index_quarter2_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                          47);
    __m512i min_index_quarter3_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i current_index_quarter0_i32x16 = _mm512_setr_epi32(64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                              78, 79);
    __m512i current_index_quarter1_i32x16 = _mm512_setr_epi32(80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                                                              94, 95);
    __m512i current_index_quarter2_i32x16 = _mm512_setr_epi32(96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                                                              108, 109, 110, 111);
    __m512i current_index_quarter3_i32x16 = _mm512_setr_epi32(112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                                                              123, 124, 125, 126, 127);
    __m512i step_i32x16 = _mm512_set1_epi32(64);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        __mmask64 lt_mask = _mm512_cmp_epu8_mask(data_u8x64, min_u8x64, _MM_CMPINT_LT);
        min_u8x64 = _mm512_mask_mov_epi8(min_u8x64, lt_mask, data_u8x64);
        __mmask16 quarter0_mask = (__mmask16)(lt_mask);
        __mmask16 quarter1_mask = (__mmask16)(lt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(lt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(lt_mask >> 48);
        min_index_quarter0_i32x16 = _mm512_mask_mov_epi32(min_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        min_index_quarter1_i32x16 = _mm512_mask_mov_epi32(min_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        min_index_quarter2_i32x16 = _mm512_mask_mov_epi32(min_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        min_index_quarter3_i32x16 = _mm512_mask_mov_epi32(min_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
        current_index_quarter0_i32x16 = _mm512_add_epi32(current_index_quarter0_i32x16, step_i32x16);
        current_index_quarter1_i32x16 = _mm512_add_epi32(current_index_quarter1_i32x16, step_i32x16);
        current_index_quarter2_i32x16 = _mm512_add_epi32(current_index_quarter2_i32x16, step_i32x16);
        current_index_quarter3_i32x16 = _mm512_add_epi32(current_index_quarter3_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i sentinel_u8x64 = _mm512_set1_epi8((char)NK_U8_MAX);
        __m512i tail_u8x64 = _mm512_mask_loadu_epi8(sentinel_u8x64, tail_mask, data + idx);
        __mmask64 lt_mask = _mm512_cmp_epu8_mask(tail_u8x64, min_u8x64, _MM_CMPINT_LT);
        min_u8x64 = _mm512_mask_mov_epi8(min_u8x64, lt_mask, tail_u8x64);
        __mmask16 quarter0_mask = (__mmask16)(lt_mask);
        __mmask16 quarter1_mask = (__mmask16)(lt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(lt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(lt_mask >> 48);
        min_index_quarter0_i32x16 = _mm512_mask_mov_epi32(min_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        min_index_quarter1_i32x16 = _mm512_mask_mov_epi32(min_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        min_index_quarter2_i32x16 = _mm512_mask_mov_epi32(min_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        min_index_quarter3_i32x16 = _mm512_mask_mov_epi32(min_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
    }

    // Horizontal reduction: find scalar min, then locate first matching lane
    nk_u8_t min_val = nk_reduce_min_u8x64_skylake_(min_u8x64);
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(min_u8x64, _mm512_set1_epi8((char)min_val));
    unsigned int first_lane = (unsigned int)_tzcnt_u64(eq_mask);

    // Extract index from the correct quarter
    nk_i32_t indices[16];
    unsigned int quarter = first_lane / 16;
    unsigned int lane_in_quarter = first_lane % 16;
    switch (quarter) {
    case 0: _mm512_storeu_si512(indices, min_index_quarter0_i32x16); break;
    case 1: _mm512_storeu_si512(indices, min_index_quarter1_i32x16); break;
    case 2: _mm512_storeu_si512(indices, min_index_quarter2_i32x16); break;
    default: _mm512_storeu_si512(indices, min_index_quarter3_i32x16); break;
    }
    *min_value = min_val;
    *min_index = (nk_size_t)indices[lane_in_quarter];
}

NK_INTERNAL void nk_reduce_max_u8_skylake_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *max_value, nk_size_t *max_index) {
    // Single-pass: track value in u8x64, indices in 4 × i32x16 (quarters of the 64 lanes)
    __m512i max_u8x64 = _mm512_loadu_si512(data);
    __m512i max_index_quarter0_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i max_index_quarter1_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                          31);
    __m512i max_index_quarter2_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                          47);
    __m512i max_index_quarter3_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i current_index_quarter0_i32x16 = _mm512_setr_epi32(64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                                              78, 79);
    __m512i current_index_quarter1_i32x16 = _mm512_setr_epi32(80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                                                              94, 95);
    __m512i current_index_quarter2_i32x16 = _mm512_setr_epi32(96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                                                              108, 109, 110, 111);
    __m512i current_index_quarter3_i32x16 = _mm512_setr_epi32(112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
                                                              123, 124, 125, 126, 127);
    __m512i step_i32x16 = _mm512_set1_epi32(64);

    nk_size_t idx = 64;
    for (; idx + 64 <= count; idx += 64) {
        __m512i data_u8x64 = _mm512_loadu_si512(data + idx);
        __mmask64 gt_mask = _mm512_cmp_epu8_mask(data_u8x64, max_u8x64, _MM_CMPINT_NLE);
        max_u8x64 = _mm512_mask_mov_epi8(max_u8x64, gt_mask, data_u8x64);
        __mmask16 quarter0_mask = (__mmask16)(gt_mask);
        __mmask16 quarter1_mask = (__mmask16)(gt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(gt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(gt_mask >> 48);
        max_index_quarter0_i32x16 = _mm512_mask_mov_epi32(max_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        max_index_quarter1_i32x16 = _mm512_mask_mov_epi32(max_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        max_index_quarter2_i32x16 = _mm512_mask_mov_epi32(max_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        max_index_quarter3_i32x16 = _mm512_mask_mov_epi32(max_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
        current_index_quarter0_i32x16 = _mm512_add_epi32(current_index_quarter0_i32x16, step_i32x16);
        current_index_quarter1_i32x16 = _mm512_add_epi32(current_index_quarter1_i32x16, step_i32x16);
        current_index_quarter2_i32x16 = _mm512_add_epi32(current_index_quarter2_i32x16, step_i32x16);
        current_index_quarter3_i32x16 = _mm512_add_epi32(current_index_quarter3_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFull, (unsigned int)remaining);
        __m512i sentinel_u8x64 = _mm512_setzero_si512();
        __m512i tail_u8x64 = _mm512_mask_loadu_epi8(sentinel_u8x64, tail_mask, data + idx);
        __mmask64 gt_mask = _mm512_cmp_epu8_mask(tail_u8x64, max_u8x64, _MM_CMPINT_NLE);
        max_u8x64 = _mm512_mask_mov_epi8(max_u8x64, gt_mask, tail_u8x64);
        __mmask16 quarter0_mask = (__mmask16)(gt_mask);
        __mmask16 quarter1_mask = (__mmask16)(gt_mask >> 16);
        __mmask16 quarter2_mask = (__mmask16)(gt_mask >> 32);
        __mmask16 quarter3_mask = (__mmask16)(gt_mask >> 48);
        max_index_quarter0_i32x16 = _mm512_mask_mov_epi32(max_index_quarter0_i32x16, quarter0_mask,
                                                          current_index_quarter0_i32x16);
        max_index_quarter1_i32x16 = _mm512_mask_mov_epi32(max_index_quarter1_i32x16, quarter1_mask,
                                                          current_index_quarter1_i32x16);
        max_index_quarter2_i32x16 = _mm512_mask_mov_epi32(max_index_quarter2_i32x16, quarter2_mask,
                                                          current_index_quarter2_i32x16);
        max_index_quarter3_i32x16 = _mm512_mask_mov_epi32(max_index_quarter3_i32x16, quarter3_mask,
                                                          current_index_quarter3_i32x16);
    }

    // Horizontal reduction: find scalar max, then locate first matching lane
    nk_u8_t max_val = nk_reduce_max_u8x64_skylake_(max_u8x64);
    __mmask64 eq_mask = _mm512_cmpeq_epi8_mask(max_u8x64, _mm512_set1_epi8((char)max_val));
    unsigned int first_lane = (unsigned int)_tzcnt_u64(eq_mask);

    // Extract index from the correct quarter
    nk_i32_t indices[16];
    unsigned int quarter = first_lane / 16;
    unsigned int lane_in_quarter = first_lane % 16;
    switch (quarter) {
    case 0: _mm512_storeu_si512(indices, max_index_quarter0_i32x16); break;
    case 1: _mm512_storeu_si512(indices, max_index_quarter1_i32x16); break;
    case 2: _mm512_storeu_si512(indices, max_index_quarter2_i32x16); break;
    default: _mm512_storeu_si512(indices, max_index_quarter3_i32x16); break;
    }
    *max_value = max_val;
    *max_index = (nk_size_t)indices[lane_in_quarter];
}

NK_INTERNAL void nk_reduce_min_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *min_value, nk_size_t *min_index) {
    // Single-pass: track value in i16x32, indices in 2 × i32x16 (low/high halves of the 32 lanes)
    __m512i min_i16x32 = _mm512_loadu_si512(data);
    __m512i min_index_low_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i min_index_high_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i current_index_low_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                         47);
    __m512i current_index_high_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i step_i32x16 = _mm512_set1_epi32(32);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        __mmask32 lt_mask = _mm512_cmp_epi16_mask(data_i16x32, min_i16x32, _MM_CMPINT_LT);
        min_i16x32 = _mm512_mask_mov_epi16(min_i16x32, lt_mask, data_i16x32);
        __mmask16 low_mask = (__mmask16)(lt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(lt_mask >> 16);
        min_index_low_i32x16 = _mm512_mask_mov_epi32(min_index_low_i32x16, low_mask, current_index_low_i32x16);
        min_index_high_i32x16 = _mm512_mask_mov_epi32(min_index_high_i32x16, high_mask, current_index_high_i32x16);
        current_index_low_i32x16 = _mm512_add_epi32(current_index_low_i32x16, step_i32x16);
        current_index_high_i32x16 = _mm512_add_epi32(current_index_high_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i sentinel_i16x32 = _mm512_set1_epi16(NK_I16_MAX);
        __m512i tail_i16x32 = _mm512_mask_loadu_epi16(sentinel_i16x32, tail_mask, data + idx);
        __mmask32 lt_mask = _mm512_cmp_epi16_mask(tail_i16x32, min_i16x32, _MM_CMPINT_LT);
        min_i16x32 = _mm512_mask_mov_epi16(min_i16x32, lt_mask, tail_i16x32);
        __mmask16 low_mask = (__mmask16)(lt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(lt_mask >> 16);
        min_index_low_i32x16 = _mm512_mask_mov_epi32(min_index_low_i32x16, low_mask, current_index_low_i32x16);
        min_index_high_i32x16 = _mm512_mask_mov_epi32(min_index_high_i32x16, high_mask, current_index_high_i32x16);
    }

    // Horizontal reduction: find scalar min, then locate first matching lane
    nk_i16_t min_val = nk_reduce_min_i16x32_skylake_(min_i16x32);
    __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(min_i16x32, _mm512_set1_epi16(min_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract index from the correct half
    nk_i32_t indices[16];
    if (first_lane < 16) {
        _mm512_storeu_si512(indices, min_index_low_i32x16);
        *min_index = (nk_size_t)indices[first_lane];
    }
    else {
        _mm512_storeu_si512(indices, min_index_high_i32x16);
        *min_index = (nk_size_t)indices[first_lane - 16];
    }
    *min_value = min_val;
}

NK_INTERNAL void nk_reduce_max_i16_skylake_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *max_value, nk_size_t *max_index) {
    // Single-pass: track value in i16x32, indices in 2 × i32x16 (low/high halves of the 32 lanes)
    __m512i max_i16x32 = _mm512_loadu_si512(data);
    __m512i max_index_low_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i max_index_high_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i current_index_low_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                         47);
    __m512i current_index_high_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i step_i32x16 = _mm512_set1_epi32(32);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_i16x32 = _mm512_loadu_si512(data + idx);
        __mmask32 gt_mask = _mm512_cmp_epi16_mask(data_i16x32, max_i16x32, _MM_CMPINT_NLE);
        max_i16x32 = _mm512_mask_mov_epi16(max_i16x32, gt_mask, data_i16x32);
        __mmask16 low_mask = (__mmask16)(gt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(gt_mask >> 16);
        max_index_low_i32x16 = _mm512_mask_mov_epi32(max_index_low_i32x16, low_mask, current_index_low_i32x16);
        max_index_high_i32x16 = _mm512_mask_mov_epi32(max_index_high_i32x16, high_mask, current_index_high_i32x16);
        current_index_low_i32x16 = _mm512_add_epi32(current_index_low_i32x16, step_i32x16);
        current_index_high_i32x16 = _mm512_add_epi32(current_index_high_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i sentinel_i16x32 = _mm512_set1_epi16(NK_I16_MIN);
        __m512i tail_i16x32 = _mm512_mask_loadu_epi16(sentinel_i16x32, tail_mask, data + idx);
        __mmask32 gt_mask = _mm512_cmp_epi16_mask(tail_i16x32, max_i16x32, _MM_CMPINT_NLE);
        max_i16x32 = _mm512_mask_mov_epi16(max_i16x32, gt_mask, tail_i16x32);
        __mmask16 low_mask = (__mmask16)(gt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(gt_mask >> 16);
        max_index_low_i32x16 = _mm512_mask_mov_epi32(max_index_low_i32x16, low_mask, current_index_low_i32x16);
        max_index_high_i32x16 = _mm512_mask_mov_epi32(max_index_high_i32x16, high_mask, current_index_high_i32x16);
    }

    // Horizontal reduction: find scalar max, then locate first matching lane
    nk_i16_t max_val = nk_reduce_max_i16x32_skylake_(max_i16x32);
    __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(max_i16x32, _mm512_set1_epi16(max_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract index from the correct half
    nk_i32_t indices[16];
    if (first_lane < 16) {
        _mm512_storeu_si512(indices, max_index_low_i32x16);
        *max_index = (nk_size_t)indices[first_lane];
    }
    else {
        _mm512_storeu_si512(indices, max_index_high_i32x16);
        *max_index = (nk_size_t)indices[first_lane - 16];
    }
    *max_value = max_val;
}

NK_INTERNAL void nk_reduce_min_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *min_value, nk_size_t *min_index) {
    // Single-pass: track value in u16x32, indices in 2 × i32x16 (low/high halves of the 32 lanes)
    __m512i min_u16x32 = _mm512_loadu_si512(data);
    __m512i min_index_low_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i min_index_high_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i current_index_low_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                         47);
    __m512i current_index_high_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i step_i32x16 = _mm512_set1_epi32(32);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        __mmask32 lt_mask = _mm512_cmp_epu16_mask(data_u16x32, min_u16x32, _MM_CMPINT_LT);
        min_u16x32 = _mm512_mask_mov_epi16(min_u16x32, lt_mask, data_u16x32);
        __mmask16 low_mask = (__mmask16)(lt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(lt_mask >> 16);
        min_index_low_i32x16 = _mm512_mask_mov_epi32(min_index_low_i32x16, low_mask, current_index_low_i32x16);
        min_index_high_i32x16 = _mm512_mask_mov_epi32(min_index_high_i32x16, high_mask, current_index_high_i32x16);
        current_index_low_i32x16 = _mm512_add_epi32(current_index_low_i32x16, step_i32x16);
        current_index_high_i32x16 = _mm512_add_epi32(current_index_high_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i sentinel_u16x32 = _mm512_set1_epi16((short)NK_U16_MAX);
        __m512i tail_u16x32 = _mm512_mask_loadu_epi16(sentinel_u16x32, tail_mask, data + idx);
        __mmask32 lt_mask = _mm512_cmp_epu16_mask(tail_u16x32, min_u16x32, _MM_CMPINT_LT);
        min_u16x32 = _mm512_mask_mov_epi16(min_u16x32, lt_mask, tail_u16x32);
        __mmask16 low_mask = (__mmask16)(lt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(lt_mask >> 16);
        min_index_low_i32x16 = _mm512_mask_mov_epi32(min_index_low_i32x16, low_mask, current_index_low_i32x16);
        min_index_high_i32x16 = _mm512_mask_mov_epi32(min_index_high_i32x16, high_mask, current_index_high_i32x16);
    }

    // Horizontal reduction: find scalar min, then locate first matching lane
    nk_u16_t min_val = nk_reduce_min_u16x32_skylake_(min_u16x32);
    __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(min_u16x32, _mm512_set1_epi16((short)min_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract index from the correct half
    nk_i32_t indices[16];
    if (first_lane < 16) {
        _mm512_storeu_si512(indices, min_index_low_i32x16);
        *min_index = (nk_size_t)indices[first_lane];
    }
    else {
        _mm512_storeu_si512(indices, min_index_high_i32x16);
        *min_index = (nk_size_t)indices[first_lane - 16];
    }
    *min_value = min_val;
}

NK_INTERNAL void nk_reduce_max_u16_skylake_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *max_value, nk_size_t *max_index) {
    // Single-pass: track value in u16x32, indices in 2 × i32x16 (low/high halves of the 32 lanes)
    __m512i max_u16x32 = _mm512_loadu_si512(data);
    __m512i max_index_low_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i max_index_high_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i current_index_low_i32x16 = _mm512_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                         47);
    __m512i current_index_high_i32x16 = _mm512_setr_epi32(48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                                          63);
    __m512i step_i32x16 = _mm512_set1_epi32(32);

    nk_size_t idx = 32;
    for (; idx + 32 <= count; idx += 32) {
        __m512i data_u16x32 = _mm512_loadu_si512(data + idx);
        __mmask32 gt_mask = _mm512_cmp_epu16_mask(data_u16x32, max_u16x32, _MM_CMPINT_NLE);
        max_u16x32 = _mm512_mask_mov_epi16(max_u16x32, gt_mask, data_u16x32);
        __mmask16 low_mask = (__mmask16)(gt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(gt_mask >> 16);
        max_index_low_i32x16 = _mm512_mask_mov_epi32(max_index_low_i32x16, low_mask, current_index_low_i32x16);
        max_index_high_i32x16 = _mm512_mask_mov_epi32(max_index_high_i32x16, high_mask, current_index_high_i32x16);
        current_index_low_i32x16 = _mm512_add_epi32(current_index_low_i32x16, step_i32x16);
        current_index_high_i32x16 = _mm512_add_epi32(current_index_high_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask32 tail_mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)remaining);
        __m512i sentinel_u16x32 = _mm512_setzero_si512();
        __m512i tail_u16x32 = _mm512_mask_loadu_epi16(sentinel_u16x32, tail_mask, data + idx);
        __mmask32 gt_mask = _mm512_cmp_epu16_mask(tail_u16x32, max_u16x32, _MM_CMPINT_NLE);
        max_u16x32 = _mm512_mask_mov_epi16(max_u16x32, gt_mask, tail_u16x32);
        __mmask16 low_mask = (__mmask16)(gt_mask & 0xFFFF);
        __mmask16 high_mask = (__mmask16)(gt_mask >> 16);
        max_index_low_i32x16 = _mm512_mask_mov_epi32(max_index_low_i32x16, low_mask, current_index_low_i32x16);
        max_index_high_i32x16 = _mm512_mask_mov_epi32(max_index_high_i32x16, high_mask, current_index_high_i32x16);
    }

    // Horizontal reduction: find scalar max, then locate first matching lane
    nk_u16_t max_val = nk_reduce_max_u16x32_skylake_(max_u16x32);
    __mmask32 eq_mask = _mm512_cmpeq_epi16_mask(max_u16x32, _mm512_set1_epi16((short)max_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract index from the correct half
    nk_i32_t indices[16];
    if (first_lane < 16) {
        _mm512_storeu_si512(indices, max_index_low_i32x16);
        *max_index = (nk_size_t)indices[first_lane];
    }
    else {
        _mm512_storeu_si512(indices, max_index_high_i32x16);
        *max_index = (nk_size_t)indices[first_lane - 16];
    }
    *max_value = max_val;
}

NK_INTERNAL void nk_reduce_min_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512i min_i32x16 = _mm512_loadu_si512(data);
    __m512i min_index_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_index_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        __mmask16 lt_mask = _mm512_cmp_epi32_mask(data_i32x16, min_i32x16, _MM_CMPINT_LT);
        min_i32x16 = _mm512_mask_mov_epi32(min_i32x16, lt_mask, data_i32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask, current_index_i32x16);
        current_index_i32x16 = _mm512_add_epi32(current_index_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i sentinel_i32x16 = _mm512_set1_epi32(NK_I32_MAX);
        __m512i tail_i32x16 = _mm512_mask_loadu_epi32(sentinel_i32x16, tail_mask, data + idx);
        __mmask16 lt_mask = _mm512_cmp_epi32_mask(tail_i32x16, min_i32x16, _MM_CMPINT_LT);
        min_i32x16 = _mm512_mask_mov_epi32(min_i32x16, lt_mask, tail_i32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask, current_index_i32x16);
    }

    // Horizontal reduction: find scalar min, then first matching lane
    nk_i32_t min_val = nk_reduce_min_i32x16_skylake_(min_i32x16);
    __mmask16 eq_mask = _mm512_cmpeq_epi32_mask(min_i32x16, _mm512_set1_epi32(min_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_index_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_i32_skylake_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512i max_i32x16 = _mm512_loadu_si512(data);
    __m512i max_index_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_index_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_i32x16 = _mm512_loadu_si512(data + idx);
        __mmask16 gt_mask = _mm512_cmp_epi32_mask(data_i32x16, max_i32x16, _MM_CMPINT_NLE);
        max_i32x16 = _mm512_mask_mov_epi32(max_i32x16, gt_mask, data_i32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask, current_index_i32x16);
        current_index_i32x16 = _mm512_add_epi32(current_index_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i sentinel_i32x16 = _mm512_set1_epi32(NK_I32_MIN);
        __m512i tail_i32x16 = _mm512_mask_loadu_epi32(sentinel_i32x16, tail_mask, data + idx);
        __mmask16 gt_mask = _mm512_cmp_epi32_mask(tail_i32x16, max_i32x16, _MM_CMPINT_NLE);
        max_i32x16 = _mm512_mask_mov_epi32(max_i32x16, gt_mask, tail_i32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask, current_index_i32x16);
    }

    // Horizontal reduction: find scalar max, then first matching lane
    nk_i32_t max_val = nk_reduce_max_i32x16_skylake_(max_i32x16);
    __mmask16 eq_mask = _mm512_cmpeq_epi32_mask(max_i32x16, _mm512_set1_epi32(max_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_index_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512i min_u32x16 = _mm512_loadu_si512(data);
    __m512i min_index_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_index_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        __mmask16 lt_mask = _mm512_cmp_epu32_mask(data_u32x16, min_u32x16, _MM_CMPINT_LT);
        min_u32x16 = _mm512_mask_mov_epi32(min_u32x16, lt_mask, data_u32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask, current_index_i32x16);
        current_index_i32x16 = _mm512_add_epi32(current_index_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i sentinel_u32x16 = _mm512_set1_epi32((nk_i32_t)NK_U32_MAX);
        __m512i tail_u32x16 = _mm512_mask_loadu_epi32(sentinel_u32x16, tail_mask, data + idx);
        __mmask16 lt_mask = _mm512_cmp_epu32_mask(tail_u32x16, min_u32x16, _MM_CMPINT_LT);
        min_u32x16 = _mm512_mask_mov_epi32(min_u32x16, lt_mask, tail_u32x16);
        min_index_i32x16 = _mm512_mask_mov_epi32(min_index_i32x16, lt_mask, current_index_i32x16);
    }

    // Horizontal reduction: find scalar min, then first matching lane
    nk_u32_t min_val = nk_reduce_min_u32x16_skylake_(min_u32x16);
    __mmask16 eq_mask = _mm512_cmpeq_epi32_mask(min_u32x16, _mm512_set1_epi32((nk_i32_t)min_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_index_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_u32_skylake_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512i max_u32x16 = _mm512_loadu_si512(data);
    __m512i max_index_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_index_i32x16 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        __m512i data_u32x16 = _mm512_loadu_si512(data + idx);
        __mmask16 gt_mask = _mm512_cmp_epu32_mask(data_u32x16, max_u32x16, _MM_CMPINT_NLE);
        max_u32x16 = _mm512_mask_mov_epi32(max_u32x16, gt_mask, data_u32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask, current_index_i32x16);
        current_index_i32x16 = _mm512_add_epi32(current_index_i32x16, step_i32x16);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512i sentinel_u32x16 = _mm512_setzero_si512();
        __m512i tail_u32x16 = _mm512_mask_loadu_epi32(sentinel_u32x16, tail_mask, data + idx);
        __mmask16 gt_mask = _mm512_cmp_epu32_mask(tail_u32x16, max_u32x16, _MM_CMPINT_NLE);
        max_u32x16 = _mm512_mask_mov_epi32(max_u32x16, gt_mask, tail_u32x16);
        max_index_i32x16 = _mm512_mask_mov_epi32(max_index_i32x16, gt_mask, current_index_i32x16);
    }

    // Horizontal reduction: find scalar max, then first matching lane
    nk_u32_t max_val = nk_reduce_max_u32x16_skylake_(max_u32x16);
    __mmask16 eq_mask = _mm512_cmpeq_epi32_mask(max_u32x16, _mm512_set1_epi32((nk_i32_t)max_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_index_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512i min_i64x8 = _mm512_loadu_si512(data);
    __m512i min_index_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_index_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        __mmask8 lt_mask = _mm512_cmp_epi64_mask(data_i64x8, min_i64x8, _MM_CMPINT_LT);
        min_i64x8 = _mm512_mask_mov_epi64(min_i64x8, lt_mask, data_i64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask, current_index_i64x8);
        current_index_i64x8 = _mm512_add_epi64(current_index_i64x8, step_i64x8);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i sentinel_i64x8 = _mm512_set1_epi64(NK_I64_MAX);
        __m512i tail_i64x8 = _mm512_mask_loadu_epi64(sentinel_i64x8, tail_mask, data + idx);
        __mmask8 lt_mask = _mm512_cmp_epi64_mask(tail_i64x8, min_i64x8, _MM_CMPINT_LT);
        min_i64x8 = _mm512_mask_mov_epi64(min_i64x8, lt_mask, tail_i64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask, current_index_i64x8);
    }

    // Horizontal reduction: find scalar min, then first matching lane
    nk_i64_t min_val = nk_reduce_min_i64x8_skylake_(min_i64x8);
    __mmask8 eq_mask = _mm512_cmpeq_epi64_mask(min_i64x8, _mm512_set1_epi64(min_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_index_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_i64_skylake_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512i max_i64x8 = _mm512_loadu_si512(data);
    __m512i max_index_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_index_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_i64x8 = _mm512_loadu_si512(data + idx);
        __mmask8 gt_mask = _mm512_cmp_epi64_mask(data_i64x8, max_i64x8, _MM_CMPINT_NLE);
        max_i64x8 = _mm512_mask_mov_epi64(max_i64x8, gt_mask, data_i64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask, current_index_i64x8);
        current_index_i64x8 = _mm512_add_epi64(current_index_i64x8, step_i64x8);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i sentinel_i64x8 = _mm512_set1_epi64(NK_I64_MIN);
        __m512i tail_i64x8 = _mm512_mask_loadu_epi64(sentinel_i64x8, tail_mask, data + idx);
        __mmask8 gt_mask = _mm512_cmp_epi64_mask(tail_i64x8, max_i64x8, _MM_CMPINT_NLE);
        max_i64x8 = _mm512_mask_mov_epi64(max_i64x8, gt_mask, tail_i64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask, current_index_i64x8);
    }

    // Horizontal reduction: find scalar max, then first matching lane
    nk_i64_t max_val = nk_reduce_max_i64x8_skylake_(max_i64x8);
    __mmask8 eq_mask = _mm512_cmpeq_epi64_mask(max_i64x8, _mm512_set1_epi64(max_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_index_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_min_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512i min_u64x8 = _mm512_loadu_si512(data);
    __m512i min_index_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_index_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        __mmask8 lt_mask = _mm512_cmp_epu64_mask(data_u64x8, min_u64x8, _MM_CMPINT_LT);
        min_u64x8 = _mm512_mask_mov_epi64(min_u64x8, lt_mask, data_u64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask, current_index_i64x8);
        current_index_i64x8 = _mm512_add_epi64(current_index_i64x8, step_i64x8);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i sentinel_u64x8 = _mm512_set1_epi64((nk_i64_t)NK_U64_MAX);
        __m512i tail_u64x8 = _mm512_mask_loadu_epi64(sentinel_u64x8, tail_mask, data + idx);
        __mmask8 lt_mask = _mm512_cmp_epu64_mask(tail_u64x8, min_u64x8, _MM_CMPINT_LT);
        min_u64x8 = _mm512_mask_mov_epi64(min_u64x8, lt_mask, tail_u64x8);
        min_index_i64x8 = _mm512_mask_mov_epi64(min_index_i64x8, lt_mask, current_index_i64x8);
    }

    // Horizontal reduction: find scalar min, then first matching lane
    nk_u64_t min_val = nk_reduce_min_u64x8_skylake_(min_u64x8);
    __mmask8 eq_mask = _mm512_cmpeq_epi64_mask(min_u64x8, _mm512_set1_epi64((nk_i64_t)min_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_index_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void nk_reduce_max_u64_skylake_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512i max_u64x8 = _mm512_loadu_si512(data);
    __m512i max_index_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_index_i64x8 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        __m512i data_u64x8 = _mm512_loadu_si512(data + idx);
        __mmask8 gt_mask = _mm512_cmp_epu64_mask(data_u64x8, max_u64x8, _MM_CMPINT_NLE);
        max_u64x8 = _mm512_mask_mov_epi64(max_u64x8, gt_mask, data_u64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask, current_index_i64x8);
        current_index_i64x8 = _mm512_add_epi64(current_index_i64x8, step_i64x8);
    }

    // Handle tail with masked load + sentinel fill
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512i sentinel_u64x8 = _mm512_setzero_si512();
        __m512i tail_u64x8 = _mm512_mask_loadu_epi64(sentinel_u64x8, tail_mask, data + idx);
        __mmask8 gt_mask = _mm512_cmp_epu64_mask(tail_u64x8, max_u64x8, _MM_CMPINT_NLE);
        max_u64x8 = _mm512_mask_mov_epi64(max_u64x8, gt_mask, tail_u64x8);
        max_index_i64x8 = _mm512_mask_mov_epi64(max_index_i64x8, gt_mask, current_index_i64x8);
    }

    // Horizontal reduction: find scalar max, then first matching lane
    nk_u64_t max_val = nk_reduce_max_u64x8_skylake_(max_u64x8);
    __mmask8 eq_mask = _mm512_cmpeq_epi64_mask(max_u64x8, _mm512_set1_epi64((nk_i64_t)max_val));
    unsigned int first_lane = _tzcnt_u32(eq_mask);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_index_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
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
    if (count == 0) *min_value = NK_I8_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i8_t) && count >= 64)
        nk_reduce_min_i8_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i8_skylake(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I8_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i8_t) && count >= 64)
        nk_reduce_max_i8_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u8_skylake(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U8_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u8_t) && count >= 64)
        nk_reduce_min_u8_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u8_skylake(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U8_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u8_t) && count >= 64)
        nk_reduce_max_u8_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i16_skylake(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I16_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i16_t) && count >= 32)
        nk_reduce_min_i16_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i16_skylake(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I16_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i16_t) && count >= 32)
        nk_reduce_max_i16_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u16_skylake(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U16_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u16_t) && count >= 32)
        nk_reduce_min_u16_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u16_skylake(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U16_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u16_t) && count >= 32)
        nk_reduce_max_u16_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i32_skylake(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I32_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i32_t) && count >= 16)
        nk_reduce_min_i32_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i32_skylake(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I32_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i32_t) && count >= 16)
        nk_reduce_max_i32_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u32_skylake(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U32_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u32_t) && count >= 16)
        nk_reduce_min_u32_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u32_skylake(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U32_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u32_t) && count >= 16)
        nk_reduce_max_u32_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i64_skylake(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_I64_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_i64_t) && count >= 8)
        nk_reduce_min_i64_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i64_skylake(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_I64_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_i64_t) && count >= 8)
        nk_reduce_max_i64_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u64_skylake(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_U64_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_u64_t) && count >= 8)
        nk_reduce_min_u64_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u64_skylake(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_U64_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_u64_t) && count >= 8)
        nk_reduce_max_u64_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

    __m512i nan_cmp_u8x64 = _mm512_set1_epi8((char)0xFF); // E4M3 NaN in comparable form
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t min_vec, argmin_vec;
    min_vec.zmm = nan_cmp_u8x64;             // Identity for min (0xFF)
    argmin_vec.zmm = _mm512_setzero_si512(); // Track which iteration each qword's min came from
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;

nk_reduce_min_e4m3_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        data_i8x64 = _mm512_mask_loadu_epi8(nan_cmp_u8x64, tail_mask, data); // 0xFF in tail (min identity)
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
    __mmask64 min_mask_m64 = nk_min_mask_e4m3x64_skylake_(min_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
    __m512i new_min_cmp_u8x64 = _mm512_mask_mov_epi8(data_cmp_u8x64, min_mask_m64, min_vec.zmm);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_min_cmp_u8x64, min_vec.zmm);

    argmin_vec.zmm = _mm512_mask_mov_epi64(argmin_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    min_vec.zmm = new_min_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_min_e4m3_skylake_cycle_;

    // Horizontal reduction: find lane with minimum value, extract its iteration index
    nk_size_t first_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmin_vec.i64s[first_lane / 8];
    *min_index = chunk_idx * 64 + (first_lane % 64);

    // Convert min value back to FP8, then to F32
    min_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(min_vec.zmm);
    nk_e4m3_to_f32_serial(&min_vec.e4m3s[first_lane], min_value);
}

NK_INTERNAL void nk_reduce_max_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

    __m512i nan_cmp_u8x64 = _mm512_set1_epi8((char)0xFF); // E4M3 NaN in comparable form
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t max_vec, argmax_vec;
    max_vec.zmm = _mm512_setzero_si512();    // Identity for max (0x00)
    argmax_vec.zmm = _mm512_setzero_si512(); // Track which iteration each qword's max came from
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;

nk_reduce_max_e4m3_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        data_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, data); // zeros in tail (max identity)
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
    __mmask64 max_mask_m64 = nk_max_mask_e4m3x64_skylake_(max_vec.zmm, data_cmp_u8x64, nan_cmp_u8x64);
    __m512i new_max_cmp_u8x64 = _mm512_mask_mov_epi8(data_cmp_u8x64, max_mask_m64, max_vec.zmm);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_max_cmp_u8x64, max_vec.zmm);

    argmax_vec.zmm = _mm512_mask_mov_epi64(argmax_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    max_vec.zmm = new_max_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_max_e4m3_skylake_cycle_;

    // Horizontal reduction: find lane with maximum value, extract its iteration index
    nk_size_t first_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmax_vec.i64s[first_lane / 8];
    *max_index = chunk_idx * 64 + (first_lane % 64);

    // Convert max value back to FP8, then to F32
    max_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(max_vec.zmm);
    nk_e4m3_to_f32_serial(&max_vec.e4m3s[first_lane], max_value);
}

NK_INTERNAL void nk_reduce_min_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

    // E5M2 NaN threshold: 0x7D-0x7F map to 0xFD-0xFF in comparable form
    __m512i nan_threshold_cmp_u8x64 = _mm512_set1_epi8((char)0xFD);
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t min_vec, argmin_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF); // Identity for min
    argmin_vec.zmm = _mm512_setzero_si512();
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;

nk_reduce_min_e5m2_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        data_i8x64 = _mm512_mask_loadu_epi8(min_vec.zmm, tail_mask, data); // 0xFF in tail (min identity)
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
    __mmask64 min_mask_m64 = nk_min_mask_e5m2x64_skylake_(min_vec.zmm, data_cmp_u8x64, nan_threshold_cmp_u8x64);
    __m512i new_min_cmp_u8x64 = _mm512_mask_mov_epi8(data_cmp_u8x64, min_mask_m64, min_vec.zmm);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_min_cmp_u8x64, min_vec.zmm);

    argmin_vec.zmm = _mm512_mask_mov_epi64(argmin_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    min_vec.zmm = new_min_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_min_e5m2_skylake_cycle_;

    // Horizontal reduction: find lane with minimum value, extract its iteration index
    nk_size_t first_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmin_vec.i64s[first_lane / 8];
    *min_index = chunk_idx * 64 + (first_lane % 64);

    // Convert min value back to FP8, then to F32
    min_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(min_vec.zmm);
    nk_e5m2_to_f32_serial(&min_vec.e5m2s[first_lane], min_value);
}

NK_INTERNAL void nk_reduce_max_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

    __m512i nan_threshold_cmp_u8x64 = _mm512_set1_epi8((char)0xFD);
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t max_vec, argmax_vec;
    max_vec.zmm = _mm512_setzero_si512(); // Identity for max (0x00)
    argmax_vec.zmm = _mm512_setzero_si512();
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;

nk_reduce_max_e5m2_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        data_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, data); // zeros in tail (max identity)
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp8x64_to_u8x64_comparable_skylake_(data_i8x64);
    __mmask64 max_mask_m64 = nk_max_mask_e5m2x64_skylake_(max_vec.zmm, data_cmp_u8x64, nan_threshold_cmp_u8x64);
    __m512i new_max_cmp_u8x64 = _mm512_mask_mov_epi8(data_cmp_u8x64, max_mask_m64, max_vec.zmm);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_max_cmp_u8x64, max_vec.zmm);

    argmax_vec.zmm = _mm512_mask_mov_epi64(argmax_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    max_vec.zmm = new_max_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_max_e5m2_skylake_cycle_;

    // Horizontal reduction: find lane with maximum value, extract its iteration index
    nk_size_t first_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmax_vec.i64s[first_lane / 8];
    *max_index = chunk_idx * 64 + (first_lane % 64);

    // Convert max value back to FP8, then to F32
    max_vec.zmm = nk_u8x64_comparable_to_fp8x64_skylake_(max_vec.zmm);
    nk_e5m2_to_f32_serial(&max_vec.e5m2s[first_lane], max_value);
}

NK_PUBLIC void nk_reduce_min_e4m3_skylake(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_e4m3_t) && count >= 64)
        nk_reduce_min_e4m3_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e4m3_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_e4m3_skylake(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_e4m3_t) && count >= 64)
        nk_reduce_max_e4m3_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e4m3_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_e5m2_skylake(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = NK_F32_MAX, *min_index = count;
    else if (stride_bytes == sizeof(nk_e5m2_t) && count >= 64)
        nk_reduce_min_e5m2_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e5m2_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_e5m2_skylake(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = NK_F32_MIN, *max_index = count;
    else if (stride_bytes == sizeof(nk_e5m2_t) && count >= 64)
        nk_reduce_max_e5m2_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e5m2_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_e4m3_skylake_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e4m3x16_to_f32x16_skylake_(data + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
    }
    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_INTERNAL void nk_reduce_add_e4m3_skylake_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    // Masked load zeros non-stride positions; e4m3 zero (0x00) converts to 0.0f, no sum impact.
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m128i data_e4m3x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data + idx_scalars);
        __m512 data_f32x16 = nk_e4m3x16_to_f32x16_skylake_(data_e4m3x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
    }
    // Tail: combine stride mask with byte-count mask for remaining bytes
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e4m3x16 = _mm_maskz_loadu_epi8(tail_mask, data + idx_scalars);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, nk_e4m3x16_to_f32x16_skylake_(data_e4m3x16));
    }
    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_reduce_add_e4m3_skylake(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    if (stride_bytes == sizeof(nk_e4m3_t)) nk_reduce_add_e4m3_skylake_contiguous_(data, count, result);
    else if (stride_bytes % sizeof(nk_e4m3_t) == 0 && stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_add_e4m3_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_e4m3_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_e5m2_skylake_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero_ps();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)(data + idx)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b512_vec_t vec;
        nk_partial_load_e5m2x16_to_f32x16_skylake_(data + idx, &vec, remaining);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, vec.zmm_ps);
    }
    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_INTERNAL void nk_reduce_add_e5m2_skylake_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    // Masked load zeros non-stride positions; e5m2 zero (0x00) converts to 0.0f, no sum impact.
    __mmask16 stride_mask_m16 = (__mmask16)nk_stride_mask_u1x64_(stride_elements);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m128i data_e5m2x16 = _mm_maskz_loadu_epi8(stride_mask_m16, data + idx_scalars);
        __m512 data_f32x16 = nk_e5m2x16_to_f32x16_skylake_(data_e5m2x16);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
    }
    // Tail: combine stride mask with byte-count mask for remaining bytes
    nk_size_t remaining_bytes = total_scalars - idx_scalars;
    if (remaining_bytes > 0) {
        __mmask16 tail_mask = stride_mask_m16 & (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining_bytes);
        __m128i data_e5m2x16 = _mm_maskz_loadu_epi8(tail_mask, data + idx_scalars);
        sum_f32x16 = _mm512_add_ps(sum_f32x16, nk_e5m2x16_to_f32x16_skylake_(data_e5m2x16));
    }
    *result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

NK_PUBLIC void nk_reduce_add_e5m2_skylake(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    if (stride_bytes == sizeof(nk_e5m2_t)) nk_reduce_add_e5m2_skylake_contiguous_(data, count, result);
    else if (stride_bytes % sizeof(nk_e5m2_t) == 0 && stride_elements >= 2 && stride_elements <= 16)
        nk_reduce_add_e5m2_skylake_strided_(data, count, stride_elements, result);
    else nk_reduce_add_e5m2_serial(data, count, stride_bytes, result);
}

NK_INTERNAL __m512i nk_fp6x64_to_u8x64_comparable_skylake_(__m512i raw_i8x64) {
    raw_i8x64 = _mm512_and_si512(raw_i8x64, _mm512_set1_epi8(0x3F)); // mask to 6 valid bits
    __mmask64 neg_m64 = _mm512_test_epi8_mask(raw_i8x64, _mm512_set1_epi8(0x20));
    __m512i pos_xor_i8x64 = _mm512_set1_epi8(0x20);
    __m512i neg_xor_i8x64 = _mm512_set1_epi8(0x3F);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(raw_i8x64, xor_i8x64);
}

NK_INTERNAL __m512i nk_u8x64_comparable_to_fp6x64_skylake_(__m512i cmp_i8x64) {
    __mmask64 was_neg_m64 = _mm512_cmplt_epu8_mask(cmp_i8x64, _mm512_set1_epi8(0x20));
    __m512i neg_xor_i8x64 = _mm512_set1_epi8(0x3F);
    __m512i pos_xor_i8x64 = _mm512_set1_epi8(0x20);
    __m512i xor_i8x64 = _mm512_mask_mov_epi8(pos_xor_i8x64, was_neg_m64, neg_xor_i8x64);
    return _mm512_xor_si512(cmp_i8x64, xor_i8x64);
}

NK_INTERNAL void nk_reduce_min_e2m3_skylake_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,              //
    nk_e2m3_t *min_value, nk_size_t *min_index) {
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t min_vec, argmin_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF);
    argmin_vec.zmm = _mm512_setzero_si512();
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;
nk_reduce_min_e2m3_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        // Pad tail with 0x1F (most positive fp6 raw → comparable 0x3F, never becomes min)
        data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x1F), tail_mask, data);
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
    // E2M3/E3M2 have no NaN encodings, so plain min/max_epu8 on comparable form is correct
    __m512i new_min_cmp_u8x64 = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_min_cmp_u8x64, min_vec.zmm);
    argmin_vec.zmm = _mm512_mask_mov_epi64(argmin_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    min_vec.zmm = new_min_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_min_e2m3_skylake_cycle_;
    nk_size_t first_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmin_vec.i64s[first_lane / 8];
    *min_index = chunk_idx * 64 + (first_lane % 64);
    min_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(min_vec.zmm);
    *min_value = min_vec.e2m3s[first_lane];
}

NK_PUBLIC void nk_reduce_min_e2m3_skylake(                          //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index) {
    if (count == 0) *(nk_u8_t *)min_value = 0, *min_index = count;
    else if (stride_bytes == sizeof(nk_e2m3_t) && count >= 64)
        nk_reduce_min_e2m3_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e2m3_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e2m3_skylake_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,              //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t max_vec, argmax_vec;
    max_vec.zmm = _mm512_setzero_si512();
    argmax_vec.zmm = _mm512_setzero_si512();
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;
nk_reduce_max_e2m3_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        // Pad tail with 0x3F (most negative fp6 raw → comparable 0x00, never becomes max)
        data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x3F), tail_mask, data);
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
    // E2M3/E3M2 have no NaN encodings, so plain min/max_epu8 on comparable form is correct
    __m512i new_max_cmp_u8x64 = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_max_cmp_u8x64, max_vec.zmm);
    argmax_vec.zmm = _mm512_mask_mov_epi64(argmax_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    max_vec.zmm = new_max_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_max_e2m3_skylake_cycle_;
    nk_size_t first_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmax_vec.i64s[first_lane / 8];
    *max_index = chunk_idx * 64 + (first_lane % 64);
    max_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(max_vec.zmm);
    *max_value = max_vec.e2m3s[first_lane];
}

NK_PUBLIC void nk_reduce_max_e2m3_skylake(                          //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    if (count == 0) *(nk_u8_t *)max_value = 0, *max_index = count;
    else if (stride_bytes == sizeof(nk_e2m3_t) && count >= 64)
        nk_reduce_max_e2m3_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e2m3_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_e3m2_skylake_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,              //
    nk_e3m2_t *min_value, nk_size_t *min_index) {
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t min_vec, argmin_vec;
    min_vec.zmm = _mm512_set1_epi8((char)0xFF);
    argmin_vec.zmm = _mm512_setzero_si512();
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;
nk_reduce_min_e3m2_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        // Pad tail with 0x1F (most positive fp6 raw → comparable 0x3F, never becomes min)
        data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x1F), tail_mask, data);
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
    // E2M3/E3M2 have no NaN encodings, so plain min/max_epu8 on comparable form is correct
    __m512i new_min_cmp_u8x64 = _mm512_min_epu8(min_vec.zmm, data_cmp_u8x64);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_min_cmp_u8x64, min_vec.zmm);
    argmin_vec.zmm = _mm512_mask_mov_epi64(argmin_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    min_vec.zmm = new_min_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_min_e3m2_skylake_cycle_;
    nk_size_t first_lane = nk_argmin_u8x64_skylake_(min_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmin_vec.i64s[first_lane / 8];
    *min_index = chunk_idx * 64 + (first_lane % 64);
    min_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(min_vec.zmm);
    *min_value = min_vec.e3m2s[first_lane];
}

NK_PUBLIC void nk_reduce_min_e3m2_skylake(                          //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index) {
    if (count == 0) *(nk_u8_t *)min_value = 0, *min_index = count;
    else if (stride_bytes == sizeof(nk_e3m2_t) && count >= 64)
        nk_reduce_min_e3m2_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e3m2_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e3m2_skylake_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,              //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    __m512i one_i64x8 = _mm512_set1_epi64(1);
    nk_b512_vec_t max_vec, argmax_vec;
    max_vec.zmm = _mm512_setzero_si512();
    argmax_vec.zmm = _mm512_setzero_si512();
    __m512i current_chunk_i64x8 = _mm512_setzero_si512();
    __m512i data_i8x64;
nk_reduce_max_e3m2_skylake_cycle_:
    if (count < 64) {
        __mmask64 tail_mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count);
        // Pad tail with 0x3F (most negative fp6 raw → comparable 0x00, never becomes max)
        data_i8x64 = _mm512_mask_loadu_epi8(_mm512_set1_epi8(0x3F), tail_mask, data);
        count = 0;
    }
    else {
        data_i8x64 = _mm512_loadu_si512(data);
        data += 64, count -= 64;
    }
    __m512i data_cmp_u8x64 = nk_fp6x64_to_u8x64_comparable_skylake_(data_i8x64);
    // E2M3/E3M2 have no NaN encodings, so plain min/max_epu8 on comparable form is correct
    __m512i new_max_cmp_u8x64 = _mm512_max_epu8(max_vec.zmm, data_cmp_u8x64);
    __mmask8 chunk_updated_m8 = _mm512_cmpneq_epi64_mask(new_max_cmp_u8x64, max_vec.zmm);
    argmax_vec.zmm = _mm512_mask_mov_epi64(argmax_vec.zmm, chunk_updated_m8, current_chunk_i64x8);
    max_vec.zmm = new_max_cmp_u8x64;
    current_chunk_i64x8 = _mm512_add_epi64(current_chunk_i64x8, one_i64x8);
    if (count) goto nk_reduce_max_e3m2_skylake_cycle_;
    nk_size_t first_lane = nk_argmax_u8x64_skylake_(max_vec.zmm);
    nk_size_t chunk_idx = (nk_size_t)argmax_vec.i64s[first_lane / 8];
    *max_index = chunk_idx * 64 + (first_lane % 64);
    max_vec.zmm = nk_u8x64_comparable_to_fp6x64_skylake_(max_vec.zmm);
    *max_value = max_vec.e3m2s[first_lane];
}

NK_PUBLIC void nk_reduce_max_e3m2_skylake(                          //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    if (count == 0) *(nk_u8_t *)max_value = 0, *max_index = count;
    else if (stride_bytes == sizeof(nk_e3m2_t) && count >= 64)
        nk_reduce_max_e3m2_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e3m2_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i4_skylake_contiguous_( //
    nk_i4x2_t const *data, nk_size_t count, nk_i64_t *result) {
    __m512i sum_low_u64x8 = _mm512_setzero_si512();
    __m512i sum_high_u64x8 = _mm512_setzero_si512();
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    nk_size_t count_nibbles = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
nk_reduce_add_i4_skylake_cycle_:;
    __m512i raw_i8x64;
    if (count_bytes < 64) {
        __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
        raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
        count_bytes = 0;
    }
    else {
        raw_i8x64 = _mm512_loadu_si512(ptr);
        ptr += 64, count_bytes -= 64;
    }
    __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
    __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
    sum_low_u64x8 = _mm512_add_epi64(sum_low_u64x8, _mm512_sad_epu8(low_i8x64, zero_i8x64));
    sum_high_u64x8 = _mm512_add_epi64(sum_high_u64x8, _mm512_sad_epu8(high_i8x64, zero_i8x64));
    if (count_bytes) goto nk_reduce_add_i4_skylake_cycle_;
    // Combine accumulators with one vector add, then a single horizontal reduce
    nk_i64_t sum = (nk_i64_t)_mm512_reduce_add_epi64(_mm512_add_epi64(sum_low_u64x8, sum_high_u64x8));
    // Handle odd nibble: the last byte's high nibble was accumulated but shouldn't be
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        sum -= (last_byte >> 4) & 0x0F;
    }
    // Each nibble was accumulated as unsigned [0..15]; signed value = unsigned - 8
    sum -= (nk_i64_t)8 * count_nibbles;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i4_skylake(                            //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == 1) nk_reduce_add_i4_skylake_contiguous_(data, count, result);
    else nk_reduce_add_i4_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u4_skylake_contiguous_( //
    nk_u4x2_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i sum_u64x8 = _mm512_setzero_si512();
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i zero_i8x64 = _mm512_setzero_si512();
    nk_size_t count_nibbles = count;
    nk_size_t count_bytes = nk_size_divide_round_up_(count, 2);
    unsigned char const *ptr = (unsigned char const *)data;
nk_reduce_add_u4_skylake_cycle_:;
    __m512i raw_i8x64;
    if (count_bytes < 64) {
        __mmask64 tail_mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)count_bytes);
        raw_i8x64 = _mm512_maskz_loadu_epi8(tail_mask, ptr);
        count_bytes = 0;
    }
    else {
        raw_i8x64 = _mm512_loadu_si512(ptr);
        ptr += 64, count_bytes -= 64;
    }
    __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
    __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
    __m512i pair_sum_i8x64 = _mm512_add_epi8(low_i8x64, high_i8x64);
    sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(pair_sum_i8x64, zero_i8x64));
    if (count_bytes) goto nk_reduce_add_u4_skylake_cycle_;
    nk_u64_t sum = _mm512_reduce_add_epi64(sum_u64x8);
    // Handle odd nibble: the last byte's high nibble was accumulated but shouldn't be
    if (count_nibbles & 1) {
        nk_u8_t last_byte = ((unsigned char const *)data)[count_nibbles / 2];
        sum -= (last_byte >> 4) & 0x0F;
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u4_skylake(                            //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == 1) nk_reduce_add_u4_skylake_contiguous_(data, count, result);
    else nk_reduce_add_u4_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_i4_skylake_contiguous_( //
    nk_i4x2_t const *data, nk_size_t count,            //
    nk_i8_t *min_value, nk_size_t *min_index) {
    __m512i min_u8x64 = _mm512_set1_epi8((char)0xFF);
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    nk_size_t bytes = count / 2;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        min_u8x64 = _mm512_min_epu8(min_u8x64, low_i8x64);
        min_u8x64 = _mm512_min_epu8(min_u8x64, high_i8x64);
    }
    nk_u8_t min_nibble = nk_reduce_min_u8x64_skylake_(min_u8x64);
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ((unsigned char const *)data)[idx];
        nk_u8_t low_val = byte_val & 0x0F, high_val = (byte_val >> 4) & 0x0F;
        if (low_val < min_nibble) min_nibble = low_val;
        if (high_val < min_nibble) min_nibble = high_val;
    }
    if (count & 1) {
        nk_u8_t low_val = ((unsigned char const *)data)[bytes] & 0x0F;
        if (low_val < min_nibble) min_nibble = low_val;
    }
    *min_value = (nk_i8_t)((min_nibble ^ 8) - 8);
    // Vectorized search for the first occurrence of min_nibble
    __m512i target_i8x64 = _mm512_set1_epi8((char)min_nibble);
    for (nk_size_t search_idx = 0; search_idx + 64 <= bytes; search_idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + search_idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __mmask64 low_match = _mm512_cmpeq_epi8_mask(low_i8x64, target_i8x64);
        __mmask64 high_match = _mm512_cmpeq_epi8_mask(high_i8x64, target_i8x64);
        nk_u64_t or_match = (nk_u64_t)(low_match | high_match);
        if (or_match) {
            nk_size_t first_byte = (nk_size_t)_tzcnt_u64(or_match);
            nk_size_t is_high = !(((nk_u64_t)low_match >> first_byte) & 1);
            *min_index = search_idx * 2 + 2 * first_byte + is_high;
            return;
        }
    }
    // Scalar tail for remaining < 64 bytes
    for (nk_size_t i = (bytes / 64) * 64 * 2; i < count; ++i) {
        nk_u8_t byte_val = ((unsigned char const *)data)[i / 2];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        if (nibble == min_nibble) {
            *min_index = i;
            return;
        }
    }
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_i4_skylake(                            //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = 7, *min_index = count;
    else if (stride_bytes == 1) nk_reduce_min_i4_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i4_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i4_skylake_contiguous_( //
    nk_i4x2_t const *data, nk_size_t count,            //
    nk_i8_t *max_value, nk_size_t *max_index) {
    __m512i max_u8x64 = _mm512_setzero_si512();
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    nk_size_t bytes = count / 2;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        max_u8x64 = _mm512_max_epu8(max_u8x64, low_i8x64);
        max_u8x64 = _mm512_max_epu8(max_u8x64, high_i8x64);
    }
    nk_u8_t max_nibble = nk_reduce_max_u8x64_skylake_(max_u8x64);
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ((unsigned char const *)data)[idx];
        nk_u8_t low_val = byte_val & 0x0F, high_val = (byte_val >> 4) & 0x0F;
        if (low_val > max_nibble) max_nibble = low_val;
        if (high_val > max_nibble) max_nibble = high_val;
    }
    if (count & 1) {
        nk_u8_t low_val = ((unsigned char const *)data)[bytes] & 0x0F;
        if (low_val > max_nibble) max_nibble = low_val;
    }
    *max_value = (nk_i8_t)((max_nibble ^ 8) - 8);
    // Vectorized search for the first occurrence of max_nibble
    __m512i target_i8x64 = _mm512_set1_epi8((char)max_nibble);
    for (nk_size_t search_idx = 0; search_idx + 64 <= bytes; search_idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + search_idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __mmask64 low_match = _mm512_cmpeq_epi8_mask(low_i8x64, target_i8x64);
        __mmask64 high_match = _mm512_cmpeq_epi8_mask(high_i8x64, target_i8x64);
        nk_u64_t or_match = (nk_u64_t)(low_match | high_match);
        if (or_match) {
            nk_size_t first_byte = (nk_size_t)_tzcnt_u64(or_match);
            nk_size_t is_high = !(((nk_u64_t)low_match >> first_byte) & 1);
            *max_index = search_idx * 2 + 2 * first_byte + is_high;
            return;
        }
    }
    // Scalar tail for remaining < 64 bytes
    for (nk_size_t i = (bytes / 64) * 64 * 2; i < count; ++i) {
        nk_u8_t byte_val = ((unsigned char const *)data)[i / 2];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        if (nibble == max_nibble) {
            *max_index = i;
            return;
        }
    }
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_i4_skylake(                            //
    nk_i4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = -8, *max_index = count;
    else if (stride_bytes == 1) nk_reduce_max_i4_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i4_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_u4_skylake_contiguous_( //
    nk_u4x2_t const *data, nk_size_t count,            //
    nk_u8_t *min_value, nk_size_t *min_index) {
    __m512i min_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    nk_size_t bytes = count / 2;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        min_u8x64 = _mm512_min_epu8(min_u8x64, low_i8x64);
        min_u8x64 = _mm512_min_epu8(min_u8x64, high_i8x64);
    }
    nk_u8_t min_val = nk_reduce_min_u8x64_skylake_(min_u8x64);
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ((unsigned char const *)data)[idx];
        nk_u8_t low_val = byte_val & 0x0F, high_val = (byte_val >> 4) & 0x0F;
        if (low_val < min_val) min_val = low_val;
        if (high_val < min_val) min_val = high_val;
    }
    if (count & 1) {
        nk_u8_t low_val = ((unsigned char const *)data)[bytes] & 0x0F;
        if (low_val < min_val) min_val = low_val;
    }
    *min_value = min_val;
    // Vectorized search for the first occurrence of min_val
    __m512i target_i8x64 = _mm512_set1_epi8((char)min_val);
    for (nk_size_t search_idx = 0; search_idx + 64 <= bytes; search_idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + search_idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __mmask64 low_match = _mm512_cmpeq_epi8_mask(low_i8x64, target_i8x64);
        __mmask64 high_match = _mm512_cmpeq_epi8_mask(high_i8x64, target_i8x64);
        nk_u64_t or_match = (nk_u64_t)(low_match | high_match);
        if (or_match) {
            nk_size_t first_byte = (nk_size_t)_tzcnt_u64(or_match);
            nk_size_t is_high = !(((nk_u64_t)low_match >> first_byte) & 1);
            *min_index = search_idx * 2 + 2 * first_byte + is_high;
            return;
        }
    }
    // Scalar tail for remaining < 64 bytes
    for (nk_size_t i = (bytes / 64) * 64 * 2; i < count; ++i) {
        nk_u8_t byte_val = ((unsigned char const *)data)[i / 2];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        if (nibble == min_val) {
            *min_index = i;
            return;
        }
    }
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_u4_skylake(                            //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = 15, *min_index = count;
    else if (stride_bytes == 1) nk_reduce_min_u4_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u4_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u4_skylake_contiguous_( //
    nk_u4x2_t const *data, nk_size_t count,            //
    nk_u8_t *max_value, nk_size_t *max_index) {
    __m512i max_u8x64 = _mm512_setzero_si512();
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    nk_size_t bytes = count / 2;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        max_u8x64 = _mm512_max_epu8(max_u8x64, low_i8x64);
        max_u8x64 = _mm512_max_epu8(max_u8x64, high_i8x64);
    }
    nk_u8_t max_val = nk_reduce_max_u8x64_skylake_(max_u8x64);
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ((unsigned char const *)data)[idx];
        nk_u8_t low_val = byte_val & 0x0F, high_val = (byte_val >> 4) & 0x0F;
        if (low_val > max_val) max_val = low_val;
        if (high_val > max_val) max_val = high_val;
    }
    if (count & 1) {
        nk_u8_t low_val = ((unsigned char const *)data)[bytes] & 0x0F;
        if (low_val > max_val) max_val = low_val;
    }
    *max_value = max_val;
    // Vectorized search for the first occurrence of max_val
    __m512i target_i8x64 = _mm512_set1_epi8((char)max_val);
    for (nk_size_t search_idx = 0; search_idx + 64 <= bytes; search_idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + search_idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __mmask64 low_match = _mm512_cmpeq_epi8_mask(low_i8x64, target_i8x64);
        __mmask64 high_match = _mm512_cmpeq_epi8_mask(high_i8x64, target_i8x64);
        nk_u64_t or_match = (nk_u64_t)(low_match | high_match);
        if (or_match) {
            nk_size_t first_byte = (nk_size_t)_tzcnt_u64(or_match);
            nk_size_t is_high = !(((nk_u64_t)low_match >> first_byte) & 1);
            *max_index = search_idx * 2 + 2 * first_byte + is_high;
            return;
        }
    }
    // Scalar tail for remaining < 64 bytes
    for (nk_size_t i = (bytes / 64) * 64 * 2; i < count; ++i) {
        nk_u8_t byte_val = ((unsigned char const *)data)[i / 2];
        nk_u8_t nibble = (i % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
        if (nibble == max_val) {
            *max_index = i;
            return;
        }
    }
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_u4_skylake(                            //
    nk_u4x2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = 0, *max_index = count;
    else if (stride_bytes == 1) nk_reduce_max_u4_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u4_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_u1_skylake_contiguous_( //
    nk_u1x8_t const *data, nk_size_t count, nk_u64_t *result) {
    __m512i lut_i8x64 = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4));
    __m512i mask_0f_i8x64 = _mm512_set1_epi8(0x0F);
    __m512i sum_u64x8 = _mm512_setzero_si512();
    nk_size_t bytes = (count + 7) / 8;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512((unsigned char const *)data + idx);
        __m512i low_i8x64 = _mm512_and_si512(raw_i8x64, mask_0f_i8x64);
        __m512i high_i8x64 = _mm512_and_si512(_mm512_srli_epi16(raw_i8x64, 4), mask_0f_i8x64);
        __m512i popcnt_i8x64 = _mm512_add_epi8(_mm512_shuffle_epi8(lut_i8x64, low_i8x64),
                                               _mm512_shuffle_epi8(lut_i8x64, high_i8x64));
        sum_u64x8 = _mm512_add_epi64(sum_u64x8, _mm512_sad_epu8(popcnt_i8x64, _mm512_setzero_si512()));
    }
    nk_u64_t sum = _mm512_reduce_add_epi64(sum_u64x8);
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ((unsigned char const *)data)[idx];
        sum += nk_u64_popcount_((nk_u64_t)byte_val);
    }
    nk_size_t tail_bits = count % 8;
    if (tail_bits && bytes > 0) {
        nk_u8_t last_byte = ((unsigned char const *)data)[bytes - 1];
        nk_u8_t extra_mask = (nk_u8_t)(0xFF << tail_bits);
        sum -= nk_u64_popcount_((nk_u64_t)(last_byte & extra_mask));
    }
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u1_skylake(                            //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == 1) nk_reduce_add_u1_skylake_contiguous_(data, count, result);
    else nk_reduce_add_u1_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_u1_skylake_contiguous_( //
    nk_u1x8_t const *data, nk_size_t count,            //
    nk_u8_t *min_value, nk_size_t *min_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t bytes = (count + 7) / 8;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512(ptr + idx);
        __mmask64 not_all_ones = _mm512_cmpneq_epi8_mask(raw_i8x64, _mm512_set1_epi8((char)0xFF));
        if (not_all_ones) {
            nk_size_t byte_off = idx + (nk_size_t)_tzcnt_u64((nk_u64_t)not_all_ones);
            nk_u8_t byte_val = ptr[byte_off];
            nk_u8_t inv_byte = (nk_u8_t)~byte_val;
            nk_size_t bit = (nk_size_t)_tzcnt_u32((nk_u32_t)inv_byte);
            nk_size_t pos = byte_off * 8 + bit;
            if (pos < count) {
                *min_value = 0;
                *min_index = pos;
                return;
            }
        }
    }
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ptr[idx];
        if (byte_val != 0xFF) {
            nk_u8_t inv_byte = (nk_u8_t)~byte_val;
            nk_size_t bit = (nk_size_t)_tzcnt_u32((nk_u32_t)inv_byte);
            nk_size_t pos = idx * 8 + bit;
            if (pos < count) {
                *min_value = 0;
                *min_index = pos;
                return;
            }
        }
    }
    *min_value = count ? 1 : 0;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_u1_skylake(                            //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (count == 0) *min_value = 0, *min_index = count;
    else if (stride_bytes == 1) nk_reduce_min_u1_skylake_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u1_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u1_skylake_contiguous_( //
    nk_u1x8_t const *data, nk_size_t count,            //
    nk_u8_t *max_value, nk_size_t *max_index) {
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t bytes = (count + 7) / 8;
    nk_size_t idx = 0;
    for (; idx + 64 <= bytes; idx += 64) {
        __m512i raw_i8x64 = _mm512_loadu_si512(ptr + idx);
        __mmask64 nonzero = _mm512_cmpneq_epi8_mask(raw_i8x64, _mm512_setzero_si512());
        if (nonzero) {
            nk_size_t byte_off = idx + (nk_size_t)_tzcnt_u64((nk_u64_t)nonzero);
            nk_u8_t byte_val = ptr[byte_off];
            nk_size_t bit = (nk_size_t)_tzcnt_u32((nk_u32_t)byte_val);
            nk_size_t pos = byte_off * 8 + bit;
            if (pos < count) {
                *max_value = 1;
                *max_index = pos;
                return;
            }
        }
    }
    for (; idx < bytes; idx++) {
        nk_u8_t byte_val = ptr[idx];
        if (byte_val != 0) {
            nk_size_t bit = (nk_size_t)_tzcnt_u32((nk_u32_t)byte_val);
            nk_size_t pos = idx * 8 + bit;
            if (pos < count) {
                *max_value = 1;
                *max_index = pos;
                return;
            }
        }
    }
    *max_value = 0;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_u1_skylake(                            //
    nk_u1x8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (count == 0) *max_value = 0, *max_index = count;
    else if (stride_bytes == 1) nk_reduce_max_u1_skylake_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u1_serial(data, count, stride_bytes, max_value, max_index);
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
#endif // NK_REDUCE_SKYLAKE_H
