/**
 *  @file   x86_avx512_f16.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, cosine similarity, inner product (same as cosine).
 *  - Uses `_mm512_maskz_loadu_epi16` intrinsics to perform masked unaligned loads.
 *  - Uses `i8` for storage, `i16` for multiplication, and `i32` for accumulation, if no better option is available.
 *  - Requires compiler capabilities: avx512fp16, avx512f, avx512vl.
 */
#include <immintrin.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

inline static simsimd_f32_t simsimd_avx512_i8_l2sq(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    __m512i d2_i32s_vec = _mm512_setzero_si512();
    simsimd_size_t i = 0;

    do {
        __mmask32 mask = d - i >= 32 ? 0xFFFFFFFF : ((1u << (d - i)) - 1u);
        __m512i a_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a + i)); // Load 8-bit integers
        __m512i b_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b + i)); // Load 8-bit integers
        __m512i d_i16s_vec = _mm512_sub_epi16(a_vec, b_vec);
        d2_i32s_vec = _mm512_add_epi32(d2_i32s_vec, _mm512_madd_epi16(d_i16s_vec, d_i16s_vec));

        i += 32;
    } while (i < d);

    return _mm512_reduce_add_epi32(d2_i32s_vec);
}

inline static simsimd_f32_t simsimd_avx512_i8_cos(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    __m512i ab_i32s_vec = _mm512_setzero_si512();
    __m512i a2_i32s_vec = _mm512_setzero_si512();
    __m512i b2_i32s_vec = _mm512_setzero_si512();
    simsimd_size_t i = 0;

    do {
        __mmask32 mask = d - i >= 32 ? 0xFFFFFFFF : ((1u << (d - i)) - 1u);
        __m512i a_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a + i)); // Load 8-bit integers
        __m512i b_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b + i)); // Load 8-bit integers

        ab_i32s_vec = _mm512_add_epi32(ab_i32s_vec, _mm512_madd_epi16(a_vec, b_vec));
        a2_i32s_vec = _mm512_add_epi32(a2_i32s_vec, _mm512_madd_epi16(a_vec, a_vec));
        b2_i32s_vec = _mm512_add_epi32(b2_i32s_vec, _mm512_madd_epi16(b_vec, b_vec));

        i += 32;
    } while (i < d);

    int ab = _mm512_reduce_add_epi32(ab_i32s_vec);
    int a2 = _mm512_reduce_add_epi32(a2_i32s_vec);
    int b2 = _mm512_reduce_add_epi32(b2_i32s_vec);

    __m128d a2_b2 = _mm_set_pd((double)a2, (double)b2);
    __m128d rsqrts = _mm_mask_rsqrt14_pd(_mm_setzero_pd(), 0xFF, a2_b2);
    double rsqrts_array[2];
    _mm_storeu_pd(rsqrts_array, rsqrts);
    return 1 - ab * rsqrts_array[0] * rsqrts_array[1];
}

inline static simsimd_f32_t simsimd_avx512_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_avx512_i8_cos(a, b, d);
}

#ifdef __cplusplus
} // extern "C"
#endif