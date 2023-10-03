/**
 *  @file   x86_avx2_i8.h
 *  @brief  x86 AVX2 implementation of the most common similarity metrics for 8-bit signed integral numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, cosine similarity, inner product (same as cosine).
 *  - As AVX2 doesn't support masked loads of 16-bit words, implementations have a separate `for`-loop for tails.
 *  - Uses `i8` for storage, `i16` for multiplication, and `i32` for accumulation, if no better option is available.
 *  - Requires compiler capabilities: avx2, f16c, fma.
 */
#include <immintrin.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

inline static simsimd_f32_t simsimd_avx2_i8_l2sq(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {

    __m256i d2_high_vec = _mm256_setzero_si256();
    __m256i d2_low_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 31 < d; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i const*)(b + i));
        __m256i d_vec = _mm256_sub_epi8(a_vec, b_vec);
        __m256i d_part_vec = _mm256_maddubs_epi16(d_vec, d_vec);

        // Upcast the 16-bit integers from `d_part_vec` into 32-bit integers
        // accumulating half into `d2_high_vec`, and another half into `d2_low_vec`.
        d2_low_vec = _mm256_add_epi32(d2_low_vec, _mm256_unpacklo_epi16(d_part_vec, _mm256_setzero_si256()));
        d2_high_vec = _mm256_add_epi32(d2_high_vec, _mm256_unpackhi_epi16(d_part_vec, _mm256_setzero_si256()));
    }

    // Accumulate the 32-bit integers from `d2_high_vec` and `d2_low_vec`
    __m256i d2_vec = _mm256_add_epi32(d2_low_vec, d2_high_vec);
    __m128i d2_sum = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    d2_sum = _mm_hadd_epi32(d2_sum, d2_sum);
    int d2 = _mm_extract_epi32(d2_sum, 0);

    // Take care of the tail:
    for (; i < d; ++i) {
        int d = a[i] - b[i];
        d2 += d * d;
    }

    return (simsimd_f32_t)d2;
}

inline static simsimd_f32_t simsimd_avx2_i8_cos(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {

    __m256i ab_high_vec = _mm256_setzero_si256();
    __m256i ab_low_vec = _mm256_setzero_si256();
    __m256i a2_high_vec = _mm256_setzero_si256();
    __m256i a2_low_vec = _mm256_setzero_si256();
    __m256i b2_high_vec = _mm256_setzero_si256();
    __m256i b2_low_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 31 < d; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i const*)(b + i));

        // Multiply and add packed 8-bit integers
        __m256i ab_part_vec = _mm256_maddubs_epi16(a_vec, b_vec);
        __m256i a2_part_vec = _mm256_maddubs_epi16(a_vec, a_vec);
        __m256i b2_part_vec = _mm256_maddubs_epi16(b_vec, b_vec);

        // Unpack and accumulate the 16-bit results into 32-bit accumulators
        ab_low_vec = _mm256_add_epi32(ab_low_vec, _mm256_unpacklo_epi16(ab_part_vec, _mm256_setzero_si256()));
        ab_high_vec = _mm256_add_epi32(ab_high_vec, _mm256_unpackhi_epi16(ab_part_vec, _mm256_setzero_si256()));
        a2_low_vec = _mm256_add_epi32(a2_low_vec, _mm256_unpacklo_epi16(a2_part_vec, _mm256_setzero_si256()));
        a2_high_vec = _mm256_add_epi32(a2_high_vec, _mm256_unpackhi_epi16(a2_part_vec, _mm256_setzero_si256()));
        b2_low_vec = _mm256_add_epi32(b2_low_vec, _mm256_unpacklo_epi16(b2_part_vec, _mm256_setzero_si256()));
        b2_high_vec = _mm256_add_epi32(b2_high_vec, _mm256_unpackhi_epi16(b2_part_vec, _mm256_setzero_si256()));
    }

    // Horizontal sum across the 256-bit register
    __m256i ab_vec = _mm256_add_epi32(ab_low_vec, ab_high_vec);
    __m128i ab_sum = _mm_add_epi32(_mm256_extracti128_si256(ab_vec, 0), _mm256_extracti128_si256(ab_vec, 1));
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);
    ab_sum = _mm_hadd_epi32(ab_sum, ab_sum);

    __m256i a2_vec = _mm256_add_epi32(a2_low_vec, a2_high_vec);
    __m128i a2_sum = _mm_add_epi32(_mm256_extracti128_si256(a2_vec, 0), _mm256_extracti128_si256(a2_vec, 1));
    a2_sum = _mm_hadd_epi32(a2_sum, a2_sum);
    a2_sum = _mm_hadd_epi32(a2_sum, a2_sum);

    __m256i b2_vec = _mm256_add_epi32(b2_low_vec, b2_high_vec);
    __m128i b2_sum = _mm_add_epi32(_mm256_extracti128_si256(b2_vec, 0), _mm256_extracti128_si256(b2_vec, 1));
    b2_sum = _mm_hadd_epi32(b2_sum, b2_sum);
    b2_sum = _mm_hadd_epi32(b2_sum, b2_sum);

    // Further reduce to a single sum for each vector
    int ab = _mm_extract_epi32(ab_sum, 0);
    int a2 = _mm_extract_epi32(a2_sum, 0);
    int b2 = _mm_extract_epi32(b2_sum, 0);

    // Take care of the tail:
    for (; i < d; ++i) {
        int ai = a[i], bi = b[i];
        ab += ai * bi, a2 += ai * ai, b2 += bi * bi;
    }

    // return 1 - ab / std::sqrt(a2 * b2);
    return 1 - ab * simsimd_approximate_inverse_square_root(a2 * b2);
}

inline static simsimd_f32_t simsimd_avx2_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_avx2_i8_cos(a, b, d);
}

#ifdef __cplusplus
} // extern "C"
#endif