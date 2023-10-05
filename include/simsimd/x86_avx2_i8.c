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

simsimd_f32_t simsimd_avx2_i8_l2sq(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {

    __m256i d2_high_vec = _mm256_setzero_si256();
    __m256i d2_low_vec = _mm256_setzero_si256();

    simsimd_size_t i = 0;
    for (; i + 31 < d; i += 32) {
        __m256i a_vec = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i const*)(b + i));

        // Sign extend int8 to int16
        __m256i a_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_vec));
        __m256i a_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 1));
        __m256i b_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_vec));
        __m256i b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));

        // Subtract and multiply
        __m256i d_low = _mm256_sub_epi16(a_low, b_low);
        __m256i d_high = _mm256_sub_epi16(a_high, b_high);
        __m256i d2_low_part = _mm256_madd_epi16(d_low, d_low);
        __m256i d2_high_part = _mm256_madd_epi16(d_high, d_high);

        // Accumulate into int32 vectors
        d2_low_vec = _mm256_add_epi32(d2_low_vec, d2_low_part);
        d2_high_vec = _mm256_add_epi32(d2_high_vec, d2_high_part);
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

simsimd_f32_t simsimd_avx2_i8_cos(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {

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

        // Unpack int8 to int32
        __m256i a_low = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(a_vec));
        __m256i a_high = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(a_vec, 1));
        __m256i b_low = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(b_vec));
        __m256i b_high = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(b_vec, 1));

        // Multiply and accumulate
        ab_low_vec = _mm256_add_epi32(ab_low_vec, _mm256_mullo_epi32(a_low, b_low));
        ab_high_vec = _mm256_add_epi32(ab_high_vec, _mm256_mullo_epi32(a_high, b_high));
        a2_low_vec = _mm256_add_epi32(a2_low_vec, _mm256_mullo_epi32(a_low, a_low));
        a2_high_vec = _mm256_add_epi32(a2_high_vec, _mm256_mullo_epi32(a_high, a_high));
        b2_low_vec = _mm256_add_epi32(b2_low_vec, _mm256_mullo_epi32(b_low, b_low));
        b2_high_vec = _mm256_add_epi32(b2_high_vec, _mm256_mullo_epi32(b_high, b_high));
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

    // Replace simsimd_approximate_inverse_square_root with `rsqrtss`
    __m128 a2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)a2));
    __m128 b2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)b2));
    __m128 result = _mm_mul_ss(a2_sqrt_recip, b2_sqrt_recip); // Multiply the reciprocal square roots
    result = _mm_mul_ss(result, _mm_set_ss((float)ab));       // Multiply by ab
    result = _mm_sub_ss(_mm_set_ss(1.0f), result);            // Subtract from 1
    return _mm_cvtss_f32(result);                             // Extract the final result
}

simsimd_f32_t simsimd_avx2_i8_ip(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t d) {
    return simsimd_avx2_i8_cos(a, b, d);
}
