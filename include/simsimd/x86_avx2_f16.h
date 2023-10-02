/**
 *  @file   x86_avx2_f16.h
 *  @brief  x86 AVX2 implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - As AVX2 doesn't support masked loads of 16-bit words, implementations have a separate `for`-loop for tails.
 *  - Uses `f16` for both storage and `f32` for accumulation.
 *  - Requires compiler capabilities: avx2, f16c, fma.
 */
#include <immintrin.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

inline static simsimd_f32_t simsimd_avx2_f16_l2sq(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t d) {
    __m256 sum_vec = _mm256_set1_ps(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        __m256 diff_vec = _mm256_sub_ps(a_vec, b_vec);
        sum_vec = _mm256_fmadd_ps(diff_vec, diff_vec, sum_vec);
    }
    sum_vec = _mm256_add_ps(_mm256_permute2f128_ps(sum_vec, sum_vec, 0x81), sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    simsimd_f32_t result[1];
    _mm_store_ss(result, _mm256_castps256_ps128(sum_vec));

    // Accumulate the tail:
    for (; i < d; ++i) {
        simsimd_f32_t diff = a[i] - b[i];
        result[0] += diff * diff;
    }
    return result[0];
}

inline static simsimd_f32_t simsimd_avx2_f16_ip(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t d) {
    __m256 sum_vec = _mm256_set1_ps(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    sum_vec = _mm256_add_ps(_mm256_permute2f128_ps(sum_vec, sum_vec, 0x81), sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

    simsimd_f32_t result[1];
    _mm_store_ss(result, _mm256_castps256_ps128(sum_vec));

    // Accumulate the tail:
    for (; i < d; ++i)
        result[0] += a[i] * b[i];
    return result[0];
}

inline static simsimd_f32_t simsimd_avx2_f16_cos(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t d) {
    __m256 sum_ab = _mm256_set1_ps(0), sum_a2 = _mm256_set1_ps(0), sum_b2 = _mm256_set1_ps(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        sum_ab = _mm256_fmadd_ps(a_vec, b_vec, sum_ab);
        sum_a2 = _mm256_fmadd_ps(a_vec, a_vec, sum_a2);
        sum_b2 = _mm256_fmadd_ps(b_vec, b_vec, sum_b2);
    }

    sum_ab = _mm256_add_ps(_mm256_permute2f128_ps(sum_ab, sum_ab, 0x81), sum_ab);
    sum_ab = _mm256_hadd_ps(sum_ab, sum_ab);
    sum_ab = _mm256_hadd_ps(sum_ab, sum_ab);

    sum_a2 = _mm256_add_ps(_mm256_permute2f128_ps(sum_a2, sum_a2, 0x81), sum_a2);
    sum_a2 = _mm256_hadd_ps(sum_a2, sum_a2);
    sum_a2 = _mm256_hadd_ps(sum_a2, sum_a2);

    sum_b2 = _mm256_add_ps(_mm256_permute2f128_ps(sum_b2, sum_b2, 0x81), sum_b2);
    sum_b2 = _mm256_hadd_ps(sum_b2, sum_b2);
    sum_b2 = _mm256_hadd_ps(sum_b2, sum_b2);

    simsimd_f32_t result[3];
    _mm_store_ss(result, _mm256_castps256_ps128(sum_ab));
    _mm_store_ss(result + 1, _mm256_castps256_ps128(sum_a2));
    _mm_store_ss(result + 2, _mm256_castps256_ps128(sum_b2));

    // Accumulate the tail:
    for (; i < d; ++i)
        result[0] += a[i] * b[i], result[1] += a[i] * a[i], result[2] += b[i] * b[i];
    return result[0] * simsimd_approximate_inverse_square_root(result[1] * result[2]);
}

#ifdef __cplusplus
} // extern "C"
#endif