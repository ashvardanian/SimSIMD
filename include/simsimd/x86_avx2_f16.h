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

inline static simsimd_f32_t simsimd_avx2_f16_l2sq(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 d2_vec = _mm256_set1_ps(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }

    d2_vec = _mm256_add_ps(_mm256_permute2f128_ps(d2_vec, d2_vec, 1), d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);

    simsimd_f32_t result[1];
    _mm_store_ss(result, _mm256_castps256_ps128(d2_vec));

    // Accumulate the tail:
    for (; i < n; ++i)
        result[0] += (a[i] - b[i]) * (a[i] - b[i]);
    return result[0];
}

inline static simsimd_f32_t simsimd_avx2_f16_ip(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 ab_vec = _mm256_set1_ps(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
    }

    ab_vec = _mm256_add_ps(_mm256_permute2f128_ps(ab_vec, ab_vec, 1), ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);

    simsimd_f32_t result[1];
    _mm_store_ss(result, _mm256_castps256_ps128(ab_vec));

    // Accumulate the tail:
    for (; i < n; ++i)
        result[0] += a[i] * b[i];
    return 1 - result[0];
}

inline static simsimd_f32_t simsimd_avx2_f16_cos(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 ab_vec = _mm256_set1_ps(0), a2_vec = _mm256_set1_ps(0), b2_vec = _mm256_set1_ps(0);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        ab_vec = _mm256_fmadd_ps(a_vec, b_vec, ab_vec);
        a2_vec = _mm256_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm256_fmadd_ps(b_vec, b_vec, b2_vec);
    }

    ab_vec = _mm256_add_ps(_mm256_permute2f128_ps(ab_vec, ab_vec, 1), ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);
    ab_vec = _mm256_hadd_ps(ab_vec, ab_vec);

    a2_vec = _mm256_add_ps(_mm256_permute2f128_ps(a2_vec, a2_vec, 1), a2_vec);
    a2_vec = _mm256_hadd_ps(a2_vec, a2_vec);
    a2_vec = _mm256_hadd_ps(a2_vec, a2_vec);

    b2_vec = _mm256_add_ps(_mm256_permute2f128_ps(b2_vec, b2_vec, 1), b2_vec);
    b2_vec = _mm256_hadd_ps(b2_vec, b2_vec);
    b2_vec = _mm256_hadd_ps(b2_vec, b2_vec);

    simsimd_f32_t ab, a2, b2;
    _mm_store_ss(&ab, _mm256_castps256_ps128(ab_vec));
    _mm_store_ss(&a2, _mm256_castps256_ps128(a2_vec));
    _mm_store_ss(&b2, _mm256_castps256_ps128(b2_vec));

    // Accumulate the tail:
    for (; i < n; ++i)
        ab += a[i] * b[i], a2 += a[i] * a[i], b2 += b[i] * b[i];

    // Replace simsimd_approximate_inverse_square_root with `rsqrtss`
    __m128 a2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)a2));
    __m128 b2_sqrt_recip = _mm_rsqrt_ss(_mm_set_ss((float)b2));
    __m128 result = _mm_mul_ss(a2_sqrt_recip, b2_sqrt_recip); // Multiply the reciprocal square roots
    result = _mm_mul_ss(result, _mm_set_ss((float)ab));       // Multiply by ab
    result = _mm_sub_ss(_mm_set_ss(1.0f), result);            // Subtract from 1
    return _mm_cvtss_f32(result);                             // Extract the final result
}

#ifdef __cplusplus
} // extern "C"
#endif