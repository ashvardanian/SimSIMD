/**
 *  @brief      SIMD-accelerated Similarity Measures for Probability Distributions.
 *  @author     Ash Vardanian
 *  @date       October 20, 2023
 *
 *  Contains:
 *  - Kullback-Leibler divergence
 *  - Jensen–Shannon divergence
 *
 *  For datatypes:
 *  - 32-bit floating point numbers
 *  - 16-bit floating point numbers
 *
 *  For hardware architectures:
 *  - Arm (NEON, SVE)
 *  - x86 (AVX2, AVX512)
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */

#pragma once
#include "types.h"

#define SIMSIMD_MAKE_KL(name, input_type, accumulator_type, converter, epsilon)                                        \
    inline static simsimd_f32_t simsimd_##name##_##input_type##_kl(                                                    \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t n) {                      \
        simsimd_##accumulator_type##_t d = 0;                                                                          \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t bi = converter(b[i]);                                                       \
            d += ai * SIMSIMD_LOG((ai + epsilon) / (bi + epsilon));                                                    \
        }                                                                                                              \
        return d;                                                                                                      \
    }

#define SIMSIMD_MAKE_JS(name, input_type, accumulator_type, converter, epsilon)                                        \
    inline static simsimd_f32_t simsimd_##name##_##input_type##_js(                                                    \
        simsimd_##input_type##_t const* a, simsimd_##input_type##_t const* b, simsimd_size_t n) {                      \
        simsimd_##accumulator_type##_t d = 0;                                                                          \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                      \
            simsimd_##accumulator_type##_t ai = converter(a[i]);                                                       \
            simsimd_##accumulator_type##_t bi = converter(b[i]);                                                       \
            simsimd_##accumulator_type##_t mi = (ai + bi) / 2;                                                         \
            d += ai * SIMSIMD_LOG((ai + epsilon) / (mi + epsilon));                                                    \
            d += bi * SIMSIMD_LOG((bi + epsilon) / (mi + epsilon));                                                    \
        }                                                                                                              \
        return d / 2;                                                                                                  \
    }

#ifdef __cplusplus
extern "C" {
#endif

SIMSIMD_MAKE_KL(serial, f32, f32, SIMSIMD_IDENTIFY, 1e-6) // simsimd_serial_f32_kl
SIMSIMD_MAKE_JS(serial, f32, f32, SIMSIMD_IDENTIFY, 1e-6) // simsimd_serial_f32_js

SIMSIMD_MAKE_KL(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16, 1e-3) // simsimd_serial_f16_kl
SIMSIMD_MAKE_JS(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16, 1e-3) // simsimd_serial_f16_js

SIMSIMD_MAKE_KL(accurate, f32, f64, SIMSIMD_IDENTIFY, 1e-6) // simsimd_accurate_f32_kl
SIMSIMD_MAKE_JS(accurate, f32, f64, SIMSIMD_IDENTIFY, 1e-6) // simsimd_accurate_f32_js

SIMSIMD_MAKE_KL(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16, 1e-6) // simsimd_accurate_f16_kl
SIMSIMD_MAKE_JS(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16, 1e-6) // simsimd_accurate_f16_js

#if SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_ARM_NEON

/*
 *  @file   arm_neon_f32.h
 *  @brief  Arm NEON implementation of the most common similarity metrics for 32-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: Kullback-Leibler and Jensen–Shannon divergence.
 *  - Uses `f32` for storage and `f32` for accumulation.
 *  - Requires compiler capabilities: +simd.
 */

__attribute__((target("+simd"))) //
inline static float32x4_t
simsimd_neon_f32_log(float32x4_t x) {
    float32x4_t a = vmlaq_f32(vdupq_n_f32(-2.29561495781f), vdupq_n_f32(5.17591238022f), x);
    float32x4_t b = vmlaq_f32(vdupq_n_f32(-5.68692588806f), vdupq_n_f32(0.844007015228f), x);
    float32x4_t c = vmlaq_f32(vdupq_n_f32(-2.47071170807f), vdupq_n_f32(4.58445882797f), x);
    float32x4_t d = vmlaq_f32(vdupq_n_f32(-0.165253549814f), vdupq_n_f32(0.0141278216615f), x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    return vmlaq_f32(vmlaq_f32(a, b, x2), vmlaq_f32(c, d, x2), x4);
}

__attribute__((target("+simd"))) //
inline static simsimd_f32_t
simsimd_neon_f32_kl(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = 1e-6;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t ratio_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(b_vec, epsilon_vec));
        float32x4_t log_ratio_vec = simsimd_neon_f32_log(ratio_vec);
        float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
        sum_vec = vaddq_f32(sum_vec, prod_vec);
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i)
        sum += a[i] * SIMSIMD_LOG((a[i] + epsilon) / (b[i] + epsilon));
    return sum;
}

__attribute__((target("+simd"))) inline static simsimd_f32_t
simsimd_neon_f32_js(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = 1e-6;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t m_vec = vaddq_f32(a_vec, b_vec); // M = P + Q
        float32x4_t ratio_a_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t ratio_b_vec = vdivq_f32(vaddq_f32(b_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t log_ratio_a_vec = simsimd_neon_f32_log(ratio_a_vec);
        float32x4_t log_ratio_b_vec = simsimd_neon_f32_log(ratio_b_vec);
        float32x4_t prod_a_vec = vmulq_f32(a_vec, log_ratio_a_vec);
        float32x4_t prod_b_vec = vmulq_f32(b_vec, log_ratio_b_vec);
        sum_vec = vaddq_f32(sum_vec, vaddq_f32(prod_a_vec, prod_b_vec));
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        simsimd_f32_t mi = a[i] + b[i];
        sum += a[i] * SIMSIMD_LOG((a[i] + epsilon) / (mi + epsilon));
        sum += b[i] * SIMSIMD_LOG((b[i] + epsilon) / (mi + epsilon));
    }
    return sum / 2;
}

/*
 *  @file   arm_neon_f16.h
 *  @brief  Arm NEON implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: Kullback-Leibler and Jensen–Shannon divergence.
 *  - Uses `f16` for storage and `f32` for accumulation, as the 16-bit FMA may not always be available.
 *  - Requires compiler capabilities: +simd+fp16.
 */

__attribute__((target("+simd+fp16"))) //
inline static simsimd_f32_t
simsimd_neon_f16_kl(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = 1e-3;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((float16_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((float16_t const*)b + i));
        float32x4_t ratio_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(b_vec, epsilon_vec));
        float32x4_t log_ratio_vec = simsimd_neon_f32_log(ratio_vec);
        float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
        sum_vec = vaddq_f32(sum_vec, prod_vec);
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i)
        sum += SIMSIMD_UNCOMPRESS_F16(a[i]) *
               SIMSIMD_LOG((SIMSIMD_UNCOMPRESS_F16(a[i]) + epsilon) / (SIMSIMD_UNCOMPRESS_F16(b[i]) + epsilon));
    return sum;
}

__attribute__((target("+simd+fp16"))) //
inline static simsimd_f32_t
simsimd_neon_f16_js(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = 1e-3;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((float16_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((float16_t const*)b + i));
        float32x4_t m_vec = vaddq_f32(a_vec, b_vec); // M = P + Q
        float32x4_t ratio_a_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t ratio_b_vec = vdivq_f32(vaddq_f32(b_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t log_ratio_a_vec = simsimd_neon_f32_log(ratio_a_vec);
        float32x4_t log_ratio_b_vec = simsimd_neon_f32_log(ratio_b_vec);
        float32x4_t prod_a_vec = vmulq_f32(a_vec, log_ratio_a_vec);
        float32x4_t prod_b_vec = vmulq_f32(b_vec, log_ratio_b_vec);
        sum_vec = vaddq_f32(sum_vec, vaddq_f32(prod_a_vec, prod_b_vec));
    }
    simsimd_f32_t sum = vaddvq_f32(sum_vec);
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_UNCOMPRESS_F16(a[i]);
        simsimd_f32_t bi = SIMSIMD_UNCOMPRESS_F16(b[i]);
        simsimd_f32_t mi = ai + bi;
        sum += a[i] * SIMSIMD_LOG((a[i] + epsilon) / (mi + epsilon));
        sum += b[i] * SIMSIMD_LOG((b[i] + epsilon) / (mi + epsilon));
    }
    return sum / 2;
}

#endif // SIMSIMD_TARGET_ARM_NEON
#endif // SIMSIMD_TARGET_ARM

#if SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_X86_AVX2

/*
 *  @file   x86_avx2_f16.h
 *  @brief  x86 AVX2 implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - As AVX2 doesn't support masked loads of 16-bit words, implementations have a separate `for`-loop for tails.
 *  - Uses `f16` for both storage and `f32` for accumulation.
 *  - Requires compiler capabilities: avx2, f16c, fma.
 */

__attribute__((target("avx2,fma"))) //
inline static __m256
simsimd_avx2_f32_log(__m256 x) {
    __m256 a = _mm256_fmadd_ps(_mm256_set1_ps(5.17591238022f), x, _mm256_set1_ps(-2.29561495781f));
    __m256 b = _mm256_fmadd_ps(_mm256_set1_ps(0.844007015228f), x, _mm256_set1_ps(-5.68692588806f));
    __m256 c = _mm256_fmadd_ps(_mm256_set1_ps(4.58445882797f), x, _mm256_set1_ps(-2.47071170807f));
    __m256 d = _mm256_fmadd_ps(_mm256_set1_ps(0.0141278216615f), x, _mm256_set1_ps(-0.165253549814f));
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    return _mm256_add_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(c, d, x2), b, x2), _mm256_mul_ps(a, x4));
}

__attribute__((target("avx2,f16c,fma"))) //
inline static simsimd_f32_t
simsimd_avx2_f16_kl(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 sum_vec = _mm256_set1_ps(0);
    simsimd_f32_t epsilon = 1e-3;
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        __m256 ratio_vec = _mm256_div_ps(_mm256_add_ps(a_vec, epsilon_vec), _mm256_add_ps(b_vec, epsilon_vec));
        __m256 log_ratio_vec = simsimd_avx2_f32_log(ratio_vec);
        __m256 prod_vec = _mm256_mul_ps(a_vec, log_ratio_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_vec);
    }

    sum_vec = _mm256_add_ps(_mm256_permute2f128_ps(sum_vec, sum_vec, 1), sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

    simsimd_f32_t sum;
    _mm_store_ss(&sum, _mm256_castps256_ps128(sum_vec));

    // Accumulate the tail:
    for (; i < n; ++i)
        sum += SIMSIMD_UNCOMPRESS_F16(a[i]) *
               SIMSIMD_LOG((SIMSIMD_UNCOMPRESS_F16(a[i]) + epsilon) / (SIMSIMD_UNCOMPRESS_F16(b[i]) + epsilon));
    return sum;
}

__attribute__((target("avx2,f16c,fma"))) //
inline static simsimd_f32_t
simsimd_avx2_f16_js(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 sum_vec = _mm256_set1_ps(0);
    simsimd_f32_t epsilon = 1e-3;
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        __m256 m_vec = _mm256_add_ps(a_vec, b_vec); // M = P + Q
        __m256 ratio_a_vec = _mm256_div_ps(_mm256_add_ps(a_vec, epsilon_vec), _mm256_add_ps(m_vec, epsilon_vec));
        __m256 ratio_b_vec = _mm256_div_ps(_mm256_add_ps(b_vec, epsilon_vec), _mm256_add_ps(m_vec, epsilon_vec));
        __m256 log_ratio_a_vec = simsimd_avx2_f32_log(ratio_a_vec);
        __m256 log_ratio_b_vec = simsimd_avx2_f32_log(ratio_b_vec);
        __m256 prod_a_vec = _mm256_mul_ps(a_vec, log_ratio_a_vec);
        __m256 prod_b_vec = _mm256_mul_ps(b_vec, log_ratio_b_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_a_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_b_vec);
    }

    sum_vec = _mm256_add_ps(_mm256_permute2f128_ps(sum_vec, sum_vec, 1), sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

    simsimd_f32_t sum;
    _mm_store_ss(&sum, _mm256_castps256_ps128(sum_vec));

    // Accumulate the tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_UNCOMPRESS_F16(a[i]);
        simsimd_f32_t bi = SIMSIMD_UNCOMPRESS_F16(b[i]);
        simsimd_f32_t mi = ai + bi;
        sum += a[i] * SIMSIMD_LOG((a[i] + epsilon) / (mi + epsilon));
        sum += b[i] * SIMSIMD_LOG((b[i] + epsilon) / (mi + epsilon));
    }
    return sum / 2;
}

#endif // SIMSIMD_TARGET_X86_AVX2

#if SIMSIMD_TARGET_X86_AVX512 && 0

/*
 *  @file   x86_avx512_f32.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 32-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - Uses `f32` for storage and `f32` for accumulation.
 *  - Requires compiler capabilities: avx512f, avx512vl.
 */

__attribute__((target("avx512f,avx512vl"))) //
inline static simsimd_f32_t
simsimd_avx512_f32_kl(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    __m512 d2_vec = _mm512_set1_ps(0);
    for (simsimd_size_t i = 0; i < n; i += 16) {
        __mmask16 mask = n - i >= 16 ? 0xFFFF : ((1u << (n - i)) - 1u);
        __m512 a_vec = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, b + i);
        __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
        d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    }
    return _mm512_reduce_add_ps(d2_vec);
}

__attribute__((target("avx512f,avx512vl"))) //
inline static simsimd_f32_t
simsimd_avx512_f32_js(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    __m512 ab_vec = _mm512_set1_ps(0);
    for (simsimd_size_t i = 0; i < n; i += 16) {
        __mmask16 mask = n - i >= 16 ? 0xFFFF : ((1u << (n - i)) - 1u);
        __m512 a_vec = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, b + i);
        ab_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_vec);
    }
    return 1 - _mm512_reduce_add_ps(ab_vec);
}

__attribute__((target("avx512f,avx512vl"))) //
inline static simsimd_f32_t
simsimd_avx512_f32_cos(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    __m512 ab_vec = _mm512_set1_ps(0);
    __m512 a2_vec = _mm512_set1_ps(0);
    __m512 b2_vec = _mm512_set1_ps(0);

    for (simsimd_size_t i = 0; i < n; i += 16) {
        __mmask16 mask = n - i >= 16 ? 0xFFFF : ((1u << (n - i)) - 1u);
        __m512 a_vec = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, b + i);
        ab_vec = _mm512_fmadd_ps(a_vec, b_vec, ab_vec);
        a2_vec = _mm512_fmadd_ps(a_vec, a_vec, a2_vec);
        b2_vec = _mm512_fmadd_ps(b_vec, b_vec, b2_vec);
    }

    simsimd_f32_t ab = _mm512_reduce_add_ps(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ps(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ps(b2_vec);

    __m128d a2_b2 = _mm_set_pd((double)a2, (double)b2);
    __m128d rsqrts = _mm_mask_rsqrt14_pd(_mm_setzero_pd(), 0xFF, a2_b2);
    double rsqrts_array[2];
    _mm_storeu_pd(rsqrts_array, rsqrts);
    return ab != 0 ? 1 - ab * rsqrts_array[0] * rsqrts_array[1] : 1;
}

/*
 *  @file   x86_avx512_f16.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, inner product, cosine similarity.
 *  - Uses `_mm512_maskz_loadu_epi16` intrinsics to perform masked unaligned loads.
 *  - Uses `f16` for both storage and accumulation, assuming it's resolution is enough for average case.
 *  - Requires compiler capabilities: avx512fp16, avx512f, avx512vl.
 */

__attribute__((target("avx512fp16,avx512vl,avx512f"))) //
inline static simsimd_f32_t
simsimd_avx512_f16_kl(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m512h d2_vec = _mm512_set1_ph(0);
    for (simsimd_size_t i = 0; i < n; i += 32) {
        __mmask32 mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        __m512h d_vec = _mm512_sub_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec));
        d2_vec = _mm512_fmadd_ph(d_vec, d_vec, d2_vec);
    }
    return _mm512_reduce_add_ph(d2_vec);
}

__attribute__((target("avx512fp16,avx512vl,avx512f"))) //
inline static simsimd_f32_t
simsimd_avx512_f16_js(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m512h ab_vec = _mm512_set1_ph(0);
    simsimd_size_t i = 0;

    for (simsimd_size_t i = 0; i < n; i += 32) {
        __mmask32 mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec), ab_vec);
    }
    return 1 - _mm512_reduce_add_ph(ab_vec);
}

__attribute__((target("avx512fp16,avx512vl,avx512f"))) //
inline static simsimd_f32_t
simsimd_avx512_f16_cos(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m512h ab_vec = _mm512_set1_ph(0);
    __m512h a2_vec = _mm512_set1_ph(0);
    __m512h b2_vec = _mm512_set1_ph(0);

    for (simsimd_size_t i = 0; i < n; i += 32) {
        __mmask32 mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        __m512i a_vec = _mm512_maskz_loadu_epi16(mask, a + i);
        __m512i b_vec = _mm512_maskz_loadu_epi16(mask, b + i);
        ab_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(b_vec), ab_vec);
        a2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(a_vec), _mm512_castsi512_ph(a_vec), a2_vec);
        b2_vec = _mm512_fmadd_ph(_mm512_castsi512_ph(b_vec), _mm512_castsi512_ph(b_vec), b2_vec);
    }

    simsimd_f32_t ab = _mm512_reduce_add_ph(ab_vec);
    simsimd_f32_t a2 = _mm512_reduce_add_ph(a2_vec);
    simsimd_f32_t b2 = _mm512_reduce_add_ph(b2_vec);

    __m128d a2_b2 = _mm_set_pd((double)a2, (double)b2);
    __m128d rsqrts = _mm_mask_rsqrt14_pd(_mm_setzero_pd(), 0xFF, a2_b2);
    double rsqrts_array[2];
    _mm_storeu_pd(rsqrts_array, rsqrts);
    return ab != 0 ? 1 - ab * rsqrts_array[0] * rsqrts_array[1] : 1;
}

/*
 *  @file   x86_avx512_i8.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 8-bit integers.
 *  @author Ash Vardanian
 *
 *  - Implements: L2 squared, cosine similarity, inner product (same as cosine).
 *  - Uses `_mm512_maskz_loadu_epi16` intrinsics to perform masked unaligned loads.
 *  - Uses `i8` for storage, `i16` for multjslication, and `i32` for accumulation, if no better option is available.
 *  - Requires compiler capabilities: avx512f, avx512vl, avx512bw.
 */

__attribute__((target("avx512vl,avx512f,avx512bw"))) //
inline static simsimd_f32_t
simsimd_avx512_i8_kl(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n) {
    __m512i d2_i32s_vec = _mm512_setzero_si512();

    for (simsimd_size_t i = 0; i < n; i += 32) {
        __mmask32 mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        __m512i a_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a + i)); // Load 8-bit integers
        __m512i b_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b + i)); // Load 8-bit integers
        __m512i d_i16s_vec = _mm512_sub_epi16(a_vec, b_vec);
        d2_i32s_vec = _mm512_add_epi32(d2_i32s_vec, _mm512_madd_epi16(d_i16s_vec, d_i16s_vec));
    }

    return _mm512_reduce_add_epi32(d2_i32s_vec);
}

__attribute__((target("avx512vl,avx512f,avx512bw"))) //
inline static simsimd_f32_t
simsimd_avx512_i8_cos(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n) {
    __m512i ab_i32s_vec = _mm512_setzero_si512();
    __m512i a2_i32s_vec = _mm512_setzero_si512();
    __m512i b2_i32s_vec = _mm512_setzero_si512();
    simsimd_size_t i = 0;

    for (simsimd_size_t i = 0; i < n; i += 32) {
        __mmask32 mask = n - i >= 32 ? 0xFFFFFFFF : ((1u << (n - i)) - 1u);
        __m512i a_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a + i)); // Load 8-bit integers
        __m512i b_vec = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b + i)); // Load 8-bit integers

        ab_i32s_vec = _mm512_add_epi32(ab_i32s_vec, _mm512_madd_epi16(a_vec, b_vec));
        a2_i32s_vec = _mm512_add_epi32(a2_i32s_vec, _mm512_madd_epi16(a_vec, a_vec));
        b2_i32s_vec = _mm512_add_epi32(b2_i32s_vec, _mm512_madd_epi16(b_vec, b_vec));
    }

    int ab = _mm512_reduce_add_epi32(ab_i32s_vec);
    int a2 = _mm512_reduce_add_epi32(a2_i32s_vec);
    int b2 = _mm512_reduce_add_epi32(b2_i32s_vec);

    __m128d a2_b2 = _mm_set_pd((double)a2, (double)b2);
    __m128d rsqrts = _mm_mask_rsqrt14_pd(_mm_setzero_pd(), 0xFF, a2_b2);
    double rsqrts_array[2];
    _mm_storeu_pd(rsqrts_array, rsqrts);
    return ab != 0 ? 1 - ab * rsqrts_array[0] * rsqrts_array[1] : 1;
}

__attribute__((target("avx512vl,avx512f,avx512bw"))) //
inline static simsimd_f32_t
simsimd_avx512_i8_js(simsimd_i8_t const* a, simsimd_i8_t const* b, simsimd_size_t n) {
    return simsimd_avx512_i8_cos(a, b, n);
}

#endif // SIMSIMD_TARGET_X86_AVX512
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif