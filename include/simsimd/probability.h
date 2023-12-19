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
#ifndef SIMSIMD_PROBABILITY_H
#define SIMSIMD_PROBABILITY_H

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
        return (simsimd_f32_t)d;                                                                                       \
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
        return (simsimd_f32_t)d / 2;                                                                                   \
    }

#ifdef __cplusplus
extern "C" {
#endif

SIMSIMD_MAKE_KL(serial, f64, f64, SIMSIMD_IDENTIFY, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f64_kl
SIMSIMD_MAKE_JS(serial, f64, f64, SIMSIMD_IDENTIFY, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f64_js

SIMSIMD_MAKE_KL(serial, f32, f32, SIMSIMD_IDENTIFY, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f32_kl
SIMSIMD_MAKE_JS(serial, f32, f32, SIMSIMD_IDENTIFY, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f32_js

SIMSIMD_MAKE_KL(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f16_kl
SIMSIMD_MAKE_JS(serial, f16, f32, SIMSIMD_UNCOMPRESS_F16, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_serial_f16_js

SIMSIMD_MAKE_KL(accurate, f32, f64, SIMSIMD_IDENTIFY, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f32_kl
SIMSIMD_MAKE_JS(accurate, f32, f64, SIMSIMD_IDENTIFY, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f32_js

SIMSIMD_MAKE_KL(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f16_kl
SIMSIMD_MAKE_JS(accurate, f16, f64, SIMSIMD_UNCOMPRESS_F16, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_accurate_f16_js

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
inline float32x4_t
simsimd_neon_f32_log2(float32x4_t x) {
    // Extracting the exponent
    int32x4_t i = vreinterpretq_s32_f32(x);
    int32x4_t e = vsubq_s32(vshrq_n_s32(vandq_s32(i, vdupq_n_s32(0x7F800000)), 23), vdupq_n_s32(127));
    float32x4_t e_float = vcvtq_f32_s32(e);

    // Extracting the mantissa
    float32x4_t m = vreinterpretq_f32_s32(vorrq_s32(vandq_s32(i, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000)));

    // Constants for polynomial
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t p = vdupq_n_f32(-3.4436006e-2f);

    // Compute polynomial using Horner's method
    p = vmlaq_f32(vdupq_n_f32(3.1821337e-1f), m, p);
    p = vmlaq_f32(vdupq_n_f32(-1.2315303f), m, p);
    p = vmlaq_f32(vdupq_n_f32(2.5988452f), m, p);
    p = vmlaq_f32(vdupq_n_f32(-3.3241990f), m, p);
    p = vmlaq_f32(vdupq_n_f32(3.1157899f), m, p);

    // Final computation
    float32x4_t result = vaddq_f32(vmulq_f32(p, vsubq_f32(m, one)), e_float);
    return result;
}

__attribute__((target("+simd"))) //
inline static simsimd_f32_t
simsimd_neon_f32_kl(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t ratio_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(b_vec, epsilon_vec));
        float32x4_t log_ratio_vec = simsimd_neon_f32_log2(ratio_vec);
        float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
        sum_vec = vaddq_f32(sum_vec, prod_vec);
    }
    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
    for (; i < n; ++i)
        sum += a[i] * SIMSIMD_LOG((a[i] + epsilon) / (b[i] + epsilon));
    return sum;
}

__attribute__((target("+simd"))) inline static simsimd_f32_t
simsimd_neon_f32_js(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t m_vec = vmulq_f32(vaddq_f32(a_vec, b_vec), vdupq_n_f32(0.5));
        float32x4_t ratio_a_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t ratio_b_vec = vdivq_f32(vaddq_f32(b_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t log_ratio_a_vec = simsimd_neon_f32_log2(ratio_a_vec);
        float32x4_t log_ratio_b_vec = simsimd_neon_f32_log2(ratio_b_vec);
        float32x4_t prod_a_vec = vmulq_f32(a_vec, log_ratio_a_vec);
        float32x4_t prod_b_vec = vmulq_f32(b_vec, log_ratio_b_vec);
        sum_vec = vaddq_f32(sum_vec, vaddq_f32(prod_a_vec, prod_b_vec));
    }
    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
    for (; i < n; ++i) {
        simsimd_f32_t mi = 0.5f * (a[i] + b[i]);
        sum += a[i] * SIMSIMD_LOG((a[i] + epsilon) / (mi + epsilon));
        sum += b[i] * SIMSIMD_LOG((b[i] + epsilon) / (mi + epsilon));
    }
    return sum;
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
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((float16_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((float16_t const*)b + i));
        float32x4_t ratio_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(b_vec, epsilon_vec));
        float32x4_t log_ratio_vec = simsimd_neon_f32_log2(ratio_vec);
        float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
        sum_vec = vaddq_f32(sum_vec, prod_vec);
    }
    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
    for (; i < n; ++i)
        sum += SIMSIMD_UNCOMPRESS_F16(a[i]) *
               SIMSIMD_LOG((SIMSIMD_UNCOMPRESS_F16(a[i]) + epsilon) / (SIMSIMD_UNCOMPRESS_F16(b[i]) + epsilon));
    return sum;
}

__attribute__((target("+simd+fp16"))) //
inline static simsimd_f32_t
simsimd_neon_f16_js(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_f16(vld1_f16((float16_t const*)a + i));
        float32x4_t b_vec = vcvt_f32_f16(vld1_f16((float16_t const*)b + i));
        float32x4_t m_vec = vmulq_f32(vaddq_f32(a_vec, b_vec), vdupq_n_f32(0.5));
        float32x4_t ratio_a_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t ratio_b_vec = vdivq_f32(vaddq_f32(b_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
        float32x4_t log_ratio_a_vec = simsimd_neon_f32_log2(ratio_a_vec);
        float32x4_t log_ratio_b_vec = simsimd_neon_f32_log2(ratio_b_vec);
        float32x4_t prod_a_vec = vmulq_f32(a_vec, log_ratio_a_vec);
        float32x4_t prod_b_vec = vmulq_f32(b_vec, log_ratio_b_vec);
        sum_vec = vaddq_f32(sum_vec, vaddq_f32(prod_a_vec, prod_b_vec));
    }
    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_UNCOMPRESS_F16(a[i]);
        simsimd_f32_t bi = SIMSIMD_UNCOMPRESS_F16(b[i]);
        simsimd_f32_t mi = 0.5f * (ai + bi);
        sum += ai * SIMSIMD_LOG((ai + epsilon) / (mi + epsilon));
        sum += bi * SIMSIMD_LOG((bi + epsilon) / (mi + epsilon));
    }
    return sum;
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
 *  - Implements: Kullback-Leibler and Jensen–Shannon divergence.
 *  - As AVX2 doesn't support masked loads of 16-bit words, implementations have a separate `for`-loop for tails.
 *  - Uses `f16` for both storage and `f32` for accumulation.
 *  - Requires compiler capabilities: avx2, f16c, fma.
 */

__attribute__((target("avx2,f16c,fma"))) //
inline __m256
simsimd_avx2_f32_log2(__m256 x) {
    // Extracting the exponent
    __m256i i = _mm256_castps_si256(x);
    __m256i e = _mm256_srli_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0x7F800000)), 23);
    e = _mm256_sub_epi32(e, _mm256_set1_epi32(127)); // removing the bias
    __m256 e_float = _mm256_cvtepi32_ps(e);

    // Extracting the mantissa
    __m256 m = _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_and_si256(i, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000)));

    // Constants for polynomial
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 p = _mm256_set1_ps(-3.4436006e-2f);

    // Compute the polynomial using Horner's method
    p = _mm256_fmadd_ps(m, p, _mm256_set1_ps(3.1821337e-1f));
    p = _mm256_fmadd_ps(m, p, _mm256_set1_ps(-1.2315303f));
    p = _mm256_fmadd_ps(m, p, _mm256_set1_ps(2.5988452f));
    p = _mm256_fmadd_ps(m, p, _mm256_set1_ps(-3.3241990f));
    p = _mm256_fmadd_ps(m, p, _mm256_set1_ps(3.1157899f));

    // Final computation
    __m256 result = _mm256_add_ps(_mm256_mul_ps(p, _mm256_sub_ps(m, one)), e_float);
    return result;
}

__attribute__((target("avx2,f16c,fma"))) //
inline static simsimd_f32_t
simsimd_avx2_f16_kl(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 b_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));
        __m256 ratio_vec = _mm256_div_ps(_mm256_add_ps(a_vec, epsilon_vec), _mm256_add_ps(b_vec, epsilon_vec));
        __m256 log_ratio_vec = simsimd_avx2_f32_log2(ratio_vec);
        __m256 prod_vec = _mm256_mul_ps(a_vec, log_ratio_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_vec);
    }

    sum_vec = _mm256_add_ps(_mm256_permute2f128_ps(sum_vec, sum_vec, 1), sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum;
    _mm_store_ss(&sum, _mm256_castps256_ps128(sum_vec));
    sum *= log2_normalizer;

    // Accumulate the tail:
    for (; i < n; ++i)
        sum += SIMSIMD_UNCOMPRESS_F16(a[i]) *
               SIMSIMD_LOG((SIMSIMD_UNCOMPRESS_F16(a[i]) + epsilon) / (SIMSIMD_UNCOMPRESS_F16(b[i]) + epsilon));
    return sum;
}

__attribute__((target("avx2,f16c,fma"))) //
inline static simsimd_f32_t
simsimd_avx2_f16_js(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i))), epsilon_vec);
        __m256 b_vec = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i))), epsilon_vec);
        __m256 m_vec = _mm256_mul_ps(_mm256_add_ps(a_vec, b_vec), _mm256_set1_ps(0.5f)); // M = (P + Q) / 2
        __m256 ratio_a_vec = _mm256_div_ps(a_vec, m_vec);
        __m256 ratio_b_vec = _mm256_div_ps(b_vec, m_vec);
        __m256 log_ratio_a_vec = simsimd_avx2_f32_log2(ratio_a_vec);
        __m256 log_ratio_b_vec = simsimd_avx2_f32_log2(ratio_b_vec);
        __m256 prod_a_vec = _mm256_mul_ps(a_vec, log_ratio_a_vec);
        __m256 prod_b_vec = _mm256_mul_ps(b_vec, log_ratio_b_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_a_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod_b_vec);
    }

    sum_vec = _mm256_add_ps(_mm256_permute2f128_ps(sum_vec, sum_vec, 1), sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum;
    _mm_store_ss(&sum, _mm256_castps256_ps128(sum_vec));
    sum *= log2_normalizer;

    // Accumulate the tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = SIMSIMD_UNCOMPRESS_F16(a[i]);
        simsimd_f32_t bi = SIMSIMD_UNCOMPRESS_F16(b[i]);
        simsimd_f32_t mi = ai + bi;
        sum += ai * SIMSIMD_LOG((ai + epsilon) / (mi + epsilon));
        sum += bi * SIMSIMD_LOG((bi + epsilon) / (mi + epsilon));
    }
    return sum / 2;
}

#endif // SIMSIMD_TARGET_X86_AVX2

#if SIMSIMD_TARGET_X86_AVX512

/*
 *  @file   x86_avx512_f32.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 32-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: Kullback-Leibler and Jensen–Shannon divergence.
 *  - Uses `f32` for storage and `f32` for accumulation.
 *  - Requires compiler capabilities: avx512f, avx512vl.
 */

__attribute__((target("avx512f,avx512vl"))) //
inline __m512
simsimd_avx512_f32_log2(__m512 x) {
    // Extract the exponent and mantissa
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 e = _mm512_getexp_ps(x);
    __m512 m = _mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512 p = _mm512_set1_ps(-3.4436006e-2f);
    p = _mm512_fmadd_ps(m, p, _mm512_set1_ps(3.1821337e-1f));
    p = _mm512_fmadd_ps(m, p, _mm512_set1_ps(-1.2315303f));
    p = _mm512_fmadd_ps(m, p, _mm512_set1_ps(2.5988452f));
    p = _mm512_fmadd_ps(m, p, _mm512_set1_ps(-3.3241990f));
    p = _mm512_fmadd_ps(m, p, _mm512_set1_ps(3.1157899f));

    return _mm512_add_ps(_mm512_mul_ps(p, _mm512_sub_ps(m, one)), e);
}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static simsimd_f32_t
simsimd_avx512_f32_kl(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    __m512 sum_vec = _mm512_setzero();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m512 epsilon_vec = _mm512_set1_ps(epsilon);
    __m512 a_vec, b_vec;

simsimd_avx512_f32_kl_cycle:
    if (n < 16) {
        __mmask16 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, a), epsilon_vec);
        b_vec = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, b), epsilon_vec);
        n = 0;
    } else {
        a_vec = _mm512_add_ps(_mm512_loadu_ps(a), epsilon_vec);
        b_vec = _mm512_add_ps(_mm512_loadu_ps(b), epsilon_vec);
        a += 16, b += 16, n -= 16;
    }
    __m512 ratio_vec = _mm512_div_ps(a_vec, b_vec);
    __m512 log_ratio_vec = simsimd_avx512_f32_log2(ratio_vec);
    __m512 prod_vec = _mm512_mul_ps(a_vec, log_ratio_vec);
    sum_vec = _mm512_add_ps(sum_vec, prod_vec);
    if (n)
        goto simsimd_avx512_f32_kl_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    return _mm512_reduce_add_ps(sum_vec) * log2_normalizer;
}

__attribute__((target("avx512f,avx512vl,bmi2"))) //
inline static simsimd_f32_t
simsimd_avx512_f32_js(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n) {
    __m512 sum_a_vec = _mm512_setzero();
    __m512 sum_b_vec = _mm512_setzero();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m512 epsilon_vec = _mm512_set1_ps(epsilon);
    __m512 a_vec, b_vec;

simsimd_avx512_f32_js_cycle:
    if (n < 16) {
        __mmask16 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    } else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    __m512 m_vec = _mm512_mul_ps(_mm512_add_ps(a_vec, b_vec), _mm512_set1_ps(0.5f));
    __mmask16 nonzero_mask_a = _mm512_cmp_ps_mask(a_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask16 nonzero_mask_b = _mm512_cmp_ps_mask(b_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask16 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512 m_recip_approx = _mm512_rcp14_ps(m_vec);
    __m512 ratio_a_vec = _mm512_mul_ps(a_vec, m_recip_approx);
    __m512 ratio_b_vec = _mm512_mul_ps(b_vec, m_recip_approx);
    __m512 log_ratio_a_vec = simsimd_avx512_f32_log2(ratio_a_vec);
    __m512 log_ratio_b_vec = simsimd_avx512_f32_log2(ratio_b_vec);
    sum_a_vec = _mm512_maskz_fmadd_ps(nonzero_mask, a_vec, log_ratio_a_vec, sum_a_vec);
    sum_b_vec = _mm512_maskz_fmadd_ps(nonzero_mask, b_vec, log_ratio_b_vec, sum_b_vec);
    if (n)
        goto simsimd_avx512_f32_js_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    return _mm512_reduce_add_ps(_mm512_add_ps(sum_a_vec, sum_b_vec)) * 0.5f * log2_normalizer;
}

/*
 *  @file   x86_avx512_f16.h
 *  @brief  x86 AVX-512 implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements: Kullback-Leibler and Jensen–Shannon divergence.
 *  - Uses `_mm512_maskz_loadu_epi16` intrinsics to perform masked unaligned loads.
 *  - Uses `f16` for both storage and accumulation, assuming it's resolution is enough for average case.
 *  - Requires compiler capabilities: avx512fp16, avx512f, avx512vl.
 */

__attribute__((target("avx512f,avx512vl,avx512fp16"))) //
inline __m512h
simsimd_avx512_f16_log2(__m512h x) {
    // Extract the exponent and mantissa
    __m512h one = _mm512_set1_ph((_Float16)1);
    __m512h e = _mm512_getexp_ph(x);
    __m512h m = _mm512_getmant_ph(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512h p = _mm512_set1_ph((_Float16)-3.4436006e-2f);
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((_Float16)3.1821337e-1f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((_Float16)-1.2315303f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((_Float16)2.5988452f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((_Float16)-3.3241990f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((_Float16)3.1157899f));

    return _mm512_add_ph(_mm512_mul_ph(p, _mm512_sub_ph(m, one)), e);
}

__attribute__((target("avx512f,avx512vl,avx512fp16,bmi2"))) //
inline static simsimd_f32_t
simsimd_avx512_f16_kl(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m512h sum_vec = _mm512_set1_ph((_Float16)0);
    __m512h epsilon_vec = _mm512_set1_ph((_Float16)SIMSIMD_F16_DIVISION_EPSILON);
    __m512h a_vec, b_vec;

simsimd_avx512_f16_kl_cycle:
    if (n < 32) {
        __mmask32 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a)), epsilon_vec);
        b_vec = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b)), epsilon_vec);
        n = 0;
    } else {
        a_vec = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(a)), epsilon_vec);
        b_vec = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(b)), epsilon_vec);
        a += 32, b += 32, n -= 32;
    }
    __m512h ratio_vec = _mm512_div_ph(a_vec, b_vec);
    __m512h log_ratio_vec = simsimd_avx512_f16_log2(ratio_vec);
    __m512h prod_vec = _mm512_mul_ph(a_vec, log_ratio_vec);
    sum_vec = _mm512_add_ph(sum_vec, prod_vec);
    if (n)
        goto simsimd_avx512_f16_kl_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    return _mm512_reduce_add_ph(sum_vec) * log2_normalizer;
}

__attribute__((target("avx512f,avx512vl,avx512fp16,bmi2"))) //
inline static simsimd_f32_t
simsimd_avx512_f16_js(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n) {
    __m512h sum_a_vec = _mm512_set1_ph((_Float16)0);
    __m512h sum_b_vec = _mm512_set1_ph((_Float16)0);
    __m512h epsilon_vec = _mm512_set1_ph((_Float16)SIMSIMD_F16_DIVISION_EPSILON);
    __m512h a_vec, b_vec;

simsimd_avx512_f16_js_cycle:
    if (n < 32) {
        __mmask32 mask = _bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    } else {
        a_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    __m512h m_vec = _mm512_mul_ph(_mm512_add_ph(a_vec, b_vec), _mm512_set1_ph((_Float16)0.5f));
    __mmask32 nonzero_mask_a = _mm512_cmp_ph_mask(a_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask32 nonzero_mask_b = _mm512_cmp_ph_mask(b_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask32 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512h m_recip_approx = _mm512_rcp_ph(m_vec);
    __m512h ratio_a_vec = _mm512_mul_ph(a_vec, m_recip_approx);
    __m512h ratio_b_vec = _mm512_mul_ph(b_vec, m_recip_approx);
    __m512h log_ratio_a_vec = simsimd_avx512_f16_log2(ratio_a_vec);
    __m512h log_ratio_b_vec = simsimd_avx512_f16_log2(ratio_b_vec);
    sum_a_vec = _mm512_maskz_fmadd_ph(nonzero_mask, a_vec, log_ratio_a_vec, sum_a_vec);
    sum_b_vec = _mm512_maskz_fmadd_ph(nonzero_mask, b_vec, log_ratio_b_vec, sum_b_vec);
    if (n)
        goto simsimd_avx512_f16_js_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    return _mm512_reduce_add_ph(_mm512_add_ph(sum_a_vec, sum_b_vec)) * 0.5f * log2_normalizer;
}

#endif // SIMSIMD_TARGET_X86_AVX512
#endif // SIMSIMD_TARGET_X86

#ifdef __cplusplus
}
#endif

#endif
