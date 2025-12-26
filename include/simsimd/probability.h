/**
 *  @brief SIMD-accelerated Similarity Measures for Probability Distributions.
 *  @file include/simsimd/probability.h
 *  @author Ash Vardanian
 *  @date October 20, 2023
 *
 *  Contains following similarity measures:
 *
 *  - Kullback-Leibler divergence
 *  - Jensen-Shannon divergence
 *
 *  For datatypes:
 *
 *  - 32-bit floating point numbers
 *  - 16-bit floating point numbers
 *  - 16-bit brain-floating point numbers
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire
 *
 *  @section numeric_stability Numeric Stability
 *
 *  The implementations use a small epsilon to avoid division by zero when inputs contain zeros.
 *  For higher-precision accumulation, use the "*_accurate" serial variants.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  KL/JS divergence requires log2(x) which decomposes into exponent extraction (VGETEXP) plus
 *  mantissa polynomial (using VGETMANT + FMA chain). This approach is faster than scalar log()
 *  calls. Division (for p/q ratio) uses either VDIVPS directly or VRCP14PS with Newton-Raphson
 *  refinement when ~14-bit precision suffices. Genoa's VGETEXP/VGETMANT are 25% faster than Ice.
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm512_getexp_ps        VGETEXPPS (ZMM, ZMM)            4c @ p0     3c @ p23
 *      _mm512_getexp_pd        VGETEXPPD (ZMM, ZMM)            4c @ p0     3c @ p23
 *      _mm512_getmant_ps       VGETMANTPS (ZMM, ZMM, I8)       4c @ p0     3c @ p23
 *      _mm512_getmant_pd       VGETMANTPD (ZMM, ZMM, I8)       4c @ p0     3c @ p23
 *      _mm512_rcp14_ps         VRCP14PS (ZMM, ZMM)             7c @ p05    5c @ p01
 *      _mm512_div_ps           VDIVPS (ZMM, ZMM, ZMM)          17c @ p05   11c @ p01
 *      _mm512_fmadd_ps         VFMADD231PS (ZMM, ZMM, ZMM)     4c @ p0     4c @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM lacks direct exponent/mantissa extraction, so log2 uses integer reinterpretation of the
 *  float bits followed by polynomial refinement. FRECPE provides ~8-bit reciprocal approximation
 *  for division, refined with FRECPS Newton-Raphson steps to ~22-bit precision.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vrecpeq_f32             FRECPE.S        3c @ V02        3c @ V02        3c @ V02
 *      vrecpsq_f32             FRECPS.S        4c @ V0123      4c @ V0123      4c @ V0123
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef SIMSIMD_PROBABILITY_H
#define SIMSIMD_PROBABILITY_H

#include "types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_kld_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_kld_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result);
/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_kld_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_kld_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/**
 *  @brief Jensen-Shannon divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_jsd_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/**
 *  @brief Jensen-Shannon divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_jsd_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                      simsimd_distance_t *result);
/**
 *  @brief Jensen-Shannon divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_jsd_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);
/**
 *  @brief Jensen-Shannon divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
SIMSIMD_DYNAMIC void simsimd_jsd_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result);

// clang-format off

/** @copydoc simsimd_kld_f64 */
SIMSIMD_PUBLIC void simsimd_kld_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f64 */
SIMSIMD_PUBLIC void simsimd_jsd_f64_serial(simsimd_f64_t const* a, simsimd_f64_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_kld_f32 */
SIMSIMD_PUBLIC void simsimd_kld_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f32 */
SIMSIMD_PUBLIC void simsimd_jsd_f32_serial(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_kld_f16 */
SIMSIMD_PUBLIC void simsimd_kld_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f16 */
SIMSIMD_PUBLIC void simsimd_jsd_f16_serial(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_kld_bf16 */
SIMSIMD_PUBLIC void simsimd_kld_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_bf16 */
SIMSIMD_PUBLIC void simsimd_jsd_bf16_serial(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);

/** @copydoc simsimd_kld_f32 */
SIMSIMD_PUBLIC void simsimd_kld_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f32 */
SIMSIMD_PUBLIC void simsimd_jsd_f32_accurate(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_kld_f16 */
SIMSIMD_PUBLIC void simsimd_kld_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f16 */
SIMSIMD_PUBLIC void simsimd_jsd_f16_accurate(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_kld_bf16 */
SIMSIMD_PUBLIC void simsimd_kld_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_bf16 */
SIMSIMD_PUBLIC void simsimd_jsd_bf16_accurate(simsimd_bf16_t const* a, simsimd_bf16_t const* b, simsimd_size_t n, simsimd_distance_t* result);

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_kld_f32 */
SIMSIMD_PUBLIC void simsimd_kld_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f32 */
SIMSIMD_PUBLIC void simsimd_jsd_f32_neon(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
/** @copydoc simsimd_kld_f16 */
SIMSIMD_PUBLIC void simsimd_kld_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f16 */
SIMSIMD_PUBLIC void simsimd_jsd_f16_neon(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_NEON_F16

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_kld_f16 */
SIMSIMD_PUBLIC void simsimd_kld_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f16 */
SIMSIMD_PUBLIC void simsimd_jsd_f16_haswell(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_kld_f32 */
SIMSIMD_PUBLIC void simsimd_kld_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f32 */
SIMSIMD_PUBLIC void simsimd_jsd_f32_skylake(simsimd_f32_t const* a, simsimd_f32_t const* b, simsimd_size_t n, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_SAPPHIRE
/** @copydoc simsimd_kld_f16 */
SIMSIMD_PUBLIC void simsimd_kld_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
/** @copydoc simsimd_jsd_f16 */
SIMSIMD_PUBLIC void simsimd_jsd_f16_sapphire(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_size_t n, simsimd_distance_t* result);
#endif // SIMSIMD_TARGET_SAPPHIRE
// clang-format on

#define SIMSIMD_MAKE_KLD(name, input_type, accumulator_type, load_and_convert, epsilon)                        \
    SIMSIMD_PUBLIC void simsimd_kld_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                          simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                          simsimd_distance_t *result) {                        \
        simsimd_##accumulator_type##_t d = 0, ai, bi;                                                          \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                              \
            load_and_convert(a + i, &ai);                                                                      \
            load_and_convert(b + i, &bi);                                                                      \
            d += ai * SIMSIMD_LOG((ai + epsilon) / (bi + epsilon));                                            \
        }                                                                                                      \
        *result = (simsimd_distance_t)d;                                                                       \
    }

#define SIMSIMD_MAKE_JSD(name, input_type, accumulator_type, load_and_convert, epsilon)                        \
    SIMSIMD_PUBLIC void simsimd_jsd_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                          simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                          simsimd_distance_t *result) {                        \
        simsimd_##accumulator_type##_t d = 0, ai, bi;                                                          \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                              \
            load_and_convert(a + i, &ai);                                                                      \
            load_and_convert(b + i, &bi);                                                                      \
            simsimd_##accumulator_type##_t mi = (ai + bi) / 2;                                                 \
            d += ai * SIMSIMD_LOG((ai + epsilon) / (mi + epsilon));                                            \
            d += bi * SIMSIMD_LOG((bi + epsilon) / (mi + epsilon));                                            \
        }                                                                                                      \
        simsimd_distance_t d_half = ((simsimd_distance_t)d / 2);                                               \
        *result = d_half > 0 ? SIMSIMD_SQRT(d_half) : 0;                                                       \
    }

SIMSIMD_MAKE_KLD(serial, f64, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_f64_serial
SIMSIMD_MAKE_JSD(serial, f64, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_f64_serial

SIMSIMD_MAKE_KLD(serial, f32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_f32_serial
SIMSIMD_MAKE_JSD(serial, f32, f32, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_f32_serial

SIMSIMD_MAKE_KLD(serial, f16, f32, simsimd_f16_to_f32, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_f16_serial
SIMSIMD_MAKE_JSD(serial, f16, f32, simsimd_f16_to_f32, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_f16_serial

SIMSIMD_MAKE_KLD(serial, bf16, f32, simsimd_bf16_to_f32, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_bf16_serial
SIMSIMD_MAKE_JSD(serial, bf16, f32, simsimd_bf16_to_f32, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_bf16_serial

SIMSIMD_MAKE_KLD(accurate, f32, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_f32_accurate
SIMSIMD_MAKE_JSD(accurate, f32, f64, SIMSIMD_ASSIGN_FROM_TO, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_f32_accurate

SIMSIMD_MAKE_KLD(accurate, f16, f64, simsimd_f16_to_f64, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_f16_accurate
SIMSIMD_MAKE_JSD(accurate, f16, f64, simsimd_f16_to_f64, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_f16_accurate

SIMSIMD_MAKE_KLD(accurate, bf16, f64, simsimd_bf16_to_f64, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_kld_bf16_accurate
SIMSIMD_MAKE_JSD(accurate, bf16, f64, simsimd_bf16_to_f64, SIMSIMD_F32_DIVISION_EPSILON) // simsimd_jsd_bf16_accurate

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SIMSIMD_PUBLIC float32x4_t _simsimd_log2_f32_neon(float32x4_t x) {
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

SIMSIMD_PUBLIC void simsimd_kld_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    float32x4_t sum_vec = vdupq_n_f32(0);
    float32x4_t a_vec, b_vec;

simsimd_kld_f32_neon_cycle:
    if (n < 4) {
        a_vec = _simsimd_partial_load_f32x4_neon(a, n);
        b_vec = _simsimd_partial_load_f32x4_neon(b, n);
        n = 0;
    }
    else {
        a_vec = vld1q_f32(a);
        b_vec = vld1q_f32(b);
        n -= 4, a += 4, b += 4;
    }

    float32x4_t ratio_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(b_vec, epsilon_vec));
    float32x4_t log_ratio_vec = _simsimd_log2_f32_neon(ratio_vec);
    float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
    sum_vec = vaddq_f32(sum_vec, prod_vec);
    if (n != 0) goto simsimd_kld_f32_neon_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_jsd_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    float32x4_t sum_vec = vdupq_n_f32(0);
    float32x4_t a_vec, b_vec;

simsimd_jsd_f32_neon_cycle:
    if (n < 4) {
        a_vec = _simsimd_partial_load_f32x4_neon(a, n);
        b_vec = _simsimd_partial_load_f32x4_neon(b, n);
        n = 0;
    }
    else {
        a_vec = vld1q_f32(a);
        b_vec = vld1q_f32(b);
        n -= 4, a += 4, b += 4;
    }

    float32x4_t m_vec = vmulq_f32(vaddq_f32(a_vec, b_vec), vdupq_n_f32(0.5));
    float32x4_t ratio_a_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
    float32x4_t ratio_b_vec = vdivq_f32(vaddq_f32(b_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
    float32x4_t log_ratio_a_vec = _simsimd_log2_f32_neon(ratio_a_vec);
    float32x4_t log_ratio_b_vec = _simsimd_log2_f32_neon(ratio_b_vec);
    float32x4_t prod_a_vec = vmulq_f32(a_vec, log_ratio_a_vec);
    float32x4_t prod_b_vec = vmulq_f32(b_vec, log_ratio_b_vec);

    sum_vec = vaddq_f32(sum_vec, vaddq_f32(prod_a_vec, prod_b_vec));
    if (n != 0) goto simsimd_jsd_f32_neon_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer / 2;
    *result = sum > 0 ? _simsimd_sqrt_f32_neon(sum) : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_kld_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    float32x4_t a_vec, b_vec;

simsimd_kld_f16_neon_cycle:
    if (n < 4) {
        a_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a, n));
        b_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b, n));
        n = 0;
    }
    else {
        a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a));
        b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }

    float32x4_t ratio_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(b_vec, epsilon_vec));
    float32x4_t log_ratio_vec = _simsimd_log2_f32_neon(ratio_vec);
    float32x4_t prod_vec = vmulq_f32(a_vec, log_ratio_vec);
    sum_vec = vaddq_f32(sum_vec, prod_vec);
    if (n) goto simsimd_kld_f16_neon_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_jsd_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_distance_t *result) {
    float32x4_t sum_vec = vdupq_n_f32(0);
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    float32x4_t a_vec, b_vec;

simsimd_jsd_f16_neon_cycle:
    if (n < 4) {
        a_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(a, n));
        b_vec = vcvt_f32_f16(_simsimd_partial_load_f16x4_neon(b, n));
        n = 0;
    }
    else {
        a_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)a));
        b_vec = vcvt_f32_f16(vld1_f16((simsimd_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }

    float32x4_t m_vec = vmulq_f32(vaddq_f32(a_vec, b_vec), vdupq_n_f32(0.5));
    float32x4_t ratio_a_vec = vdivq_f32(vaddq_f32(a_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
    float32x4_t ratio_b_vec = vdivq_f32(vaddq_f32(b_vec, epsilon_vec), vaddq_f32(m_vec, epsilon_vec));
    float32x4_t log_ratio_a_vec = _simsimd_log2_f32_neon(ratio_a_vec);
    float32x4_t log_ratio_b_vec = _simsimd_log2_f32_neon(ratio_b_vec);
    float32x4_t prod_a_vec = vmulq_f32(a_vec, log_ratio_a_vec);
    float32x4_t prod_b_vec = vmulq_f32(b_vec, log_ratio_b_vec);
    sum_vec = vaddq_f32(sum_vec, vaddq_f32(prod_a_vec, prod_b_vec));
    if (n) goto simsimd_jsd_f16_neon_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = vaddvq_f32(sum_vec) * log2_normalizer / 2;
    *result = sum > 0 ? _simsimd_sqrt_f32_neon(sum) : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16
#endif // _SIMSIMD_TARGET_ARM

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_INTERNAL __m256 _simsimd_log2_f32_haswell(__m256 x) {
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

SIMSIMD_PUBLIC void simsimd_kld_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    __m256 sum_vec = _mm256_setzero_ps();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    __m256 a_vec, b_vec;

simsimd_kld_f16_haswell_cycle:
    if (n < 8) {
        a_vec = _simsimd_partial_load_f16x8_haswell(a, n);
        b_vec = _simsimd_partial_load_f16x8_haswell(b, n);
        n = 0;
    }
    else {
        a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    a_vec = _mm256_add_ps(a_vec, epsilon_vec);
    b_vec = _mm256_add_ps(b_vec, epsilon_vec);
    __m256 ratio_vec = _mm256_div_ps(a_vec, b_vec);
    __m256 log_ratio_vec = _simsimd_log2_f32_haswell(ratio_vec);
    __m256 prod_vec = _mm256_mul_ps(a_vec, log_ratio_vec);
    sum_vec = _mm256_add_ps(sum_vec, prod_vec);
    if (n) goto simsimd_kld_f16_haswell_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = _simsimd_reduce_f32x8_haswell(sum_vec);
    sum *= log2_normalizer;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_jsd_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 a_vec, b_vec;

simsimd_jsd_f16_haswell_cycle:
    if (n < 8) {
        a_vec = _simsimd_partial_load_f16x8_haswell(a, n);
        b_vec = _simsimd_partial_load_f16x8_haswell(b, n);
        n = 0;
    }
    else {
        a_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)a));
        b_vec = _mm256_cvtph_ps(_mm_lddqu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 m_vec = _mm256_mul_ps(_mm256_add_ps(a_vec, b_vec), _mm256_set1_ps(0.5f)); // M = (P + Q) / 2
    __m256 ratio_a_vec = _mm256_div_ps(_mm256_add_ps(a_vec, epsilon_vec), _mm256_add_ps(m_vec, epsilon_vec));
    __m256 ratio_b_vec = _mm256_div_ps(_mm256_add_ps(b_vec, epsilon_vec), _mm256_add_ps(m_vec, epsilon_vec));
    __m256 log_ratio_a_vec = _simsimd_log2_f32_haswell(ratio_a_vec);
    __m256 log_ratio_b_vec = _simsimd_log2_f32_haswell(ratio_b_vec);
    __m256 prod_a_vec = _mm256_mul_ps(a_vec, log_ratio_a_vec);
    __m256 prod_b_vec = _mm256_mul_ps(b_vec, log_ratio_b_vec);
    sum_vec = _mm256_add_ps(sum_vec, prod_a_vec);
    sum_vec = _mm256_add_ps(sum_vec, prod_b_vec);
    if (n) goto simsimd_jsd_f16_haswell_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = _simsimd_reduce_f32x8_haswell(sum_vec);
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? _simsimd_sqrt_f32_haswell(sum) : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2"))), apply_to = function)

SIMSIMD_INTERNAL __m512 _simsimd_log2_f32_skylake(__m512 x) {
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

SIMSIMD_PUBLIC void simsimd_kld_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    __m512 sum_vec = _mm512_setzero();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m512 epsilon_vec = _mm512_set1_ps(epsilon);
    __m512 a_vec, b_vec;

simsimd_kld_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, a), epsilon_vec);
        b_vec = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, b), epsilon_vec);
        n = 0;
    }
    else {
        a_vec = _mm512_add_ps(_mm512_loadu_ps(a), epsilon_vec);
        b_vec = _mm512_add_ps(_mm512_loadu_ps(b), epsilon_vec);
        a += 16, b += 16, n -= 16;
    }
    __m512 ratio_vec = _mm512_div_ps(a_vec, b_vec);
    __m512 log_ratio_vec = _simsimd_log2_f32_skylake(ratio_vec);
    __m512 prod_vec = _mm512_mul_ps(a_vec, log_ratio_vec);
    sum_vec = _mm512_add_ps(sum_vec, prod_vec);
    if (n) goto simsimd_kld_f32_skylake_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    *result = _mm512_reduce_add_ps(sum_vec) * log2_normalizer;
}

SIMSIMD_PUBLIC void simsimd_jsd_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_distance_t *result) {
    __m512 sum_a_vec = _mm512_setzero();
    __m512 sum_b_vec = _mm512_setzero();
    simsimd_f32_t epsilon = SIMSIMD_F32_DIVISION_EPSILON;
    __m512 epsilon_vec = _mm512_set1_ps(epsilon);
    __m512 a_vec, b_vec;

simsimd_jsd_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    __m512 m_vec = _mm512_mul_ps(_mm512_add_ps(a_vec, b_vec), _mm512_set1_ps(0.5f));
    __mmask16 nonzero_mask_a = _mm512_cmp_ps_mask(a_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask16 nonzero_mask_b = _mm512_cmp_ps_mask(b_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask16 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512 m_with_epsilon = _mm512_add_ps(m_vec, epsilon_vec);
    __m512 m_recip_approx = _mm512_rcp14_ps(m_with_epsilon);
    __m512 ratio_a_vec = _mm512_mul_ps(_mm512_add_ps(a_vec, epsilon_vec), m_recip_approx);
    __m512 ratio_b_vec = _mm512_mul_ps(_mm512_add_ps(b_vec, epsilon_vec), m_recip_approx);
    __m512 log_ratio_a_vec = _simsimd_log2_f32_skylake(ratio_a_vec);
    __m512 log_ratio_b_vec = _simsimd_log2_f32_skylake(ratio_b_vec);
    sum_a_vec = _mm512_mask3_fmadd_ps(a_vec, log_ratio_a_vec, sum_a_vec, nonzero_mask);
    sum_b_vec = _mm512_mask3_fmadd_ps(b_vec, log_ratio_b_vec, sum_b_vec, nonzero_mask);
    if (n) goto simsimd_jsd_f32_skylake_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = _mm512_reduce_add_ps(_mm512_add_ps(sum_a_vec, sum_b_vec));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? _simsimd_sqrt_f32_haswell(sum) : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512fp16"))), apply_to = function)

SIMSIMD_INTERNAL __m512h _simsimd_log2_f16_sapphire(__m512h x) {
    // Extract the exponent and mantissa
    __m512h one = _mm512_set1_ph((simsimd_f16_t)1);
    __m512h e = _mm512_getexp_ph(x);
    __m512h m = _mm512_getmant_ph(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512h p = _mm512_set1_ph((simsimd_f16_t)-3.4436006e-2f);
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((simsimd_f16_t)3.1821337e-1f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((simsimd_f16_t)-1.2315303f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((simsimd_f16_t)2.5988452f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((simsimd_f16_t)-3.3241990f));
    p = _mm512_fmadd_ph(m, p, _mm512_set1_ph((simsimd_f16_t)3.1157899f));

    return _mm512_add_ph(_mm512_mul_ph(p, _mm512_sub_ph(m, one)), e);
}

SIMSIMD_PUBLIC void simsimd_kld_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    __m512h sum_vec = _mm512_setzero_ph();
    __m512h epsilon_vec = _mm512_set1_ph((simsimd_f16_t)SIMSIMD_F16_DIVISION_EPSILON);
    __m512h a_vec, b_vec;

simsimd_kld_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a)), epsilon_vec);
        b_vec = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b)), epsilon_vec);
        n = 0;
    }
    else {
        a_vec = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(a)), epsilon_vec);
        b_vec = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(b)), epsilon_vec);
        a += 32, b += 32, n -= 32;
    }
    __m512h ratio_vec = _mm512_div_ph(a_vec, b_vec);
    __m512h log_ratio_vec = _simsimd_log2_f16_sapphire(ratio_vec);
    __m512h prod_vec = _mm512_mul_ph(a_vec, log_ratio_vec);
    sum_vec = _mm512_add_ph(sum_vec, prod_vec);
    if (n) goto simsimd_kld_f16_sapphire_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    *result = _mm512_reduce_add_ph(sum_vec) * log2_normalizer;
}

SIMSIMD_PUBLIC void simsimd_jsd_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_distance_t *result) {
    __m512h sum_a_vec = _mm512_setzero_ph();
    __m512h sum_b_vec = _mm512_setzero_ph();
    __m512h epsilon_vec = _mm512_set1_ph((simsimd_f16_t)SIMSIMD_F16_DIVISION_EPSILON);
    __m512h a_vec, b_vec;

simsimd_jsd_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_vec = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    __m512h m_vec = _mm512_mul_ph(_mm512_add_ph(a_vec, b_vec), _mm512_set1_ph((simsimd_f16_t)0.5f));
    __mmask32 nonzero_mask_a = _mm512_cmp_ph_mask(a_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask32 nonzero_mask_b = _mm512_cmp_ph_mask(b_vec, epsilon_vec, _CMP_GE_OQ);
    __mmask32 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512h m_with_epsilon = _mm512_add_ph(m_vec, epsilon_vec);
    __m512h m_recip_approx = _mm512_rcp_ph(m_with_epsilon);
    __m512h ratio_a_vec = _mm512_mul_ph(_mm512_add_ph(a_vec, epsilon_vec), m_recip_approx);
    __m512h ratio_b_vec = _mm512_mul_ph(_mm512_add_ph(b_vec, epsilon_vec), m_recip_approx);
    __m512h log_ratio_a_vec = _simsimd_log2_f16_sapphire(ratio_a_vec);
    __m512h log_ratio_b_vec = _simsimd_log2_f16_sapphire(ratio_b_vec);
    sum_a_vec = _mm512_mask3_fmadd_ph(a_vec, log_ratio_a_vec, sum_a_vec, nonzero_mask);
    sum_b_vec = _mm512_mask3_fmadd_ph(b_vec, log_ratio_b_vec, sum_b_vec, nonzero_mask);
    if (n) goto simsimd_jsd_f16_sapphire_cycle;

    simsimd_f32_t log2_normalizer = 0.693147181f;
    simsimd_f32_t sum = _mm512_reduce_add_ph(_mm512_add_ph(sum_a_vec, sum_b_vec));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? _simsimd_sqrt_f32_haswell(sum) : 0;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE
#endif // _SIMSIMD_TARGET_X86

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_kld_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result) {
#if SIMSIMD_TARGET_NEON_F16
    simsimd_kld_f16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_kld_f16_haswell(a, b, n, result);
#else
    simsimd_kld_f16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_kld_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result) {
    simsimd_kld_bf16_serial(a, b, n, result);
}

SIMSIMD_PUBLIC void simsimd_kld_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_kld_f32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_kld_f32_skylake(a, b, n, result);
#else
    simsimd_kld_f32_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_kld_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result) {
    simsimd_kld_f64_serial(a, b, n, result);
}

SIMSIMD_PUBLIC void simsimd_jsd_f16(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result) {
#if SIMSIMD_TARGET_NEON_F16
    simsimd_jsd_f16_neon(a, b, n, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_jsd_f16_haswell(a, b, n, result);
#else
    simsimd_jsd_f16_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_jsd_bf16(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                     simsimd_distance_t *result) {
    simsimd_jsd_bf16_serial(a, b, n, result);
}

SIMSIMD_PUBLIC void simsimd_jsd_f32(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result) {
#if SIMSIMD_TARGET_NEON
    simsimd_jsd_f32_neon(a, b, n, result);
#elif SIMSIMD_TARGET_SKYLAKE
    simsimd_jsd_f32_skylake(a, b, n, result);
#else
    simsimd_jsd_f32_serial(a, b, n, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_jsd_f64(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                    simsimd_distance_t *result) {
    simsimd_jsd_f64_serial(a, b, n, result);
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
