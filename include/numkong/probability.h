/**
 *  @brief SIMD-accelerated Similarity Measures for Probability Distributions.
 *  @file include/numkong/probability.h
 *  @author Ash Vardanian
 *  @date October 20, 2023
 *
 *  Contains following similarity measures:
 *
 *  - Kullback-Leibler Divergence (KLD)
 *  - Jensen-Shannon Divergence (JSD)
 *
 *  For dtypes:
 *
 *  - 64-bit floating point numbers → 64-bit
 *  - 32-bit floating point numbers → 32-bit
 *  - 16-bit floating point numbers → 32-bit
 *  - 16-bit brain-floating point numbers → 32-bit
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire
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
#ifndef NK_PROBABILITY_H
#define NK_PROBABILITY_H

#include "numkong/types.h"
#include "numkong/reduce.h" // For horizontal reduction helpers

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
NK_DYNAMIC void nk_kld_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
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
NK_DYNAMIC void nk_kld_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
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
NK_DYNAMIC void nk_kld_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
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
NK_DYNAMIC void nk_kld_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
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
NK_DYNAMIC void nk_jsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
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
NK_DYNAMIC void nk_jsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
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
NK_DYNAMIC void nk_jsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
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
NK_DYNAMIC void nk_jsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);

/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_kld_bf16 */
NK_PUBLIC void nk_kld_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_bf16 */
NK_PUBLIC void nk_jsd_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SAPPHIRE

#include "numkong/cast/serial.h"    // `nk_f16_to_f64_serial`
#include "numkong/spatial/serial.h" // `nk_f32_sqrt_serial`

#define nk_define_kld_(input_type, accumulator_type, output_type, load_and_convert, epsilon, compute_log)   \
    NK_PUBLIC void nk_kld_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, output_type *result) {                         \
        nk_##accumulator_type##_t d = 0, ai, bi;                                                            \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            d += ai * compute_log((ai + epsilon) / (bi + epsilon));                                         \
        }                                                                                                   \
        *result = (output_type)d;                                                                           \
    }

#define nk_define_jsd_(input_type, accumulator_type, output_type, load_and_convert, epsilon, compute_log,   \
                       compute_sqrt)                                                                        \
    NK_PUBLIC void nk_jsd_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, output_type *result) {                         \
        nk_##accumulator_type##_t d = 0, ai, bi;                                                            \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            nk_##accumulator_type##_t mi = (ai + bi) / 2;                                                   \
            d += ai * compute_log((ai + epsilon) / (mi + epsilon));                                         \
            d += bi * compute_log((bi + epsilon) / (mi + epsilon));                                         \
        }                                                                                                   \
        output_type d_half = ((output_type)d / 2);                                                          \
        *result = d_half > 0 ? compute_sqrt(d_half) : 0;                                                    \
    }

/**
 *  @brief  Computes `log(x)` for any positive float using IEEE 754 bit extraction
 *          and a fast-converging series expansion.
 *
 *  Exploits the IEEE 754 representation to extract the exponent and mantissa:
 *  `log(x) = log(2) * exponent + log(mantissa)`. The mantissa is reduced to the
 *  range `[√2/2, √2]` for optimal convergence. Uses the transformation
 *  `u = (m-1)/(m+1)` which converges much faster than the classic Mercator series,
 *  since `u` is bounded to approximately `[-0.17, 0.17]` after range reduction.
 *
 *  Maximum relative error is approximately 0.00001% across all positive floats,
 *  roughly 300,000x more accurate than the 3-term Mercator series (which also
 *  only converges for inputs in `(0, 2)`).
 *
 *  https://en.wikipedia.org/wiki/Logarithm#Power_series
 */
NK_INTERNAL nk_f32_t nk_f32_log_serial_(nk_f32_t x) {
    nk_fui32_t conv;
    conv.f = x;
    int exp = ((conv.u >> 23) & 0xFF) - 127;
    conv.u = (conv.u & 0x007FFFFF) | 0x3F800000; // mantissa ∈ [1, 2)
    nk_f32_t m = conv.f;
    // Range reduction: if m > √2, halve it and increment exponent
    if (m > 1.41421356f) m *= 0.5f, exp++;
    // Use (m-1)/(m+1) transformation for faster convergence
    nk_f32_t u = (m - 1.0f) / (m + 1.0f);
    nk_f32_t u2 = u * u;
    // log(m) = 2 × (u + u³/3 + u⁵/5 + u⁷/7)
    nk_f32_t log_m = 2.0f * u * (1.0f + u2 * (0.3333333333f + u2 * (0.2f + u2 * 0.142857143f)));
    return (nk_f32_t)exp * 0.6931471805599453f + log_m;
}

/**
 *  @brief  Computes `log(x)` for any positive double using IEEE 754 bit extraction
 *          and a fast-converging series expansion.
 *
 *  Exploits the IEEE 754 representation to extract the 11-bit exponent and 52-bit mantissa:
 *  `log(x) = log(2) * exponent + log(mantissa)`. The mantissa is reduced to the
 *  range `[√2/2, √2]` for optimal convergence. Uses the transformation
 *  `u = (m-1)/(m+1)` which converges much faster than the classic Mercator series,
 *  since `u` is bounded to approximately `[-0.17, 0.17]` after range reduction.
 *
 *  Uses more series terms than the f32 version to achieve near-full f64 precision,
 *  with maximum relative error approximately 0.0000000001% across all positive doubles.
 *
 *  https://en.wikipedia.org/wiki/Logarithm#Power_series
 */
NK_INTERNAL nk_f64_t nk_f64_log_serial_(nk_f64_t x) {
    nk_fui64_t conv;
    conv.f = x;
    int exp = ((conv.u >> 52) & 0x7FF) - 1023;
    conv.u = (conv.u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL; // mantissa ∈ [1, 2)
    nk_f64_t m = conv.f;
    // Range reduction: if m > √2, halve it and increment exponent
    if (m > 1.4142135623730950488) m *= 0.5, exp++;
    // Use (m-1)/(m+1) transformation for faster convergence
    nk_f64_t u = (m - 1.0) / (m + 1.0);
    nk_f64_t u2 = u * u;
    // log(m) = 2 × (u + u³/3 + u⁵/5 + u⁷/7 + u⁹/9 + u¹¹/11 + u¹³/13)
    nk_f64_t log_m = 2.0 * u *
                     (1.0 + u2 * (0.3333333333333333 +
                                  u2 * (0.2 + u2 * (0.14285714285714285 +
                                                    u2 * (0.1111111111111111 +
                                                          u2 * (0.09090909090909091 + u2 * 0.07692307692307693))))));
    return (nk_f64_t)exp * 0.6931471805599453 + log_m;
}

nk_define_kld_(f64, f64, nk_f64_t, nk_assign_from_to_, NK_F64_DIVISION_EPSILON, nk_f64_log_serial_)
nk_define_jsd_(f64, f64, nk_f64_t, nk_assign_from_to_, NK_F64_DIVISION_EPSILON, nk_f64_log_serial_, nk_f64_sqrt_serial)

nk_define_kld_(f32, f32, nk_f32_t, nk_assign_from_to_, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_)
nk_define_jsd_(f32, f32, nk_f32_t, nk_assign_from_to_, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_, nk_f32_sqrt_serial)

nk_define_kld_(f16, f32, nk_f32_t, nk_f16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_)
nk_define_jsd_(f16, f32, nk_f32_t, nk_f16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_,
               nk_f32_sqrt_serial)

nk_define_kld_(bf16, f32, nk_f32_t, nk_bf16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_)
nk_define_jsd_(bf16, f32, nk_f32_t, nk_bf16_to_f32_serial, NK_F32_DIVISION_EPSILON, nk_f32_log_serial_,
               nk_f32_sqrt_serial)

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

NK_PUBLIC float32x4_t nk_log2_f32_neon_(float32x4_t x) {
    // Extracting the exponent
    int32x4_t bits_i32x4 = vreinterpretq_s32_f32(x);
    int32x4_t exponent_i32x4 = vsubq_s32(vshrq_n_s32(vandq_s32(bits_i32x4, vdupq_n_s32(0x7F800000)), 23),
                                         vdupq_n_s32(127));
    float32x4_t exponent_f32x4 = vcvtq_f32_s32(exponent_i32x4);

    // Extracting the mantissa
    float32x4_t mantissa_f32x4 = vreinterpretq_f32_s32(
        vorrq_s32(vandq_s32(bits_i32x4, vdupq_n_s32(0x007FFFFF)), vdupq_n_s32(0x3F800000)));

    // Constants for polynomial
    float32x4_t one_f32x4 = vdupq_n_f32(1.0f);
    float32x4_t poly_f32x4 = vdupq_n_f32(-3.4436006e-2f);

    // Compute polynomial using Horner's method
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(3.1821337e-1f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(-1.2315303f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(2.5988452f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(-3.3241990f), mantissa_f32x4, poly_f32x4);
    poly_f32x4 = vmlaq_f32(vdupq_n_f32(3.1157899f), mantissa_f32x4, poly_f32x4);

    // Final computation
    float32x4_t result_f32x4 = vaddq_f32(vmulq_f32(poly_f32x4, vsubq_f32(mantissa_f32x4, one_f32x4)), exponent_f32x4);
    return result_f32x4;
}

NK_PUBLIC void nk_kld_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t a_f32x4, b_f32x4;

nk_kld_f32_neon_cycle:
    if (n < 4) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b32x4_serial_(a, &a_vec, n);
        nk_partial_load_b32x4_serial_(b, &b_vec, n);
        a_f32x4 = a_vec.f32x4;
        b_f32x4 = b_vec.f32x4;
        n = 0;
    }
    else {
        a_f32x4 = vld1q_f32(a);
        b_f32x4 = vld1q_f32(b);
        n -= 4, a += 4, b += 4;
    }

    float32x4_t ratio_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(b_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_f32x4 = nk_log2_f32_neon_(ratio_f32x4);
    float32x4_t contribution_f32x4 = vmulq_f32(a_f32x4, log_ratio_f32x4);
    sum_f32x4 = vaddq_f32(sum_f32x4, contribution_f32x4);
    if (n != 0) goto nk_kld_f32_neon_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t a_f32x4, b_f32x4;

nk_jsd_f32_neon_cycle:
    if (n < 4) {
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b32x4_serial_(a, &a_vec, n);
        nk_partial_load_b32x4_serial_(b, &b_vec, n);
        a_f32x4 = a_vec.f32x4;
        b_f32x4 = b_vec.f32x4;
        n = 0;
    }
    else {
        a_f32x4 = vld1q_f32(a);
        b_f32x4 = vld1q_f32(b);
        n -= 4, a += 4, b += 4;
    }

    float32x4_t mean_f32x4 = vmulq_n_f32(vaddq_f32(a_f32x4, b_f32x4), 0.5f);
    float32x4_t ratio_a_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_f32x4 = vdivq_f32(vaddq_f32(b_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_a_f32x4 = nk_log2_f32_neon_(ratio_a_f32x4);
    float32x4_t log_ratio_b_f32x4 = nk_log2_f32_neon_(ratio_b_f32x4);
    float32x4_t contribution_a_f32x4 = vmulq_f32(a_f32x4, log_ratio_a_f32x4);
    float32x4_t contribution_b_f32x4 = vmulq_f32(b_f32x4, log_ratio_b_f32x4);

    sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(contribution_a_f32x4, contribution_b_f32x4));
    if (n != 0) goto nk_jsd_f32_neon_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_neon(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

#include "numkong/cast/serial.h" // nk_partial_load_b16x4_serial_

NK_PUBLIC void nk_kld_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t a_f32x4, b_f32x4;

nk_kld_f16_neonhalf_cycle:
    if (n < 4) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b16x4_serial_(a, &a_vec, n);
        nk_partial_load_b16x4_serial_(b, &b_vec, n);
        a_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(a_vec.u16x4));
        b_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(b_vec.u16x4));
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }

    float32x4_t ratio_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(b_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_f32x4 = nk_log2_f32_neon_(ratio_f32x4);
    float32x4_t contribution_f32x4 = vmulq_f32(a_f32x4, log_ratio_f32x4);
    sum_f32x4 = vaddq_f32(sum_f32x4, contribution_f32x4);
    if (n) goto nk_kld_f16_neonhalf_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    float32x4_t epsilon_f32x4 = vdupq_n_f32(epsilon);
    float32x4_t a_f32x4, b_f32x4;

nk_jsd_f16_neonhalf_cycle:
    if (n < 4) {
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b16x4_serial_(a, &a_vec, n);
        nk_partial_load_b16x4_serial_(b, &b_vec, n);
        a_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(a_vec.u16x4));
        b_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(b_vec.u16x4));
        n = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b));
        n -= 4, a += 4, b += 4;
    }

    float32x4_t mean_f32x4 = vmulq_n_f32(vaddq_f32(a_f32x4, b_f32x4), 0.5f);
    float32x4_t ratio_a_f32x4 = vdivq_f32(vaddq_f32(a_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t ratio_b_f32x4 = vdivq_f32(vaddq_f32(b_f32x4, epsilon_f32x4), vaddq_f32(mean_f32x4, epsilon_f32x4));
    float32x4_t log_ratio_a_f32x4 = nk_log2_f32_neon_(ratio_a_f32x4);
    float32x4_t log_ratio_b_f32x4 = nk_log2_f32_neon_(ratio_b_f32x4);
    float32x4_t contribution_a_f32x4 = vmulq_f32(a_f32x4, log_ratio_a_f32x4);
    float32x4_t contribution_b_f32x4 = vmulq_f32(b_f32x4, log_ratio_b_f32x4);
    sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(contribution_a_f32x4, contribution_b_f32x4));
    if (n) goto nk_jsd_f16_neonhalf_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = vaddvq_f32(sum_f32x4) * log2_normalizer / 2;
    *result = sum > 0 ? nk_f32_sqrt_neon(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m256 nk_log2_f32_haswell_(__m256 x) {
    // Extracting the exponent
    __m256i bits_i32x8 = _mm256_castps_si256(x);
    __m256i exponent_i32x8 = _mm256_srli_epi32(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x7F800000)), 23);
    exponent_i32x8 = _mm256_sub_epi32(exponent_i32x8, _mm256_set1_epi32(127)); // removing the bias
    __m256 exponent_f32x8 = _mm256_cvtepi32_ps(exponent_i32x8);

    // Extracting the mantissa
    __m256 mantissa_f32x8 = _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000)));

    // Constants for polynomial
    __m256 one_f32x8 = _mm256_set1_ps(1.0f);
    __m256 poly_f32x8 = _mm256_set1_ps(-3.4436006e-2f);

    // Compute the polynomial using Horner's method
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(3.1821337e-1f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(-1.2315303f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(2.5988452f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(-3.3241990f));
    poly_f32x8 = _mm256_fmadd_ps(mantissa_f32x8, poly_f32x8, _mm256_set1_ps(3.1157899f));

    // Final computation
    __m256 result_f32x8 = _mm256_add_ps(_mm256_mul_ps(poly_f32x8, _mm256_sub_ps(mantissa_f32x8, one_f32x8)),
                                        exponent_f32x8);
    return result_f32x8;
}

NK_PUBLIC void nk_kld_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 sum_f32x8 = _mm256_setzero_ps();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m256 epsilon_f32x8 = _mm256_set1_ps(epsilon);
    __m256 a_f32x8, b_f32x8;

nk_kld_f16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(a, n);
        b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b, n);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    a_f32x8 = _mm256_add_ps(a_f32x8, epsilon_f32x8);
    b_f32x8 = _mm256_add_ps(b_f32x8, epsilon_f32x8);
    __m256 ratio_f32x8 = _mm256_div_ps(a_f32x8, b_f32x8);
    __m256 log_ratio_f32x8 = nk_log2_f32_haswell_(ratio_f32x8);
    __m256 contribution_f32x8 = _mm256_mul_ps(a_f32x8, log_ratio_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_f32x8);
    if (n) goto nk_kld_f16_haswell_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    sum *= log2_normalizer;
    *result = sum;
}

NK_PUBLIC void nk_jsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m256 epsilon_f32x8 = _mm256_set1_ps(epsilon);
    __m256 sum_f32x8 = _mm256_setzero_ps();
    __m256 a_f32x8, b_f32x8;

nk_jsd_f16_haswell_cycle:
    if (n < 8) {
        a_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(a, n);
        b_f32x8 = nk_partial_load_f16x8_to_f32x8_haswell_(b, n);
        n = 0;
    }
    else {
        a_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)a));
        b_f32x8 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)b));
        n -= 8, a += 8, b += 8;
    }
    __m256 mean_f32x8 = _mm256_mul_ps(_mm256_add_ps(a_f32x8, b_f32x8), _mm256_set1_ps(0.5f)); // M = (P + Q) / 2
    __m256 ratio_a_f32x8 = _mm256_div_ps(_mm256_add_ps(a_f32x8, epsilon_f32x8),
                                         _mm256_add_ps(mean_f32x8, epsilon_f32x8));
    __m256 ratio_b_f32x8 = _mm256_div_ps(_mm256_add_ps(b_f32x8, epsilon_f32x8),
                                         _mm256_add_ps(mean_f32x8, epsilon_f32x8));
    __m256 log_ratio_a_f32x8 = nk_log2_f32_haswell_(ratio_a_f32x8);
    __m256 log_ratio_b_f32x8 = nk_log2_f32_haswell_(ratio_b_f32x8);
    __m256 contribution_a_f32x8 = _mm256_mul_ps(a_f32x8, log_ratio_a_f32x8);
    __m256 contribution_b_f32x8 = _mm256_mul_ps(b_f32x8, log_ratio_b_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_a_f32x8);
    sum_f32x8 = _mm256_add_ps(sum_f32x8, contribution_b_f32x8);
    if (n) goto nk_jsd_f16_haswell_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = nk_reduce_add_f32x8_haswell_(sum_f32x8);
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_sqrt_f32_haswell_(sum) : 0;
}

NK_PUBLIC void nk_kld_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_kld_f64_serial(a, b, n, result);
}

NK_PUBLIC void nk_jsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_jsd_f64_serial(a, b, n, result);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m512 nk_log2_f32_skylake_(__m512 x) {
    // Extract the exponent and mantissa
    __m512 one_f32x16 = _mm512_set1_ps(1.0f);
    __m512 exponent_f32x16 = _mm512_getexp_ps(x);
    __m512 mantissa_f32x16 = _mm512_getmant_ps(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512 poly_f32x16 = _mm512_set1_ps(-3.4436006e-2f);
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(3.1821337e-1f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(-1.2315303f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(2.5988452f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(-3.3241990f));
    poly_f32x16 = _mm512_fmadd_ps(mantissa_f32x16, poly_f32x16, _mm512_set1_ps(3.1157899f));

    return _mm512_add_ps(_mm512_mul_ps(poly_f32x16, _mm512_sub_ps(mantissa_f32x16, one_f32x16)), exponent_f32x16);
}

NK_PUBLIC void nk_kld_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_f32x16 = _mm512_setzero();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m512 epsilon_f32x16 = _mm512_set1_ps(epsilon);
    __m512 a_f32x16, b_f32x16;

nk_kld_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, a), epsilon_f32x16);
        b_f32x16 = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, b), epsilon_f32x16);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_add_ps(_mm512_loadu_ps(a), epsilon_f32x16);
        b_f32x16 = _mm512_add_ps(_mm512_loadu_ps(b), epsilon_f32x16);
        a += 16, b += 16, n -= 16;
    }
    __m512 ratio_f32x16 = _mm512_div_ps(a_f32x16, b_f32x16);
    __m512 log_ratio_f32x16 = nk_log2_f32_skylake_(ratio_f32x16);
    __m512 contribution_f32x16 = _mm512_mul_ps(a_f32x16, log_ratio_f32x16);
    sum_f32x16 = _mm512_add_ps(sum_f32x16, contribution_f32x16);
    if (n) goto nk_kld_f32_skylake_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    *result = _mm512_reduce_add_ps(sum_f32x16) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512 sum_a_f32x16 = _mm512_setzero();
    __m512 sum_b_f32x16 = _mm512_setzero();
    nk_f32_t epsilon = NK_F32_DIVISION_EPSILON;
    __m512 epsilon_f32x16 = _mm512_set1_ps(epsilon);
    __m512 a_f32x16, b_f32x16;

nk_jsd_f32_skylake_cycle:
    if (n < 16) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_f32x16 = _mm512_maskz_loadu_ps(mask, a);
        b_f32x16 = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_f32x16 = _mm512_loadu_ps(a);
        b_f32x16 = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    __m512 mean_f32x16 = _mm512_mul_ps(_mm512_add_ps(a_f32x16, b_f32x16), _mm512_set1_ps(0.5f));
    __mmask16 nonzero_mask_a = _mm512_cmp_ps_mask(a_f32x16, epsilon_f32x16, _CMP_GE_OQ);
    __mmask16 nonzero_mask_b = _mm512_cmp_ps_mask(b_f32x16, epsilon_f32x16, _CMP_GE_OQ);
    __mmask16 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512 mean_with_epsilon_f32x16 = _mm512_add_ps(mean_f32x16, epsilon_f32x16);
    __m512 mean_recip_approx_f32x16 = _mm512_rcp14_ps(mean_with_epsilon_f32x16);
    __m512 ratio_a_f32x16 = _mm512_mul_ps(_mm512_add_ps(a_f32x16, epsilon_f32x16), mean_recip_approx_f32x16);
    __m512 ratio_b_f32x16 = _mm512_mul_ps(_mm512_add_ps(b_f32x16, epsilon_f32x16), mean_recip_approx_f32x16);
    __m512 log_ratio_a_f32x16 = nk_log2_f32_skylake_(ratio_a_f32x16);
    __m512 log_ratio_b_f32x16 = nk_log2_f32_skylake_(ratio_b_f32x16);
    sum_a_f32x16 = _mm512_mask3_fmadd_ps(a_f32x16, log_ratio_a_f32x16, sum_a_f32x16, nonzero_mask);
    sum_b_f32x16 = _mm512_mask3_fmadd_ps(b_f32x16, log_ratio_b_f32x16, sum_b_f32x16, nonzero_mask);
    if (n) goto nk_jsd_f32_skylake_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = _mm512_reduce_add_ps(_mm512_add_ps(sum_a_f32x16, sum_b_f32x16));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_sqrt_f32_haswell_(sum) : 0;
}

NK_INTERNAL __m512d nk_log2_f64_skylake_(__m512d x) {
    // Extract the exponent and mantissa: x = 2^exp × m, m ∈ [1, 2)
    __m512d one_f64x8 = _mm512_set1_pd(1.0);
    __m512d two_f64x8 = _mm512_set1_pd(2.0);
    __m512d exponent_f64x8 = _mm512_getexp_pd(x);
    __m512d mantissa_f64x8 = _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute log2(m) using the s-series: s = (m-1)/(m+1), s ∈ [0, 1/3] for m ∈ [1, 2)
    // ln(m) = 2 × s × (1 + s²/3 + s⁴/5 + s⁶/7 + ...) converges fast since s² ≤ 1/9
    // log2(m) = ln(m) × log2(e)
    __m512d s_f64x8 = _mm512_div_pd(_mm512_sub_pd(mantissa_f64x8, one_f64x8), _mm512_add_pd(mantissa_f64x8, one_f64x8));
    __m512d s2_f64x8 = _mm512_mul_pd(s_f64x8, s_f64x8);

    // Polynomial P(s²) = 1 + s²/3 + s⁴/5 + ... using Horner's method
    // 14 terms (k=0..13) achieves ~1 ULP accuracy for f64
    __m512d poly_f64x8 = _mm512_set1_pd(1.0 / 27.0); // 1/(2*13+1)
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 25.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 23.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 21.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 19.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 17.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 15.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 13.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 11.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 9.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 7.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 5.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0 / 3.0));
    poly_f64x8 = _mm512_fmadd_pd(s2_f64x8, poly_f64x8, _mm512_set1_pd(1.0));

    // ln(m) = 2 × s × P(s²), then log2(m) = ln(m) × log2(e)
    __m512d ln_m_f64x8 = _mm512_mul_pd(_mm512_mul_pd(two_f64x8, s_f64x8), poly_f64x8);
    __m512d log2e_f64x8 = _mm512_set1_pd(1.4426950408889634); // 1/ln(2)
    __m512d log2_m_f64x8 = _mm512_mul_pd(ln_m_f64x8, log2e_f64x8);

    // log2(x) = exponent + log2(m)
    return _mm512_add_pd(exponent_f64x8, log2_m_f64x8);
}

NK_PUBLIC void nk_kld_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_f64_t epsilon = NK_F64_DIVISION_EPSILON;
    __m512d epsilon_f64x8 = _mm512_set1_pd(epsilon);
    __m512d a_f64x8, b_f64x8;

nk_kld_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        a_f64x8 = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, a), epsilon_f64x8);
        b_f64x8 = _mm512_add_pd(_mm512_maskz_loadu_pd(mask, b), epsilon_f64x8);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_add_pd(_mm512_loadu_pd(a), epsilon_f64x8);
        b_f64x8 = _mm512_add_pd(_mm512_loadu_pd(b), epsilon_f64x8);
        a += 8, b += 8, n -= 8;
    }
    __m512d ratio_f64x8 = _mm512_div_pd(a_f64x8, b_f64x8);
    __m512d log_ratio_f64x8 = nk_log2_f64_skylake_(ratio_f64x8);
    __m512d contribution_f64x8 = _mm512_mul_pd(a_f64x8, log_ratio_f64x8);
    sum_f64x8 = _mm512_add_pd(sum_f64x8, contribution_f64x8);
    if (n) goto nk_kld_f64_skylake_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    *result = _mm512_reduce_add_pd(sum_f64x8) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m512d sum_a_f64x8 = _mm512_setzero_pd();
    __m512d sum_b_f64x8 = _mm512_setzero_pd();
    nk_f64_t epsilon = NK_F64_DIVISION_EPSILON;
    __m512d epsilon_f64x8 = _mm512_set1_pd(epsilon);
    __m512d a_f64x8, b_f64x8;

nk_jsd_f64_skylake_cycle:
    if (n < 8) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        a_f64x8 = _mm512_maskz_loadu_pd(mask, a);
        b_f64x8 = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_f64x8 = _mm512_loadu_pd(a);
        b_f64x8 = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    __m512d mean_f64x8 = _mm512_mul_pd(_mm512_add_pd(a_f64x8, b_f64x8), _mm512_set1_pd(0.5));
    __mmask8 nonzero_mask_a = _mm512_cmp_pd_mask(a_f64x8, epsilon_f64x8, _CMP_GE_OQ);
    __mmask8 nonzero_mask_b = _mm512_cmp_pd_mask(b_f64x8, epsilon_f64x8, _CMP_GE_OQ);
    __mmask8 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512d mean_with_epsilon_f64x8 = _mm512_add_pd(mean_f64x8, epsilon_f64x8);
    // Use full precision division (not rcp14 approximate which only has 14 bits)
    __m512d ratio_a_f64x8 = _mm512_div_pd(_mm512_add_pd(a_f64x8, epsilon_f64x8), mean_with_epsilon_f64x8);
    __m512d ratio_b_f64x8 = _mm512_div_pd(_mm512_add_pd(b_f64x8, epsilon_f64x8), mean_with_epsilon_f64x8);
    __m512d log_ratio_a_f64x8 = nk_log2_f64_skylake_(ratio_a_f64x8);
    __m512d log_ratio_b_f64x8 = nk_log2_f64_skylake_(ratio_b_f64x8);
    sum_a_f64x8 = _mm512_mask3_fmadd_pd(a_f64x8, log_ratio_a_f64x8, sum_a_f64x8, nonzero_mask);
    sum_b_f64x8 = _mm512_mask3_fmadd_pd(b_f64x8, log_ratio_b_f64x8, sum_b_f64x8, nonzero_mask);
    if (n) goto nk_jsd_f64_skylake_cycle;

    nk_f64_t log2_normalizer = 0.6931471805599453;
    nk_f64_t sum = _mm512_reduce_add_pd(_mm512_add_pd(sum_a_f64x8, sum_b_f64x8));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_sqrt_f64_haswell_(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_SAPPHIRE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_INTERNAL __m512h nk_log2_f16_sapphire_(__m512h x) {
    // Extract the exponent and mantissa
    __m512h one_f16x32 = _mm512_set1_ph((nk_f16_t)1);
    __m512h exponent_f16x32 = _mm512_getexp_ph(x);
    __m512h mantissa_f16x32 = _mm512_getmant_ph(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);

    // Compute the polynomial using Horner's method
    __m512h poly_f16x32 = _mm512_set1_ph((nk_f16_t)-3.4436006e-2f);
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)3.1821337e-1f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)-1.2315303f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)2.5988452f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)-3.3241990f));
    poly_f16x32 = _mm512_fmadd_ph(mantissa_f16x32, poly_f16x32, _mm512_set1_ph((nk_f16_t)3.1157899f));

    return _mm512_add_ph(_mm512_mul_ph(poly_f16x32, _mm512_sub_ph(mantissa_f16x32, one_f16x32)), exponent_f16x32);
}

NK_PUBLIC void nk_kld_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512h sum_f16x32 = _mm512_setzero_ph();
    __m512h epsilon_f16x32 = _mm512_set1_ph((nk_f16_t)NK_F16_DIVISION_EPSILON);
    __m512h a_f16x32, b_f16x32;

nk_kld_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a)), epsilon_f16x32);
        b_f16x32 = _mm512_maskz_add_ph(mask, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b)), epsilon_f16x32);
        n = 0;
    }
    else {
        a_f16x32 = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(a)), epsilon_f16x32);
        b_f16x32 = _mm512_add_ph(_mm512_castsi512_ph(_mm512_loadu_epi16(b)), epsilon_f16x32);
        a += 32, b += 32, n -= 32;
    }
    __m512h ratio_f16x32 = _mm512_div_ph(a_f16x32, b_f16x32);
    __m512h log_ratio_f16x32 = nk_log2_f16_sapphire_(ratio_f16x32);
    __m512h contribution_f16x32 = _mm512_mul_ph(a_f16x32, log_ratio_f16x32);
    sum_f16x32 = _mm512_add_ph(sum_f16x32, contribution_f16x32);
    if (n) goto nk_kld_f16_sapphire_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    *result = _mm512_reduce_add_ph(sum_f16x32) * log2_normalizer;
}

NK_PUBLIC void nk_jsd_f16_sapphire(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m512h sum_a_f16x32 = _mm512_setzero_ph();
    __m512h sum_b_f16x32 = _mm512_setzero_ph();
    __m512h epsilon_f16x32 = _mm512_set1_ph((nk_f16_t)NK_F16_DIVISION_EPSILON);
    __m512h a_f16x32, b_f16x32;

nk_jsd_f16_sapphire_cycle:
    if (n < 32) {
        __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(a));
        b_f16x32 = _mm512_castsi512_ph(_mm512_loadu_epi16(b));
        a += 32, b += 32, n -= 32;
    }
    __m512h mean_f16x32 = _mm512_mul_ph(_mm512_add_ph(a_f16x32, b_f16x32), _mm512_set1_ph((nk_f16_t)0.5f));
    __mmask32 nonzero_mask_a = _mm512_cmp_ph_mask(a_f16x32, epsilon_f16x32, _CMP_GE_OQ);
    __mmask32 nonzero_mask_b = _mm512_cmp_ph_mask(b_f16x32, epsilon_f16x32, _CMP_GE_OQ);
    __mmask32 nonzero_mask = nonzero_mask_a & nonzero_mask_b;
    __m512h mean_with_epsilon_f16x32 = _mm512_add_ph(mean_f16x32, epsilon_f16x32);
    __m512h mean_recip_approx_f16x32 = _mm512_rcp_ph(mean_with_epsilon_f16x32);
    __m512h ratio_a_f16x32 = _mm512_mul_ph(_mm512_add_ph(a_f16x32, epsilon_f16x32), mean_recip_approx_f16x32);
    __m512h ratio_b_f16x32 = _mm512_mul_ph(_mm512_add_ph(b_f16x32, epsilon_f16x32), mean_recip_approx_f16x32);
    __m512h log_ratio_a_f16x32 = nk_log2_f16_sapphire_(ratio_a_f16x32);
    __m512h log_ratio_b_f16x32 = nk_log2_f16_sapphire_(ratio_b_f16x32);
    sum_a_f16x32 = _mm512_mask3_fmadd_ph(a_f16x32, log_ratio_a_f16x32, sum_a_f16x32, nonzero_mask);
    sum_b_f16x32 = _mm512_mask3_fmadd_ph(b_f16x32, log_ratio_b_f16x32, sum_b_f16x32, nonzero_mask);
    if (n) goto nk_jsd_f16_sapphire_cycle;

    nk_f32_t log2_normalizer = 0.693147181f;
    nk_f32_t sum = _mm512_reduce_add_ph(_mm512_add_ph(sum_a_f16x32, sum_b_f16x32));
    sum *= log2_normalizer / 2;
    *result = sum > 0 ? nk_sqrt_f32_haswell_(sum) : 0;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_kld_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_NEONHALF
    nk_kld_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_kld_f16_haswell(a, b, n, result);
#else
    nk_kld_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_kld_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_kld_bf16_serial(a, b, n, result);
}

NK_PUBLIC void nk_kld_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_NEON
    nk_kld_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_kld_f32_skylake(a, b, n, result);
#else
    nk_kld_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_kld_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kld_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_kld_f64_haswell(a, b, n, result);
#else
    nk_kld_f64_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_NEONHALF
    nk_jsd_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jsd_f16_haswell(a, b, n, result);
#else
    nk_jsd_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_jsd_bf16_serial(a, b, n, result);
}

NK_PUBLIC void nk_jsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_NEON
    nk_jsd_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_jsd_f32_skylake(a, b, n, result);
#else
    nk_jsd_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_jsd_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jsd_f64_haswell(a, b, n, result);
#else
    nk_jsd_f64_serial(a, b, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
}
#endif

#endif
