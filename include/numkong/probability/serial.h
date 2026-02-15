/**
 *  @brief Serial Probability Distribution Similarity Measures.
 *  @file include/numkong/probability/serial.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/probability.h
 */
#ifndef NK_PROBABILITY_SERIAL_H
#define NK_PROBABILITY_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"    // `nk_f16_to_f32_serial`, `nk_bf16_to_f32_serial`, `nk_assign_from_to_`
#include "numkong/spatial/serial.h" // `nk_f32_sqrt_serial`, `nk_f64_sqrt_serial`

#if defined(__cplusplus)
extern "C" {
#endif

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

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_PROBABILITY_SERIAL_H
