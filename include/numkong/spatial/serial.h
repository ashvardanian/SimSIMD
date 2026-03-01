/**
 *  @brief SWAR-accelerated Spatial Similarity Measures for SIMD-free CPUs.
 *  @file include/numkong/spatial/serial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/spatial.h
 */
#ifndef NK_SPATIAL_SERIAL_H
#define NK_SPATIAL_SERIAL_H

#include "numkong/types.h"
#include "numkong/cast/serial.h"
#include "numkong/dot/serial.h" // `nk_dot_f64x2_state_serial_t`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for L2 squared distance with Neumaier compensated summation.
 *
 *  Implements Neumaier's Kahan-Babuška variant to minimize floating-point rounding errors.
 *  Unlike Kahan, Neumaier handles the case where the term being added is larger than the
 *  running sum. Achieves O(1) error growth regardless of vector dimension.
 *
 *  Performance vs Accuracy Tradeoff:
 *  - Adds ~30% overhead (3 extra FP operations per iteration) compared to naive summation
 *  - Reduces relative error from ~10⁻⁵ to ~10⁻⁷ at n=100K for f32
 *  - Benefits all floating-point types: f64, f32, f16, bf16
 *  - Integer types (i8) maintain perfect accuracy regardless
 *
 *  Algorithm: For each term, compute t = sum + term, then:
 *    - If |sum| ≥ |term|: c += (sum − t) + term   (lost low-order bits of term)
 *    - Else:              c += (term − t) + sum   (lost low-order bits of sum)
 *
 *  @see Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
 */
#define nk_define_sqeuclidean_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_sqeuclidean_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                        nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t sum = 0, compensation = 0, a_element, b_element;                                  \
        for (nk_size_t i = 0; i != n; ++i) {                                                                        \
            load_and_convert(a + i, &a_element);                                                                    \
            load_and_convert(b + i, &b_element);                                                                    \
            nk_##accumulator_type##_t diff = a_element - b_element;                                                 \
            nk_##accumulator_type##_t term = diff * diff, t = sum + term;                                           \
            compensation += (nk_##accumulator_type##_abs_(sum) >= nk_##accumulator_type##_abs_(term))               \
                                ? ((sum - t) + term)                                                                \
                                : ((term - t) + sum);                                                               \
            sum = t;                                                                                                \
        }                                                                                                           \
        *result = (nk_##output_type##_t)(sum + compensation);                                                       \
    }

#define nk_define_euclidean_(input_type, accumulator_type, l2sq_output_type, output_type, load_and_convert,       \
                             compute_sqrt)                                                                        \
    NK_PUBLIC void nk_euclidean_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                      nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##l2sq_output_type##_t distance_sq;                                                                    \
        nk_sqeuclidean_##input_type##_serial(a, b, n, &distance_sq);                                              \
        *result = compute_sqrt((nk_##output_type##_t)distance_sq);                                                \
    }

/**
 *  @brief Macro for cosine/angular distance with Neumaier compensated summation.
 *
 *  Uses Neumaier summation for all three accumulators (dot_product, a_norm_sq, b_norm_sq).
 *  Achieves O(1) error growth regardless of vector dimension.
 *
 *  @see nk_define_sqeuclidean_ for detailed documentation on Neumaier summation.
 */
#define nk_define_angular_(input_type, accumulator_type, output_type, load_and_convert, compute_rsqrt)            \
    NK_PUBLIC void nk_angular_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b,   \
                                                    nk_size_t n, nk_##output_type##_t *result) {                  \
        nk_##accumulator_type##_t dot_sum = 0, a_sum = 0, b_sum = 0, a_element, b_element;                        \
        nk_##accumulator_type##_t compensation_dot = 0, compensation_a = 0, compensation_b = 0;                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                      \
            load_and_convert(a + i, &a_element);                                                                  \
            load_and_convert(b + i, &b_element);                                                                  \
            nk_##accumulator_type##_t term_dot = a_element * b_element, t_dot = dot_sum + term_dot;               \
            nk_##accumulator_type##_t term_a = a_element * a_element, t_a = a_sum + term_a;                       \
            nk_##accumulator_type##_t term_b = b_element * b_element, t_b = b_sum + term_b;                       \
            compensation_dot += (nk_##accumulator_type##_abs_(dot_sum) >= nk_##accumulator_type##_abs_(term_dot)) \
                                    ? ((dot_sum - t_dot) + term_dot)                                              \
                                    : ((term_dot - t_dot) + dot_sum);                                             \
            compensation_a += (nk_##accumulator_type##_abs_(a_sum) >= nk_##accumulator_type##_abs_(term_a))       \
                                  ? ((a_sum - t_a) + term_a)                                                      \
                                  : ((term_a - t_a) + a_sum);                                                     \
            compensation_b += (nk_##accumulator_type##_abs_(b_sum) >= nk_##accumulator_type##_abs_(term_b))       \
                                  ? ((b_sum - t_b) + term_b)                                                      \
                                  : ((term_b - t_b) + b_sum);                                                     \
            dot_sum = t_dot;                                                                                      \
            a_sum = t_a;                                                                                          \
            b_sum = t_b;                                                                                          \
        }                                                                                                         \
        nk_##accumulator_type##_t dot_product = dot_sum + compensation_dot;                                       \
        nk_##accumulator_type##_t a_norm_sq = a_sum + compensation_a;                                             \
        nk_##accumulator_type##_t b_norm_sq = b_sum + compensation_b;                                             \
        if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }                                                    \
        else if (dot_product == 0) { *result = 1; }                                                               \
        else {                                                                                                    \
            nk_##output_type##_t unclipped_distance = 1 - dot_product * compute_rsqrt(a_norm_sq) *                \
                                                              compute_rsqrt(b_norm_sq);                           \
            *result = unclipped_distance > 0 ? unclipped_distance : 0;                                            \
        }                                                                                                         \
    }

/**
 *  @brief  Computes `1/√x` using the trick from Quake 3,
 *          with three Newton-Raphson iterations for full f32 accuracy.
 *
 *  The initial guess uses bit manipulation exploiting IEEE 754 float representation.
 *  The magic constant `0x5F375A86` is an improved version of Lomont's constant.
 *  Three Newton-Raphson refinement steps yield ~34.9 correct bits, exceeding the
 *  23-bit f32 mantissa for near-correctly-rounded results.
 *
 *  Subsequent additions by hardware manufacturers have made this algorithm redundant for the most part.
 *  For example, on x86, Intel introduced the SSE instruction `rsqrtss` in 1999. In a 2009 benchmark on
 *  the Intel Core 2, this instruction took 0.85ns per float compared to 3.54ns for the fast inverse
 *  square root algorithm, and had less error. Carmack's Magic Number `rsqrt` had an average error
 *  of 0.0990%, while SSE `rsqrtss` had 0.0094%, a 10x improvement.
 *
 *  https://web.archive.org/web/20210208132927/http://assemblyrequired.crashworks.org/timing-square-root/
 *  https://stackoverflow.com/a/41460625/2766161
 */
NK_PUBLIC nk_f32_t nk_f32_rsqrt_serial(nk_f32_t number) {
    nk_fui32_t conv;
    conv.f = number;
    conv.u = 0x5F375A86 - (conv.u >> 1);
    nk_f32_t y = conv.f;
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    y = y * (1.5f - 0.5f * number * y * y);
    return y;
}

/**
 *  @brief  Approximates `√x` using the identity `√x = x × rsqrt(x)`.
 *
 *  Leverages the fast inverse square root approximation and multiplies by `number`.
 *  Inherits the ~0.0005% maximum relative error from the underlying `rsqrt` implementation.
 *  This technique is useful where `sqrt` approximation is needed in performance-critical code,
 *  though modern hardware provides optimized alternatives like SSE `sqrtss`.
 */
NK_PUBLIC nk_f32_t nk_f32_sqrt_serial(nk_f32_t number) { return number > 0 ? number * nk_f32_rsqrt_serial(number) : 0; }

/**
 *  @brief  Computes `1/√x` for double precision using the Quake 3 trick,
 *          with four Newton-Raphson iterations for full f64 accuracy.
 *
 *  The initial guess uses bit manipulation exploiting IEEE 754 double representation.
 *  The magic constant `0x5FE6EB50C7B537A9` is the 64-bit analog of Lomont's constant,
 *  derived using the same methodology but adjusted for the 11-bit exponent and 52-bit
 *  mantissa of doubles. Four Newton-Raphson iterations yield ~69.3 correct bits,
 *  exceeding the 52-bit f64 mantissa for near-correctly-rounded results.
 *
 *  For modern x86, the `sqrtsd` instruction followed by division, or `_mm_cvtsd_f64(_mm_rsqrt14_sd(...))`
 *  with AVX-512, may be preferable for production use.
 */
NK_PUBLIC nk_f64_t nk_f64_rsqrt_serial(nk_f64_t number) {
    nk_fui64_t conv;
    conv.f = number;
    conv.u = 0x5FE6EB50C7B537A9ULL - (conv.u >> 1);
    nk_f64_t y = conv.f;
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    y = y * (1.5 - 0.5 * number * y * y);
    return y;
}

/**
 *  @brief  Approximates `√x` for double precision using `√x = x × rsqrt(x)`.
 *
 *  Leverages the fast inverse square root approximation and multiplies by `number`.
 *  Inherits near-full f64 precision from the underlying `rsqrt` implementation.
 */
NK_PUBLIC nk_f64_t nk_f64_sqrt_serial(nk_f64_t number) { return number > 0 ? number * nk_f64_rsqrt_serial(number) : 0; }

nk_define_angular_(f64, f64, f64, nk_assign_from_to_, nk_f64_rsqrt_serial)       // nk_angular_f64_serial
nk_define_sqeuclidean_(f64, f64, f64, nk_assign_from_to_)                        // nk_sqeuclidean_f64_serial
nk_define_euclidean_(f64, f64, f64, f64, nk_assign_from_to_, nk_f64_sqrt_serial) // nk_euclidean_f64_serial

nk_define_angular_(f32, f64, f32, nk_assign_from_to_, nk_f64_rsqrt_serial)       // nk_angular_f32_serial
nk_define_sqeuclidean_(f32, f64, f32, nk_assign_from_to_)                        // nk_sqeuclidean_f32_serial
nk_define_euclidean_(f32, f64, f32, f32, nk_assign_from_to_, nk_f64_sqrt_serial) // nk_euclidean_f32_serial

nk_define_angular_(f16, f32, f32, nk_f16_to_f32_serial, nk_f32_rsqrt_serial)       // nk_angular_f16_serial
nk_define_sqeuclidean_(f16, f32, f32, nk_f16_to_f32_serial)                        // nk_sqeuclidean_f16_serial
nk_define_euclidean_(f16, f32, f32, f32, nk_f16_to_f32_serial, nk_f32_sqrt_serial) // nk_euclidean_f16_serial

nk_define_angular_(bf16, f32, f32, nk_bf16_to_f32_serial, nk_f32_rsqrt_serial)       // nk_angular_bf16_serial
nk_define_sqeuclidean_(bf16, f32, f32, nk_bf16_to_f32_serial)                        // nk_sqeuclidean_bf16_serial
nk_define_euclidean_(bf16, f32, f32, f32, nk_bf16_to_f32_serial, nk_f32_sqrt_serial) // nk_euclidean_bf16_serial

nk_define_angular_(e4m3, f32, f32, nk_e4m3_to_f32_serial, nk_f32_rsqrt_serial)       // nk_angular_e4m3_serial
nk_define_sqeuclidean_(e4m3, f32, f32, nk_e4m3_to_f32_serial)                        // nk_sqeuclidean_e4m3_serial
nk_define_euclidean_(e4m3, f32, f32, f32, nk_e4m3_to_f32_serial, nk_f32_sqrt_serial) // nk_euclidean_e4m3_serial

nk_define_angular_(e5m2, f32, f32, nk_e5m2_to_f32_serial, nk_f32_rsqrt_serial)       // nk_angular_e5m2_serial
nk_define_sqeuclidean_(e5m2, f32, f32, nk_e5m2_to_f32_serial)                        // nk_sqeuclidean_e5m2_serial
nk_define_euclidean_(e5m2, f32, f32, f32, nk_e5m2_to_f32_serial, nk_f32_sqrt_serial) // nk_euclidean_e5m2_serial

nk_define_angular_(e2m3, f32, f32, nk_e2m3_to_f32_serial, nk_f32_rsqrt_serial)       // nk_angular_e2m3_serial
nk_define_sqeuclidean_(e2m3, f32, f32, nk_e2m3_to_f32_serial)                        // nk_sqeuclidean_e2m3_serial
nk_define_euclidean_(e2m3, f32, f32, f32, nk_e2m3_to_f32_serial, nk_f32_sqrt_serial) // nk_euclidean_e2m3_serial

nk_define_angular_(e3m2, f32, f32, nk_e3m2_to_f32_serial, nk_f32_rsqrt_serial)       // nk_angular_e3m2_serial
nk_define_sqeuclidean_(e3m2, f32, f32, nk_e3m2_to_f32_serial)                        // nk_sqeuclidean_e3m2_serial
nk_define_euclidean_(e3m2, f32, f32, f32, nk_e3m2_to_f32_serial, nk_f32_sqrt_serial) // nk_euclidean_e3m2_serial

nk_define_angular_(i8, i32, f32, nk_assign_from_to_, nk_f32_rsqrt_serial)       // nk_angular_i8_serial
nk_define_sqeuclidean_(i8, i32, u32, nk_assign_from_to_)                        // nk_sqeuclidean_i8_serial
nk_define_euclidean_(i8, i32, u32, f32, nk_assign_from_to_, nk_f32_sqrt_serial) // nk_euclidean_i8_serial

nk_define_angular_(u8, u32, f32, nk_assign_from_to_, nk_f32_rsqrt_serial)       // nk_angular_u8_serial
nk_define_sqeuclidean_(u8, u32, u32, nk_assign_from_to_)                        // nk_sqeuclidean_u8_serial
nk_define_euclidean_(u8, u32, u32, f32, nk_assign_from_to_, nk_f32_sqrt_serial) // nk_euclidean_u8_serial

#undef nk_define_sqeuclidean_
#undef nk_define_euclidean_
#undef nk_define_angular_

NK_PUBLIC void nk_sqeuclidean_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Sign extension: (nibble ^ 8) - 8 maps [0,15] to [-8,7]
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    nk_i32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        // Extract low nibbles (first dimension of pair)
        nk_i32_t a_lo = (nk_i32_t)((a[i] & 0x0F) ^ 8) - 8;
        nk_i32_t b_lo = (nk_i32_t)((b[i] & 0x0F) ^ 8) - 8;
        nk_i32_t diff_lo = a_lo - b_lo;
        sum += diff_lo * diff_lo;

        // Extract high nibbles (second dimension of pair) - skip if n is odd and this is last byte
        if (2 * i + 1 < n) {
            nk_i32_t a_hi = (nk_i32_t)(((a[i] >> 4) & 0x0F) ^ 8) - 8;
            nk_i32_t b_hi = (nk_i32_t)(((b[i] >> 4) & 0x0F) ^ 8) - 8;
            nk_i32_t diff_hi = a_hi - b_hi;
            sum += diff_hi * diff_hi;
        }
    }
    *result = (nk_u32_t)sum;
}

NK_PUBLIC void nk_euclidean_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq;
    nk_sqeuclidean_i4_serial(a, b, n, &distance_sq);
    *result = nk_f32_sqrt_serial((nk_f32_t)distance_sq);
}

NK_PUBLIC void nk_angular_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    nk_i32_t dot_sum = 0, a_norm_sq = 0, b_norm_sq = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        // Extract low nibbles
        nk_i32_t a_lo = (nk_i32_t)((a[i] & 0x0F) ^ 8) - 8;
        nk_i32_t b_lo = (nk_i32_t)((b[i] & 0x0F) ^ 8) - 8;
        dot_sum += a_lo * b_lo;
        a_norm_sq += a_lo * a_lo;
        b_norm_sq += b_lo * b_lo;

        // Extract high nibbles - skip if n is odd and this is last byte
        if (2 * i + 1 < n) {
            nk_i32_t a_hi = (nk_i32_t)(((a[i] >> 4) & 0x0F) ^ 8) - 8;
            nk_i32_t b_hi = (nk_i32_t)(((b[i] >> 4) & 0x0F) ^ 8) - 8;
            dot_sum += a_hi * b_hi;
            a_norm_sq += a_hi * a_hi;
            b_norm_sq += b_hi * b_hi;
        }
    }
    if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }
    else if (dot_sum == 0) { *result = 1; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_sum * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

NK_PUBLIC void nk_sqeuclidean_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // No sign extension needed - values are in [0,15].
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    nk_u32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        // Extract low nibbles
        nk_i32_t a_lo = (nk_i32_t)(a[i] & 0x0F);
        nk_i32_t b_lo = (nk_i32_t)(b[i] & 0x0F);
        nk_i32_t diff_lo = a_lo - b_lo;
        sum += (nk_u32_t)(diff_lo * diff_lo);

        // Extract high nibbles - skip if n is odd and this is last byte
        if (2 * i + 1 < n) {
            nk_i32_t a_hi = (nk_i32_t)((a[i] >> 4) & 0x0F);
            nk_i32_t b_hi = (nk_i32_t)((b[i] >> 4) & 0x0F);
            nk_i32_t diff_hi = a_hi - b_hi;
            sum += (nk_u32_t)(diff_hi * diff_hi);
        }
    }
    *result = sum;
}

NK_PUBLIC void nk_euclidean_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq;
    nk_sqeuclidean_u4_serial(a, b, n, &distance_sq);
    *result = nk_f32_sqrt_serial((nk_f32_t)distance_sq);
}

NK_PUBLIC void nk_angular_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    nk_u32_t dot_sum = 0, a_norm_sq = 0, b_norm_sq = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        // Extract low nibbles
        nk_u32_t a_lo = (nk_u32_t)(a[i] & 0x0F);
        nk_u32_t b_lo = (nk_u32_t)(b[i] & 0x0F);
        dot_sum += a_lo * b_lo;
        a_norm_sq += a_lo * a_lo;
        b_norm_sq += b_lo * b_lo;

        // Extract high nibbles - skip if n is odd and this is last byte
        if (2 * i + 1 < n) {
            nk_u32_t a_hi = (nk_u32_t)((a[i] >> 4) & 0x0F);
            nk_u32_t b_hi = (nk_u32_t)((b[i] >> 4) & 0x0F);
            dot_sum += a_hi * b_hi;
            a_norm_sq += a_hi * a_hi;
            b_norm_sq += b_hi * b_hi;
        }
    }
    if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }
    else if (dot_sum == 0) { *result = 1; }
    else {
        nk_f32_t unclipped = 1.0f - (nk_f32_t)dot_sum * nk_f32_rsqrt_serial((nk_f32_t)a_norm_sq) *
                                        nk_f32_rsqrt_serial((nk_f32_t)b_norm_sq);
        *result = unclipped > 0 ? unclipped : 0;
    }
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs (serial). */
NK_INTERNAL void nk_angular_through_f32_from_dot_serial_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                         nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t product = query_sumsq * target_sumsqs.f32s[i];
        if (product > 0) {
            nk_f32_t rsqrt_val = nk_f32_rsqrt_serial(product);
            nk_f32_t normalized = dots.f32s[i] * rsqrt_val;
            nk_f32_t result = 1.0f - normalized;
            results->f32s[i] = result > 0 ? result : 0;
        }
        else { results->f32s[i] = (dots.f32s[i] == 0) ? 0.0f : 1.0f; }
    }
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs (serial). */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_serial_(nk_b128_vec_t dots, nk_f32_t query_sumsq,
                                                           nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dist_sq = query_sumsq + target_sumsqs.f32s[i] - 2.0f * dots.f32s[i];
        results->f32s[i] = dist_sq > 0 ? nk_f32_sqrt_serial(dist_sq) : 0.0f;
    }
}

/** @brief Angular from_dot for f64 precision. */
NK_INTERNAL void nk_angular_through_f64_from_dot_serial_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                         nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f64_t product = query_sumsq * target_sumsqs.f64s[i];
        if (product > 0) {
            nk_f64_t rsqrt_val = nk_f64_rsqrt_serial(product);
            nk_f64_t normalized = dots.f64s[i] * rsqrt_val;
            nk_f64_t result = 1.0 - normalized;
            results->f64s[i] = result > 0 ? result : 0;
        }
        else { results->f64s[i] = (dots.f64s[i] == 0) ? 0.0 : 1.0; }
    }
}

/** @brief Euclidean from_dot for f64 precision. */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_serial_(nk_b256_vec_t dots, nk_f64_t query_sumsq,
                                                           nk_b256_vec_t target_sumsqs, nk_b256_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f64_t dist_sq = query_sumsq + target_sumsqs.f64s[i] - 2.0 * dots.f64s[i];
        results->f64s[i] = dist_sq > 0 ? nk_f64_sqrt_serial(dist_sq) : 0.0;
    }
}

/** @brief Angular from_dot for i32 accumulators: cast to f32, then same math as f32 variant. */
NK_INTERNAL void nk_angular_through_i32_from_dot_serial_(nk_b128_vec_t dots, nk_i32_t query_sumsq,
                                                         nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t product = (nk_f32_t)query_sumsq * (nk_f32_t)target_sumsqs.i32s[i];
        if (product > 0) {
            nk_f32_t rsqrt_val = nk_f32_rsqrt_serial(product);
            nk_f32_t normalized = (nk_f32_t)dots.i32s[i] * rsqrt_val;
            nk_f32_t result = 1.0f - normalized;
            results->f32s[i] = result > 0 ? result : 0;
        }
        else { results->f32s[i] = (dots.i32s[i] == 0) ? 0.0f : 1.0f; }
    }
}

/** @brief Euclidean from_dot for i32 accumulators: cast to f32, then same math as f32 variant. */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_serial_(nk_b128_vec_t dots, nk_i32_t query_sumsq,
                                                           nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dist_sq = (nk_f32_t)query_sumsq + (nk_f32_t)target_sumsqs.i32s[i] - 2.0f * (nk_f32_t)dots.i32s[i];
        results->f32s[i] = dist_sq > 0 ? nk_f32_sqrt_serial(dist_sq) : 0.0f;
    }
}

/** @brief Angular from_dot for u32 accumulators: cast to f32, then same math as f32 variant. */
NK_INTERNAL void nk_angular_through_u32_from_dot_serial_(nk_b128_vec_t dots, nk_u32_t query_sumsq,
                                                         nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t product = (nk_f32_t)query_sumsq * (nk_f32_t)target_sumsqs.u32s[i];
        if (product > 0) {
            nk_f32_t rsqrt_val = nk_f32_rsqrt_serial(product);
            nk_f32_t normalized = (nk_f32_t)dots.u32s[i] * rsqrt_val;
            nk_f32_t result = 1.0f - normalized;
            results->f32s[i] = result > 0 ? result : 0;
        }
        else { results->f32s[i] = (dots.u32s[i] == 0) ? 0.0f : 1.0f; }
    }
}

/** @brief Euclidean from_dot for u32 accumulators: cast to f32, then same math as f32 variant. */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_serial_(nk_b128_vec_t dots, nk_u32_t query_sumsq,
                                                           nk_b128_vec_t target_sumsqs, nk_b128_vec_t *results) {
    for (int i = 0; i < 4; ++i) {
        nk_f32_t dist_sq = (nk_f32_t)query_sumsq + (nk_f32_t)target_sumsqs.u32s[i] - 2.0f * (nk_f32_t)dots.u32s[i];
        results->f32s[i] = dist_sq > 0 ? nk_f32_sqrt_serial(dist_sq) : 0.0f;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SPATIAL_SERIAL_H
