/**
 *  @brief SIMD-accelerated Spatial Similarity Measures optimized for SIMD-free CPUs.
 *  @file include/numkong/spatial/serial.h
 *  @sa include/numkong/spatial.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_SPATIAL_SERIAL_H
#define NK_SPATIAL_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for L2 squared distance with Kahan compensated summation.
 *
 *  Implements Kahan-Babuška algorithm to minimize floating-point rounding errors during accumulation.
 *  Achieves O(1) error growth regardless of vector dimension, compared to O(√n) for naive summation.
 *
 *  Performance vs Accuracy Tradeoff:
 *  - Adds ~30% overhead (3 extra FP operations per iteration) compared to naive summation
 *  - Reduces relative error from ~10^-5 to ~10^-7 at n=100K for f32
 *  - Benefits all floating-point types: f64, f32, f16, bf16
 *  - Integer types (i8) maintain perfect accuracy regardless
 *
 *  Error Growth Comparison (measured on serial implementations, compared to f64 baseline):
 *  @code
 *  Type    Dimension    Naive (O(√n))    Kahan (O(1))     Improvement
 *  f32     1K           1.78e-07         2.37e-08         7.5x
 *  f32     10K          5.63e-07         2.16e-08         26x
 *  f32     100K         7.08e-06         2.32e-08         305x
 *  f32     1M           ~7.0e-05 (proj)  2.21e-08         ~3,167x
 *
 *  bf16    1K           4.86e-07         (expected ~1e-07)
 *  bf16    100K         1.56e-05         (expected ~1e-07) 156x expected
 *
 *  f16     1K           2.38e-07         (expected ~5e-08)
 *  f16     100K         4.68e-06         (expected ~5e-08) 94x expected
 *  @endcode
 *
 *  Algorithm (Kahan-Babuška Compensated Summation):
 *  - Maintain compensation term 'c' to capture floating-point rounding errors
 *  - Each iteration: y = term - c (corrected input), t = sum + y (tentative sum)
 *  - Update: c = (t - sum) - y (new compensation), sum = t
 *  - The compensation captures the low-order bits lost during addition
 *
 *  References:
 *  - Kahan, W. (1965). "Further remarks on reducing truncation errors"
 *  - https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 *
 *  @param name         Variant name (e.g., serial, accurate)
 *  @param input_type   Input element type (f32, f16, bf16, f64, i8, etc.)
 *  @param accumulator_type Type used for accumulation
 *  @param output_type  Result type
 *  @param load_and_convert Conversion macro for input elements
 */
#define NK_MAKE_L2SQ(name, input_type, accumulator_type, output_type, load_and_convert)                      \
    NK_PUBLIC void nk_l2sq_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                 nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t distance_sq = 0, compensation = 0, a_element, b_element;                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                 \
            load_and_convert(a + i, &a_element);                                                             \
            load_and_convert(b + i, &b_element);                                                             \
            nk_##accumulator_type##_t diff = a_element - b_element;                                          \
            nk_##accumulator_type##_t term = diff * diff;                                                    \
            /* Kahan compensated summation: */                                                               \
            nk_##accumulator_type##_t y = term - compensation; /* Subtract previous compensation */          \
            nk_##accumulator_type##_t t = distance_sq + y;     /* Add corrected term */                      \
            compensation = (t - distance_sq) - y;              /* Update compensation for next iteration */  \
            distance_sq = t;                                                                                 \
        }                                                                                                    \
        *result = (nk_##output_type##_t)distance_sq;                                                         \
    }

#define NK_MAKE_L2(name, input_type, accumulator_type, l2sq_output_type, output_type, load_and_convert, compute_sqrt) \
    NK_PUBLIC void nk_l2_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b,            \
                                               nk_size_t n, nk_##output_type##_t *result) {                           \
        nk_##l2sq_output_type##_t distance_sq;                                                                        \
        nk_l2sq_##input_type##_##name(a, b, n, &distance_sq);                                                         \
        *result = compute_sqrt((nk_##output_type##_t)distance_sq);                                                    \
    }

/**
 *  @brief Macro for cosine/angular distance with Kahan compensated summation.
 *
 *  Uses Kahan summation for all three accumulators (dot_product, a_norm_sq, b_norm_sq).
 *  Achieves O(1) error growth regardless of vector dimension.
 *
 *  @see NK_MAKE_L2SQ for detailed documentation on Kahan summation.
 */
#define NK_MAKE_COS(name, input_type, accumulator_type, output_type, load_and_convert, compute_rsqrt)           \
    NK_PUBLIC void nk_angular_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                    nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t dot_product = 0, a_norm_sq = 0, b_norm_sq = 0, a_element, b_element;          \
        nk_##accumulator_type##_t c_dot = 0, c_a = 0, c_b = 0; /* Kahan compensation terms */                   \
        for (nk_size_t i = 0; i != n; ++i) {                                                                    \
            load_and_convert(a + i, &a_element);                                                                \
            load_and_convert(b + i, &b_element);                                                                \
            /* Kahan for dot_product */                                                                         \
            nk_##accumulator_type##_t y_dot = a_element * b_element - c_dot;                                    \
            nk_##accumulator_type##_t t_dot = dot_product + y_dot;                                              \
            c_dot = (t_dot - dot_product) - y_dot;                                                              \
            dot_product = t_dot;                                                                                \
            /* Kahan for a_norm_sq */                                                                           \
            nk_##accumulator_type##_t y_a = a_element * a_element - c_a;                                        \
            nk_##accumulator_type##_t t_a = a_norm_sq + y_a;                                                    \
            c_a = (t_a - a_norm_sq) - y_a;                                                                      \
            a_norm_sq = t_a;                                                                                    \
            /* Kahan for b_norm_sq */                                                                           \
            nk_##accumulator_type##_t y_b = b_element * b_element - c_b;                                        \
            nk_##accumulator_type##_t t_b = b_norm_sq + y_b;                                                    \
            c_b = (t_b - b_norm_sq) - y_b;                                                                      \
            b_norm_sq = t_b;                                                                                    \
        }                                                                                                       \
        if (a_norm_sq == 0 && b_norm_sq == 0) { *result = 0; }                                                  \
        else if (dot_product == 0) { *result = 1; }                                                             \
        else {                                                                                                  \
            nk_##output_type##_t unclipped_distance = 1 - dot_product * compute_rsqrt(a_norm_sq) *              \
                                                              compute_rsqrt(b_norm_sq);                         \
            *result = unclipped_distance > 0 ? unclipped_distance : 0;                                          \
        }                                                                                                       \
    }

NK_MAKE_COS(serial, f64, f64, f64, nk_assign_from_to_, NK_F64_RSQRT)    // nk_angular_f64_serial
NK_MAKE_L2SQ(serial, f64, f64, f64, nk_assign_from_to_)                 // nk_l2sq_f64_serial
NK_MAKE_L2(serial, f64, f64, f64, f64, nk_assign_from_to_, NK_F64_SQRT) // nk_l2_f64_serial

NK_MAKE_COS(serial, f32, f32, f32, nk_assign_from_to_, NK_F32_RSQRT)    // nk_angular_f32_serial
NK_MAKE_L2SQ(serial, f32, f32, f32, nk_assign_from_to_)                 // nk_l2sq_f32_serial
NK_MAKE_L2(serial, f32, f32, f32, f32, nk_assign_from_to_, NK_F32_SQRT) // nk_l2_f32_serial

NK_MAKE_COS(serial, f16, f32, f32, nk_f16_to_f32, NK_F32_RSQRT)    // nk_angular_f16_serial
NK_MAKE_L2SQ(serial, f16, f32, f32, nk_f16_to_f32)                 // nk_l2sq_f16_serial
NK_MAKE_L2(serial, f16, f32, f32, f32, nk_f16_to_f32, NK_F32_SQRT) // nk_l2_f16_serial

NK_MAKE_COS(serial, bf16, f32, f32, nk_bf16_to_f32, NK_F32_RSQRT)    // nk_angular_bf16_serial
NK_MAKE_L2SQ(serial, bf16, f32, f32, nk_bf16_to_f32)                 // nk_l2sq_bf16_serial
NK_MAKE_L2(serial, bf16, f32, f32, f32, nk_bf16_to_f32, NK_F32_SQRT) // nk_l2_bf16_serial

NK_MAKE_COS(serial, i8, i32, f32, nk_assign_from_to_, NK_F32_RSQRT)      // nk_angular_i8_serial
NK_MAKE_L2SQ(serial, i8, i32, u32, nk_assign_from_to_)                   // nk_l2sq_i8_serial
NK_MAKE_L2SQ(accurate, i8, i32, u32, nk_assign_from_to_)                 // nk_l2sq_i8_accurate
NK_MAKE_L2(serial, i8, i32, u32, f32, nk_assign_from_to_, NK_F32_SQRT)   // nk_l2_i8_serial
NK_MAKE_L2(accurate, i8, i32, u32, f64, nk_assign_from_to_, NK_F64_SQRT) // nk_l2_i8_accurate

NK_MAKE_COS(serial, u8, i32, f32, nk_assign_from_to_, NK_F32_RSQRT)      // nk_angular_u8_serial
NK_MAKE_L2SQ(serial, u8, i32, u32, nk_assign_from_to_)                   // nk_l2sq_u8_serial
NK_MAKE_L2SQ(accurate, u8, i32, u32, nk_assign_from_to_)                 // nk_l2sq_u8_accurate
NK_MAKE_L2(serial, u8, i32, u32, f32, nk_assign_from_to_, NK_F32_SQRT)   // nk_l2_u8_serial
NK_MAKE_L2(accurate, u8, i32, u32, f64, nk_assign_from_to_, NK_F64_SQRT) // nk_l2_u8_accurate

NK_MAKE_COS(accurate, f32, f64, f64, nk_assign_from_to_, NK_F64_RSQRT)    // nk_angular_f32_accurate
NK_MAKE_L2SQ(accurate, f32, f64, f64, nk_assign_from_to_)                 // nk_l2sq_f32_accurate
NK_MAKE_L2(accurate, f32, f64, f64, f64, nk_assign_from_to_, NK_F64_SQRT) // nk_l2_f32_accurate

NK_MAKE_COS(accurate, f16, f64, f64, nk_f16_to_f64, NK_F64_RSQRT)    // nk_angular_f16_accurate
NK_MAKE_L2SQ(accurate, f16, f64, f64, nk_f16_to_f64)                 // nk_l2sq_f16_accurate
NK_MAKE_L2(accurate, f16, f64, f64, f64, nk_f16_to_f64, NK_F64_SQRT) // nk_l2_f16_accurate

NK_MAKE_COS(accurate, bf16, f64, f64, nk_bf16_to_f64, NK_F64_RSQRT)    // nk_angular_bf16_accurate
NK_MAKE_L2SQ(accurate, bf16, f64, f64, nk_bf16_to_f64)                 // nk_l2sq_bf16_accurate
NK_MAKE_L2(accurate, bf16, f64, f64, f64, nk_bf16_to_f64, NK_F64_SQRT) // nk_l2_bf16_accurate

typedef nk_dot_f64x2_state_serial_t nk_angular_f64x2_state_serial_t;
NK_INTERNAL void nk_angular_f64x2_init_serial(nk_angular_f64x2_state_serial_t *state) {
    nk_dot_f64x2_init_serial(state);
}
NK_INTERNAL void nk_angular_f64x2_update_serial(nk_angular_f64x2_state_serial_t *state, nk_b128_vec_t a,
                                                nk_b128_vec_t b) {
    nk_dot_f64x2_update_serial(state, a, b);
}
NK_INTERNAL void nk_angular_f64x2_finalize_serial(nk_angular_f64x2_state_serial_t const *state_a,
                                                  nk_angular_f64x2_state_serial_t const *state_b,
                                                  nk_angular_f64x2_state_serial_t const *state_c,
                                                  nk_angular_f64x2_state_serial_t const *state_d, nk_f64_t query_norm,
                                                  nk_f64_t target_norm_a, nk_f64_t target_norm_b,
                                                  nk_f64_t target_norm_c, nk_f64_t target_norm_d, nk_f64_t *results) {
    nk_f64_t dot_product_a = state_a->sums[0] + state_a->sums[1];
    nk_f64_t dot_product_b = state_b->sums[0] + state_b->sums[1];
    nk_f64_t dot_product_c = state_c->sums[0] + state_c->sums[1];
    nk_f64_t dot_product_d = state_d->sums[0] + state_d->sums[1];

    nk_f64_t query_norm_sq = query_norm * query_norm;
    nk_f64_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f64_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f64_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f64_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Precompute query rsqrt once (was computed 4x before)
    nk_f64_t query_rsqrt = query_norm_sq > 0 ? NK_F64_RSQRT(query_norm_sq) : 0;
    nk_f64_t target_rsqrt_a = target_norm_sq_a > 0 ? NK_F64_RSQRT(target_norm_sq_a) : 0;
    nk_f64_t target_rsqrt_b = target_norm_sq_b > 0 ? NK_F64_RSQRT(target_norm_sq_b) : 0;
    nk_f64_t target_rsqrt_c = target_norm_sq_c > 0 ? NK_F64_RSQRT(target_norm_sq_c) : 0;
    nk_f64_t target_rsqrt_d = target_norm_sq_d > 0 ? NK_F64_RSQRT(target_norm_sq_d) : 0;

    nk_f64_t unclipped_distance_a = 1 - dot_product_a * query_rsqrt * target_rsqrt_a;
    nk_f64_t unclipped_distance_b = 1 - dot_product_b * query_rsqrt * target_rsqrt_b;
    nk_f64_t unclipped_distance_c = 1 - dot_product_c * query_rsqrt * target_rsqrt_c;
    nk_f64_t unclipped_distance_d = 1 - dot_product_d * query_rsqrt * target_rsqrt_d;

    results[0] = (query_norm_sq == 0 && target_norm_sq_a == 0) ? 0
                                                               : (unclipped_distance_a > 0 ? unclipped_distance_a : 0);
    results[1] = (query_norm_sq == 0 && target_norm_sq_b == 0) ? 0
                                                               : (unclipped_distance_b > 0 ? unclipped_distance_b : 0);
    results[2] = (query_norm_sq == 0 && target_norm_sq_c == 0) ? 0
                                                               : (unclipped_distance_c > 0 ? unclipped_distance_c : 0);
    results[3] = (query_norm_sq == 0 && target_norm_sq_d == 0) ? 0
                                                               : (unclipped_distance_d > 0 ? unclipped_distance_d : 0);
}

typedef nk_dot_f64x2_state_serial_t nk_l2_f64x2_state_serial_t;
NK_INTERNAL void nk_l2_f64x2_init_serial(nk_l2_f64x2_state_serial_t *state) { nk_dot_f64x2_init_serial(state); }
NK_INTERNAL void nk_l2_f64x2_update_serial(nk_l2_f64x2_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_f64x2_update_serial(state, a, b);
}
NK_INTERNAL void nk_l2_f64x2_finalize_serial(nk_l2_f64x2_state_serial_t const *state_a,
                                             nk_l2_f64x2_state_serial_t const *state_b,
                                             nk_l2_f64x2_state_serial_t const *state_c,
                                             nk_l2_f64x2_state_serial_t const *state_d, nk_f64_t query_norm,
                                             nk_f64_t target_norm_a, nk_f64_t target_norm_b, nk_f64_t target_norm_c,
                                             nk_f64_t target_norm_d, nk_f64_t *results) {
    nk_f64_t dot_product_a = state_a->sums[0] + state_a->sums[1];
    nk_f64_t dot_product_b = state_b->sums[0] + state_b->sums[1];
    nk_f64_t dot_product_c = state_c->sums[0] + state_c->sums[1];
    nk_f64_t dot_product_d = state_d->sums[0] + state_d->sums[1];

    nk_f64_t query_norm_sq = query_norm * query_norm;
    nk_f64_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f64_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f64_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f64_t target_norm_sq_d = target_norm_d * target_norm_d;

    nk_f64_t distance_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_product_a;
    nk_f64_t distance_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_product_b;
    nk_f64_t distance_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_product_c;
    nk_f64_t distance_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_product_d;

    results[0] = distance_sq_a > 0 ? NK_F64_SQRT(distance_sq_a) : 0;
    results[1] = distance_sq_b > 0 ? NK_F64_SQRT(distance_sq_b) : 0;
    results[2] = distance_sq_c > 0 ? NK_F64_SQRT(distance_sq_c) : 0;
    results[3] = distance_sq_d > 0 ? NK_F64_SQRT(distance_sq_d) : 0;
}

typedef nk_dot_f32x4_state_serial_t nk_angular_f32x4_state_serial_t;
NK_INTERNAL void nk_angular_f32x4_init_serial(nk_angular_f32x4_state_serial_t *state) {
    nk_dot_f32x4_init_serial(state);
}
NK_INTERNAL void nk_angular_f32x4_update_serial(nk_angular_f32x4_state_serial_t *state, nk_b128_vec_t a,
                                                nk_b128_vec_t b) {
    nk_dot_f32x4_update_serial(state, a, b);
}
NK_INTERNAL void nk_angular_f32x4_finalize_serial(nk_angular_f32x4_state_serial_t const *state_a,
                                                  nk_angular_f32x4_state_serial_t const *state_b,
                                                  nk_angular_f32x4_state_serial_t const *state_c,
                                                  nk_angular_f32x4_state_serial_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_f32_t dots[4];
    nk_dot_f32x4_finalize_serial(state_a, state_b, state_c, state_d, dots);

    nk_f32_t dot_product_a = dots[0], dot_product_b = dots[1];
    nk_f32_t dot_product_c = dots[2], dot_product_d = dots[3];

    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Precompute query rsqrt once (was computed 4x before)
    nk_f32_t query_rsqrt = query_norm_sq > 0 ? NK_F32_RSQRT(query_norm_sq) : 0;
    nk_f32_t target_rsqrt_a = target_norm_sq_a > 0 ? NK_F32_RSQRT(target_norm_sq_a) : 0;
    nk_f32_t target_rsqrt_b = target_norm_sq_b > 0 ? NK_F32_RSQRT(target_norm_sq_b) : 0;
    nk_f32_t target_rsqrt_c = target_norm_sq_c > 0 ? NK_F32_RSQRT(target_norm_sq_c) : 0;
    nk_f32_t target_rsqrt_d = target_norm_sq_d > 0 ? NK_F32_RSQRT(target_norm_sq_d) : 0;

    nk_f32_t unclipped_distance_a = 1 - dot_product_a * query_rsqrt * target_rsqrt_a;
    nk_f32_t unclipped_distance_b = 1 - dot_product_b * query_rsqrt * target_rsqrt_b;
    nk_f32_t unclipped_distance_c = 1 - dot_product_c * query_rsqrt * target_rsqrt_c;
    nk_f32_t unclipped_distance_d = 1 - dot_product_d * query_rsqrt * target_rsqrt_d;

    results[0] = (query_norm_sq == 0 && target_norm_sq_a == 0) ? 0
                                                               : (unclipped_distance_a > 0 ? unclipped_distance_a : 0);
    results[1] = (query_norm_sq == 0 && target_norm_sq_b == 0) ? 0
                                                               : (unclipped_distance_b > 0 ? unclipped_distance_b : 0);
    results[2] = (query_norm_sq == 0 && target_norm_sq_c == 0) ? 0
                                                               : (unclipped_distance_c > 0 ? unclipped_distance_c : 0);
    results[3] = (query_norm_sq == 0 && target_norm_sq_d == 0) ? 0
                                                               : (unclipped_distance_d > 0 ? unclipped_distance_d : 0);
}

typedef nk_dot_f32x4_state_serial_t nk_l2_f32x4_state_serial_t;
NK_INTERNAL void nk_l2_f32x4_init_serial(nk_l2_f32x4_state_serial_t *state) { nk_dot_f32x4_init_serial(state); }
NK_INTERNAL void nk_l2_f32x4_update_serial(nk_l2_f32x4_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_f32x4_update_serial(state, a, b);
}
NK_INTERNAL void nk_l2_f32x4_finalize_serial(nk_l2_f32x4_state_serial_t const *state_a,
                                             nk_l2_f32x4_state_serial_t const *state_b,
                                             nk_l2_f32x4_state_serial_t const *state_c,
                                             nk_l2_f32x4_state_serial_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_f32_t *results) {
    nk_f32_t dots[4];
    nk_dot_f32x4_finalize_serial(state_a, state_b, state_c, state_d, dots);

    nk_f32_t dot_product_a = dots[0], dot_product_b = dots[1];
    nk_f32_t dot_product_c = dots[2], dot_product_d = dots[3];

    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    nk_f32_t distance_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_product_a;
    nk_f32_t distance_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_product_b;
    nk_f32_t distance_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_product_c;
    nk_f32_t distance_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_product_d;

    results[0] = distance_sq_a > 0 ? NK_F32_SQRT(distance_sq_a) : 0;
    results[1] = distance_sq_b > 0 ? NK_F32_SQRT(distance_sq_b) : 0;
    results[2] = distance_sq_c > 0 ? NK_F32_SQRT(distance_sq_c) : 0;
    results[3] = distance_sq_d > 0 ? NK_F32_SQRT(distance_sq_d) : 0;
}

typedef nk_dot_f16x8_state_serial_t nk_angular_f16x8_state_serial_t;
NK_INTERNAL void nk_angular_f16x8_init_serial(nk_angular_f16x8_state_serial_t *state) {
    nk_dot_f16x8_init_serial(state);
}
NK_INTERNAL void nk_angular_f16x8_update_serial(nk_angular_f16x8_state_serial_t *state, nk_b128_vec_t a,
                                                nk_b128_vec_t b) {
    nk_dot_f16x8_update_serial(state, a, b);
}
NK_INTERNAL void nk_angular_f16x8_finalize_serial(nk_angular_f16x8_state_serial_t const *state_a,
                                                  nk_angular_f16x8_state_serial_t const *state_b,
                                                  nk_angular_f16x8_state_serial_t const *state_c,
                                                  nk_angular_f16x8_state_serial_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    nk_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    nk_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    nk_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    nk_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared norms (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    nk_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0) results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0) results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0) results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0) results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef nk_dot_f16x8_state_serial_t nk_l2_f16x8_state_serial_t;
NK_INTERNAL void nk_l2_f16x8_init_serial(nk_l2_f16x8_state_serial_t *state) { nk_dot_f16x8_init_serial(state); }
NK_INTERNAL void nk_l2_f16x8_update_serial(nk_l2_f16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_f16x8_update_serial(state, a, b);
}
NK_INTERNAL void nk_l2_f16x8_finalize_serial(nk_l2_f16x8_state_serial_t const *state_a,
                                             nk_l2_f16x8_state_serial_t const *state_b,
                                             nk_l2_f16x8_state_serial_t const *state_c,
                                             nk_l2_f16x8_state_serial_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    nk_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    nk_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    nk_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    nk_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared distances (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    nk_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    nk_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    nk_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    nk_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? NK_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? NK_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? NK_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? NK_F32_SQRT(dist_sq_d) : 0;
}

typedef nk_dot_bf16x8_state_serial_t nk_angular_bf16x8_state_serial_t;
NK_INTERNAL void nk_angular_bf16x8_init_serial(nk_angular_bf16x8_state_serial_t *state) {
    nk_dot_bf16x8_init_serial(state);
}
NK_INTERNAL void nk_angular_bf16x8_update_serial(nk_angular_bf16x8_state_serial_t *state, nk_b128_vec_t a,
                                                 nk_b128_vec_t b) {
    nk_dot_bf16x8_update_serial(state, a, b);
}
NK_INTERNAL void nk_angular_bf16x8_finalize_serial(nk_angular_bf16x8_state_serial_t const *state_a,
                                                   nk_angular_bf16x8_state_serial_t const *state_b,
                                                   nk_angular_bf16x8_state_serial_t const *state_c,
                                                   nk_angular_bf16x8_state_serial_t const *state_d, nk_f32_t query_norm,
                                                   nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                   nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    nk_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    nk_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    nk_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    nk_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared norms (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    nk_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0) results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0) results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0) results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0) results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef nk_dot_bf16x8_state_serial_t nk_l2_bf16x8_state_serial_t;
NK_INTERNAL void nk_l2_bf16x8_init_serial(nk_l2_bf16x8_state_serial_t *state) { nk_dot_bf16x8_init_serial(state); }
NK_INTERNAL void nk_l2_bf16x8_update_serial(nk_l2_bf16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_bf16x8_update_serial(state, a, b);
}
NK_INTERNAL void nk_l2_bf16x8_finalize_serial(nk_l2_bf16x8_state_serial_t const *state_a,
                                              nk_l2_bf16x8_state_serial_t const *state_b,
                                              nk_l2_bf16x8_state_serial_t const *state_c,
                                              nk_l2_bf16x8_state_serial_t const *state_d, nk_f32_t query_norm,
                                              nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                              nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 4 partial sums
    nk_f32_t dot_a = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    nk_f32_t dot_b = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    nk_f32_t dot_c = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    nk_f32_t dot_d = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];

    // Compute squared distances (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    nk_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    nk_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    nk_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    nk_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? NK_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? NK_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? NK_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? NK_F32_SQRT(dist_sq_d) : 0;
}

typedef nk_dot_i8x16_state_serial_t nk_angular_i8x16_state_serial_t;
NK_INTERNAL void nk_angular_i8x16_init_serial(nk_angular_i8x16_state_serial_t *state) {
    nk_dot_i8x16_init_serial(state);
}
NK_INTERNAL void nk_angular_i8x16_update_serial(nk_angular_i8x16_state_serial_t *state, nk_b128_vec_t a,
                                                nk_b128_vec_t b) {
    nk_dot_i8x16_update_serial(state, a, b);
}
NK_INTERNAL void nk_angular_i8x16_finalize_serial(nk_angular_i8x16_state_serial_t const *state_a,
                                                  nk_angular_i8x16_state_serial_t const *state_b,
                                                  nk_angular_i8x16_state_serial_t const *state_c,
                                                  nk_angular_i8x16_state_serial_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    nk_i64_t dot_a = state_a->sums[0] + state_a->sums[1];
    nk_i64_t dot_b = state_b->sums[0] + state_b->sums[1];
    nk_i64_t dot_c = state_c->sums[0] + state_c->sums[1];
    nk_i64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared norms (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    nk_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0) results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0) results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0) results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0) results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef nk_dot_i8x16_state_serial_t nk_l2_i8x16_state_serial_t;
NK_INTERNAL void nk_l2_i8x16_init_serial(nk_l2_i8x16_state_serial_t *state) { nk_dot_i8x16_init_serial(state); }
NK_INTERNAL void nk_l2_i8x16_update_serial(nk_l2_i8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_i8x16_update_serial(state, a, b);
}
NK_INTERNAL void nk_l2_i8x16_finalize_serial(nk_l2_i8x16_state_serial_t const *state_a,
                                             nk_l2_i8x16_state_serial_t const *state_b,
                                             nk_l2_i8x16_state_serial_t const *state_c,
                                             nk_l2_i8x16_state_serial_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    nk_i64_t dot_a = state_a->sums[0] + state_a->sums[1];
    nk_i64_t dot_b = state_b->sums[0] + state_b->sums[1];
    nk_i64_t dot_c = state_c->sums[0] + state_c->sums[1];
    nk_i64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared distances (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    nk_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    nk_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    nk_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    nk_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? NK_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? NK_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? NK_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? NK_F32_SQRT(dist_sq_d) : 0;
}

typedef nk_dot_u8x16_state_serial_t nk_angular_u8x16_state_serial_t;
NK_INTERNAL void nk_angular_u8x16_init_serial(nk_angular_u8x16_state_serial_t *state) {
    nk_dot_u8x16_init_serial(state);
}
NK_INTERNAL void nk_angular_u8x16_update_serial(nk_angular_u8x16_state_serial_t *state, nk_b128_vec_t a,
                                                nk_b128_vec_t b) {
    nk_dot_u8x16_update_serial(state, a, b);
}
NK_INTERNAL void nk_angular_u8x16_finalize_serial(nk_angular_u8x16_state_serial_t const *state_a,
                                                  nk_angular_u8x16_state_serial_t const *state_b,
                                                  nk_angular_u8x16_state_serial_t const *state_c,
                                                  nk_angular_u8x16_state_serial_t const *state_d, nk_f32_t query_norm,
                                                  nk_f32_t target_norm_a, nk_f32_t target_norm_b,
                                                  nk_f32_t target_norm_c, nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    nk_u64_t dot_a = state_a->sums[0] + state_a->sums[1];
    nk_u64_t dot_b = state_b->sums[0] + state_b->sums[1];
    nk_u64_t dot_c = state_c->sums[0] + state_c->sums[1];
    nk_u64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared norms (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    // Compute angular distances (loop-unrolled)
    nk_f32_t unclipped_a, unclipped_b, unclipped_c, unclipped_d;

    if (query_norm_sq == 0 && target_norm_sq_a == 0) results[0] = 0;
    else if (dot_a == 0) results[0] = 1;
    else {
        unclipped_a = 1 - dot_a * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_a);
        results[0] = unclipped_a > 0 ? unclipped_a : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_b == 0) results[1] = 0;
    else if (dot_b == 0) results[1] = 1;
    else {
        unclipped_b = 1 - dot_b * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_b);
        results[1] = unclipped_b > 0 ? unclipped_b : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_c == 0) results[2] = 0;
    else if (dot_c == 0) results[2] = 1;
    else {
        unclipped_c = 1 - dot_c * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_c);
        results[2] = unclipped_c > 0 ? unclipped_c : 0;
    }

    if (query_norm_sq == 0 && target_norm_sq_d == 0) results[3] = 0;
    else if (dot_d == 0) results[3] = 1;
    else {
        unclipped_d = 1 - dot_d * NK_F32_RSQRT(query_norm_sq) * NK_F32_RSQRT(target_norm_sq_d);
        results[3] = unclipped_d > 0 ? unclipped_d : 0;
    }
}

typedef nk_dot_u8x16_state_serial_t nk_l2_u8x16_state_serial_t;
NK_INTERNAL void nk_l2_u8x16_init_serial(nk_l2_u8x16_state_serial_t *state) { nk_dot_u8x16_init_serial(state); }
NK_INTERNAL void nk_l2_u8x16_update_serial(nk_l2_u8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_dot_u8x16_update_serial(state, a, b);
}
NK_INTERNAL void nk_l2_u8x16_finalize_serial(nk_l2_u8x16_state_serial_t const *state_a,
                                             nk_l2_u8x16_state_serial_t const *state_b,
                                             nk_l2_u8x16_state_serial_t const *state_c,
                                             nk_l2_u8x16_state_serial_t const *state_d, nk_f32_t query_norm,
                                             nk_f32_t target_norm_a, nk_f32_t target_norm_b, nk_f32_t target_norm_c,
                                             nk_f32_t target_norm_d, nk_f32_t *results) {
    // Extract dots from states by summing the 2 partial sums
    nk_u64_t dot_a = state_a->sums[0] + state_a->sums[1];
    nk_u64_t dot_b = state_b->sums[0] + state_b->sums[1];
    nk_u64_t dot_c = state_c->sums[0] + state_c->sums[1];
    nk_u64_t dot_d = state_d->sums[0] + state_d->sums[1];

    // Compute squared distances (loop-unrolled)
    nk_f32_t query_norm_sq = query_norm * query_norm;
    nk_f32_t target_norm_sq_a = target_norm_a * target_norm_a;
    nk_f32_t target_norm_sq_b = target_norm_b * target_norm_b;
    nk_f32_t target_norm_sq_c = target_norm_c * target_norm_c;
    nk_f32_t target_norm_sq_d = target_norm_d * target_norm_d;

    nk_f32_t dist_sq_a = query_norm_sq + target_norm_sq_a - 2 * dot_a;
    nk_f32_t dist_sq_b = query_norm_sq + target_norm_sq_b - 2 * dot_b;
    nk_f32_t dist_sq_c = query_norm_sq + target_norm_sq_c - 2 * dot_c;
    nk_f32_t dist_sq_d = query_norm_sq + target_norm_sq_d - 2 * dot_d;

    // Final sqrt (loop-unrolled)
    results[0] = dist_sq_a > 0 ? NK_F32_SQRT(dist_sq_a) : 0;
    results[1] = dist_sq_b > 0 ? NK_F32_SQRT(dist_sq_b) : 0;
    results[2] = dist_sq_c > 0 ? NK_F32_SQRT(dist_sq_c) : 0;
    results[3] = dist_sq_d > 0 ? NK_F32_SQRT(dist_sq_d) : 0;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SPATIAL_SERIAL_H