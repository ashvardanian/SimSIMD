/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for SIMD-free CPUs.
 *  @file include/numkong/dot/serial.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SERIAL_H
#define NK_DOT_SERIAL_H
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for dot product with Kahan compensated summation.
 *
 *  Implements Kahan-Babuška algorithm to minimize floating-point rounding errors.
 *  Achieves O(1) error growth regardless of vector dimension, compared to O(√n) for naive summation.
 *
 *  @see NK_MAKE_L2SQ in spatial.h for detailed documentation on Kahan summation.
 */
#define NK_MAKE_DOT(name, input_type, accumulator_type, output_type, load_and_convert)                      \
    NK_PUBLIC void nk_dot_##input_type##_##name(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t ab = 0, compensation = 0, ai, bi;                                         \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            nk_##accumulator_type##_t term = ai * bi;                                                       \
            nk_##accumulator_type##_t y = term - compensation;                                              \
            nk_##accumulator_type##_t t = ab + y;                                                           \
            compensation = (t - ab) - y;                                                                    \
            ab = t;                                                                                         \
        }                                                                                                   \
        *result = (nk_##output_type##_t)ab;                                                                 \
    }

#define NK_MAKE_COMPLEX_DOT(name, input_type, accumulator_type, output_complex_type, load_and_convert)     \
    NK_PUBLIC void nk_dot_##input_type##_##name(nk_##input_type##_t const *a_pairs,                        \
                                                nk_##input_type##_t const *b_pairs, nk_size_t count_pairs, \
                                                nk_##output_complex_type##_t *result) {                    \
        nk_##accumulator_type##_t ab_real = 0, ab_imag = 0, ar, br, ai, bi;                                \
        for (nk_size_t i = 0; i != count_pairs; ++i) {                                                     \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                   \
            load_and_convert(&(b_pairs + i)->real, &br);                                                   \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                   \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                   \
            ab_real += ar * br - ai * bi;                                                                  \
            ab_imag += ar * bi + ai * br;                                                                  \
        }                                                                                                  \
        result->real = ab_real;                                                                            \
        result->imag = ab_imag;                                                                            \
    }

#define NK_MAKE_COMPLEX_VDOT(name, input_type, accumulator_type, output_complex_type, load_and_convert)     \
    NK_PUBLIC void nk_vdot_##input_type##_##name(nk_##input_type##_t const *a_pairs,                        \
                                                 nk_##input_type##_t const *b_pairs, nk_size_t count_pairs, \
                                                 nk_##output_complex_type##_t *result) {                    \
        nk_##accumulator_type##_t ab_real = 0, ab_imag = 0, ar, br, ai, bi;                                 \
        for (nk_size_t i = 0; i != count_pairs; ++i) {                                                      \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                    \
            load_and_convert(&(b_pairs + i)->real, &br);                                                    \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                    \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                    \
            ab_real += ar * br + ai * bi;                                                                   \
            ab_imag += ar * bi - ai * br;                                                                   \
        }                                                                                                   \
        result->real = ab_real;                                                                             \
        result->imag = ab_imag;                                                                             \
    }

NK_MAKE_DOT(serial, f64, f64, f64, NK_ASSIGN_FROM_TO)            // nk_dot_f64_serial
NK_MAKE_COMPLEX_DOT(serial, f64c, f64, f64c, NK_ASSIGN_FROM_TO)  // nk_dot_f64c_serial
NK_MAKE_COMPLEX_VDOT(serial, f64c, f64, f64c, NK_ASSIGN_FROM_TO) // nk_vdot_f64c_serial

NK_MAKE_DOT(serial, f32, f32, f32, NK_ASSIGN_FROM_TO)            // nk_dot_f32_serial
NK_MAKE_COMPLEX_DOT(serial, f32c, f32, f32c, NK_ASSIGN_FROM_TO)  // nk_dot_f32c_serial
NK_MAKE_COMPLEX_VDOT(serial, f32c, f32, f32c, NK_ASSIGN_FROM_TO) // nk_vdot_f32c_serial

NK_MAKE_DOT(serial, f16, f32, f32, nk_f16_to_f32)            // nk_dot_f16_serial
NK_MAKE_COMPLEX_DOT(serial, f16c, f32, f32c, nk_f16_to_f32)  // nk_dot_f16c_serial
NK_MAKE_COMPLEX_VDOT(serial, f16c, f32, f32c, nk_f16_to_f32) // nk_vdot_f16c_serial

NK_MAKE_DOT(serial, bf16, f32, f32, nk_bf16_to_f32)            // nk_dot_bf16_serial
NK_MAKE_COMPLEX_DOT(serial, bf16c, f32, f32c, nk_bf16_to_f32)  // nk_dot_bf16c_serial
NK_MAKE_COMPLEX_VDOT(serial, bf16c, f32, f32c, nk_bf16_to_f32) // nk_vdot_bf16c_serial

NK_MAKE_DOT(serial, i8, i64, i32, NK_ASSIGN_FROM_TO) // nk_dot_i8_serial
NK_MAKE_DOT(serial, u8, u64, u32, NK_ASSIGN_FROM_TO) // nk_dot_u8_serial

NK_MAKE_DOT(serial, e4m3, f32, f32, nk_e4m3_to_f32) // nk_dot_e4m3_serial
NK_MAKE_DOT(serial, e5m2, f32, f32, nk_e5m2_to_f32) // nk_dot_e5m2_serial

NK_MAKE_DOT(accurate, f32, f64, f64, NK_ASSIGN_FROM_TO)            // nk_dot_f32_accurate
NK_MAKE_COMPLEX_DOT(accurate, f32c, f64, f64c, NK_ASSIGN_FROM_TO)  // nk_dot_f32c_accurate
NK_MAKE_COMPLEX_VDOT(accurate, f32c, f64, f64c, NK_ASSIGN_FROM_TO) // nk_vdot_f32c_accurate

NK_MAKE_DOT(accurate, f16, f64, f64, nk_f16_to_f64)            // nk_dot_f16_accurate
NK_MAKE_COMPLEX_DOT(accurate, f16c, f64, f64c, nk_f16_to_f64)  // nk_dot_f16c_accurate
NK_MAKE_COMPLEX_VDOT(accurate, f16c, f64, f64c, nk_f16_to_f64) // nk_vdot_f16c_accurate

NK_MAKE_DOT(accurate, bf16, f64, f64, nk_bf16_to_f64)            // nk_dot_bf16_accurate
NK_MAKE_COMPLEX_DOT(accurate, bf16c, f64, f64c, nk_bf16_to_f64)  // nk_dot_bf16c_accurate
NK_MAKE_COMPLEX_VDOT(accurate, bf16c, f64, f64c, nk_bf16_to_f64) // nk_vdot_bf16c_accurate

typedef struct nk_dot_f64x2_state_serial_t {
    nk_f64_t sums[2];
} nk_dot_f64x2_state_serial_t;

NK_INTERNAL void nk_dot_f64x2_init_serial(nk_dot_f64x2_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_f64x2_update_serial(nk_dot_f64x2_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f64_t sum0 = state->sums[0];
    nk_f64_t sum1 = state->sums[1];

    sum0 += a.f64s[0] * b.f64s[0], sum1 += a.f64s[1] * b.f64s[1];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

NK_INTERNAL void nk_dot_f64x2_finalize_serial(                                              //
    nk_dot_f64x2_state_serial_t const *state_a, nk_dot_f64x2_state_serial_t const *state_b, //
    nk_dot_f64x2_state_serial_t const *state_c, nk_dot_f64x2_state_serial_t const *state_d, //
    nk_f64_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1];
    results[1] = state_b->sums[0] + state_b->sums[1];
    results[2] = state_c->sums[0] + state_c->sums[1];
    results[3] = state_d->sums[0] + state_d->sums[1];
}

typedef struct nk_dot_f32x4_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_f32x4_state_serial_t;

NK_INTERNAL void nk_dot_f32x4_init_serial(nk_dot_f32x4_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_f32x4_update_serial(nk_dot_f32x4_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];

    sum0 += a.f32s[0] * b.f32s[0], sum1 += a.f32s[1] * b.f32s[1];
    sum2 += a.f32s[2] * b.f32s[2], sum3 += a.f32s[3] * b.f32s[3];

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_f32x4_finalize_serial(                                              //
    nk_dot_f32x4_state_serial_t const *state_a, nk_dot_f32x4_state_serial_t const *state_b, //
    nk_dot_f32x4_state_serial_t const *state_c, nk_dot_f32x4_state_serial_t const *state_d, //
    nk_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_f16x8_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_f16x8_state_serial_t;

NK_INTERNAL void nk_dot_f16x8_init_serial(nk_dot_f16x8_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_f16x8_update_serial(nk_dot_f16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (nk_size_t i = 0; i < 8; i += 4) {
        nk_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        nk_f16_to_f32(a.f16s + i + 0, &a0), nk_f16_to_f32(a.f16s + i + 1, &a1);
        nk_f16_to_f32(a.f16s + i + 2, &a2), nk_f16_to_f32(a.f16s + i + 3, &a3);
        nk_f16_to_f32(b.f16s + i + 0, &b0), nk_f16_to_f32(b.f16s + i + 1, &b1);
        nk_f16_to_f32(b.f16s + i + 2, &b2), nk_f16_to_f32(b.f16s + i + 3, &b3);
        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_f16x8_finalize_serial(                                              //
    nk_dot_f16x8_state_serial_t const *state_a, nk_dot_f16x8_state_serial_t const *state_b, //
    nk_dot_f16x8_state_serial_t const *state_c, nk_dot_f16x8_state_serial_t const *state_d, //
    nk_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_bf16x8_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_bf16x8_state_serial_t;

NK_INTERNAL void nk_dot_bf16x8_init_serial(nk_dot_bf16x8_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_bf16x8_update_serial(nk_dot_bf16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (nk_size_t i = 0; i < 8; i += 4) {
        nk_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        nk_bf16_to_f32(a.bf16s + i + 0, &a0), nk_bf16_to_f32(a.bf16s + i + 1, &a1);
        nk_bf16_to_f32(a.bf16s + i + 2, &a2), nk_bf16_to_f32(a.bf16s + i + 3, &a3);
        nk_bf16_to_f32(b.bf16s + i + 0, &b0), nk_bf16_to_f32(b.bf16s + i + 1, &b1);
        nk_bf16_to_f32(b.bf16s + i + 2, &b2), nk_bf16_to_f32(b.bf16s + i + 3, &b3);
        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_bf16x8_finalize_serial(                                               //
    nk_dot_bf16x8_state_serial_t const *state_a, nk_dot_bf16x8_state_serial_t const *state_b, //
    nk_dot_bf16x8_state_serial_t const *state_c, nk_dot_bf16x8_state_serial_t const *state_d, //
    nk_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_i8x16_state_serial_t {
    nk_i64_t sums[2];
} nk_dot_i8x16_state_serial_t;

NK_INTERNAL void nk_dot_i8x16_init_serial(nk_dot_i8x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_i8x16_update_serial(nk_dot_i8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_i64_t sum0 = state->sums[0];
    nk_i64_t sum1 = state->sums[1];

    sum0 += (nk_i16_t)a.i8s[0] * (nk_i16_t)b.i8s[0], sum1 += (nk_i16_t)a.i8s[1] * (nk_i16_t)b.i8s[1];
    sum0 += (nk_i16_t)a.i8s[2] * (nk_i16_t)b.i8s[2], sum1 += (nk_i16_t)a.i8s[3] * (nk_i16_t)b.i8s[3];
    sum0 += (nk_i16_t)a.i8s[4] * (nk_i16_t)b.i8s[4], sum1 += (nk_i16_t)a.i8s[5] * (nk_i16_t)b.i8s[5];
    sum0 += (nk_i16_t)a.i8s[6] * (nk_i16_t)b.i8s[6], sum1 += (nk_i16_t)a.i8s[7] * (nk_i16_t)b.i8s[7];
    sum0 += (nk_i16_t)a.i8s[8] * (nk_i16_t)b.i8s[8], sum1 += (nk_i16_t)a.i8s[9] * (nk_i16_t)b.i8s[9];
    sum0 += (nk_i16_t)a.i8s[10] * (nk_i16_t)b.i8s[10], sum1 += (nk_i16_t)a.i8s[11] * (nk_i16_t)b.i8s[11];
    sum0 += (nk_i16_t)a.i8s[12] * (nk_i16_t)b.i8s[12], sum1 += (nk_i16_t)a.i8s[13] * (nk_i16_t)b.i8s[13];
    sum0 += (nk_i16_t)a.i8s[14] * (nk_i16_t)b.i8s[14], sum1 += (nk_i16_t)a.i8s[15] * (nk_i16_t)b.i8s[15];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

NK_INTERNAL void nk_dot_i8x16_finalize_serial(                                              //
    nk_dot_i8x16_state_serial_t const *state_a, nk_dot_i8x16_state_serial_t const *state_b, //
    nk_dot_i8x16_state_serial_t const *state_c, nk_dot_i8x16_state_serial_t const *state_d, //
    nk_i32_t *results) {
    results[0] = (nk_i32_t)(state_a->sums[0] + state_a->sums[1]);
    results[1] = (nk_i32_t)(state_b->sums[0] + state_b->sums[1]);
    results[2] = (nk_i32_t)(state_c->sums[0] + state_c->sums[1]);
    results[3] = (nk_i32_t)(state_d->sums[0] + state_d->sums[1]);
}

typedef struct nk_dot_u8x16_state_serial_t {
    nk_u64_t sums[2];
} nk_dot_u8x16_state_serial_t;

NK_INTERNAL void nk_dot_u8x16_init_serial(nk_dot_u8x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
}

NK_INTERNAL void nk_dot_u8x16_update_serial(nk_dot_u8x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_u64_t sum0 = state->sums[0];
    nk_u64_t sum1 = state->sums[1];

    sum0 += (nk_u16_t)a.u8s[0] * (nk_u16_t)b.u8s[0], sum1 += (nk_u16_t)a.u8s[1] * (nk_u16_t)b.u8s[1];
    sum0 += (nk_u16_t)a.u8s[2] * (nk_u16_t)b.u8s[2], sum1 += (nk_u16_t)a.u8s[3] * (nk_u16_t)b.u8s[3];
    sum0 += (nk_u16_t)a.u8s[4] * (nk_u16_t)b.u8s[4], sum1 += (nk_u16_t)a.u8s[5] * (nk_u16_t)b.u8s[5];
    sum0 += (nk_u16_t)a.u8s[6] * (nk_u16_t)b.u8s[6], sum1 += (nk_u16_t)a.u8s[7] * (nk_u16_t)b.u8s[7];
    sum0 += (nk_u16_t)a.u8s[8] * (nk_u16_t)b.u8s[8], sum1 += (nk_u16_t)a.u8s[9] * (nk_u16_t)b.u8s[9];
    sum0 += (nk_u16_t)a.u8s[10] * (nk_u16_t)b.u8s[10], sum1 += (nk_u16_t)a.u8s[11] * (nk_u16_t)b.u8s[11];
    sum0 += (nk_u16_t)a.u8s[12] * (nk_u16_t)b.u8s[12], sum1 += (nk_u16_t)a.u8s[13] * (nk_u16_t)b.u8s[13];
    sum0 += (nk_u16_t)a.u8s[14] * (nk_u16_t)b.u8s[14], sum1 += (nk_u16_t)a.u8s[15] * (nk_u16_t)b.u8s[15];

    state->sums[0] = sum0, state->sums[1] = sum1;
}

NK_INTERNAL void nk_dot_u8x16_finalize_serial(                                              //
    nk_dot_u8x16_state_serial_t const *state_a, nk_dot_u8x16_state_serial_t const *state_b, //
    nk_dot_u8x16_state_serial_t const *state_c, nk_dot_u8x16_state_serial_t const *state_d, //
    nk_u32_t *results) {
    results[0] = (nk_u32_t)(state_a->sums[0] + state_a->sums[1]);
    results[1] = (nk_u32_t)(state_b->sums[0] + state_b->sums[1]);
    results[2] = (nk_u32_t)(state_c->sums[0] + state_c->sums[1]);
    results[3] = (nk_u32_t)(state_d->sums[0] + state_d->sums[1]);
}

typedef struct nk_dot_e4m3x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e4m3x16_state_serial_t;

NK_INTERNAL void nk_dot_e4m3x16_init_serial(nk_dot_e4m3x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e4m3x16_update_serial(nk_dot_e4m3x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e4m3_to_f32(a.e4m3s + i, &ai0), nk_e4m3_to_f32(b.e4m3s + i, &bi0);
        nk_e4m3_to_f32(a.e4m3s + i + 1, &ai1), nk_e4m3_to_f32(b.e4m3s + i + 1, &bi1);
        nk_e4m3_to_f32(a.e4m3s + i + 2, &ai2), nk_e4m3_to_f32(b.e4m3s + i + 2, &bi2);
        nk_e4m3_to_f32(a.e4m3s + i + 3, &ai3), nk_e4m3_to_f32(b.e4m3s + i + 3, &bi3);
        sum0 += ai0 * bi0;
        sum1 += ai1 * bi1;
        sum2 += ai2 * bi2;
        sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e4m3x16_finalize_serial(                                                //
    nk_dot_e4m3x16_state_serial_t const *state_a, nk_dot_e4m3x16_state_serial_t const *state_b, //
    nk_dot_e4m3x16_state_serial_t const *state_c, nk_dot_e4m3x16_state_serial_t const *state_d, //
    nk_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_e5m2x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e5m2x16_state_serial_t;

NK_INTERNAL void nk_dot_e5m2x16_init_serial(nk_dot_e5m2x16_state_serial_t *state) {
    state->sums[0] = 0;
    state->sums[1] = 0;
    state->sums[2] = 0;
    state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e5m2x16_update_serial(nk_dot_e5m2x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e5m2_to_f32(a.e5m2s + i, &ai0), nk_e5m2_to_f32(b.e5m2s + i, &bi0);
        nk_e5m2_to_f32(a.e5m2s + i + 1, &ai1), nk_e5m2_to_f32(b.e5m2s + i + 1, &bi1);
        nk_e5m2_to_f32(a.e5m2s + i + 2, &ai2), nk_e5m2_to_f32(b.e5m2s + i + 2, &bi2);
        nk_e5m2_to_f32(a.e5m2s + i + 3, &ai3), nk_e5m2_to_f32(b.e5m2s + i + 3, &bi3);
        sum0 += ai0 * bi0;
        sum1 += ai1 * bi1;
        sum2 += ai2 * bi2;
        sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e5m2x16_finalize_serial(                                                //
    nk_dot_e5m2x16_state_serial_t const *state_a, nk_dot_e5m2x16_state_serial_t const *state_b, //
    nk_dot_e5m2x16_state_serial_t const *state_c, nk_dot_e5m2x16_state_serial_t const *state_d, //
    nk_f32_t *results) {
    results[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    results[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    results[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    results[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_DOT_SERIAL_H