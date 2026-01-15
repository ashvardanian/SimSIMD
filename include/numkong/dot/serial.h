/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Serial (SIMD-free) CPUs.
 *  @file include/numkong/dot/serial.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_SERIAL_H
#define NK_DOT_SERIAL_H

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_f64_abs_`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Macro for dot product with simple accumulation.
 */
#define nk_define_dot_(input_type, accumulator_type, output_type, load_and_convert)                         \
    NK_PUBLIC void nk_dot_##input_type##_serial(nk_##input_type##_t const *a, nk_##input_type##_t const *b, \
                                                nk_size_t n, nk_##output_type##_t *result) {                \
        nk_##accumulator_type##_t sum = 0, ai, bi;                                                          \
        for (nk_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                   \
            load_and_convert(b + i, &bi);                                                                   \
            sum += ai * bi;                                                                                 \
        }                                                                                                   \
        *result = (nk_##output_type##_t)sum;                                                                \
    }

#define nk_define_dot_complex_(input_type, accumulator_type, output_complex_type, load_and_convert)        \
    NK_PUBLIC void nk_dot_##input_type##_serial(nk_##input_type##_t const *a_pairs,                        \
                                                nk_##input_type##_t const *b_pairs, nk_size_t count_pairs, \
                                                nk_##output_complex_type##_t *result) {                    \
        nk_##accumulator_type##_t sum_real = 0, sum_imag = 0;                                              \
        nk_##accumulator_type##_t ar, br, ai, bi;                                                          \
        for (nk_size_t i = 0; i != count_pairs; ++i) {                                                     \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                   \
            load_and_convert(&(b_pairs + i)->real, &br);                                                   \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                   \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                   \
            sum_real += ar * br - ai * bi;                                                                 \
            sum_imag += ar * bi + ai * br;                                                                 \
        }                                                                                                  \
        result->real = sum_real;                                                                           \
        result->imag = sum_imag;                                                                           \
    }

#define nk_define_vdot_complex_(input_type, accumulator_type, output_complex_type, load_and_convert)        \
    NK_PUBLIC void nk_vdot_##input_type##_serial(nk_##input_type##_t const *a_pairs,                        \
                                                 nk_##input_type##_t const *b_pairs, nk_size_t count_pairs, \
                                                 nk_##output_complex_type##_t *result) {                    \
        nk_##accumulator_type##_t sum_real = 0, sum_imag = 0;                                               \
        nk_##accumulator_type##_t ar, br, ai, bi;                                                           \
        for (nk_size_t i = 0; i != count_pairs; ++i) {                                                      \
            load_and_convert(&(a_pairs + i)->real, &ar);                                                    \
            load_and_convert(&(b_pairs + i)->real, &br);                                                    \
            load_and_convert(&(a_pairs + i)->imag, &ai);                                                    \
            load_and_convert(&(b_pairs + i)->imag, &bi);                                                    \
            sum_real += ar * br + ai * bi;                                                                  \
            sum_imag += ar * bi - ai * br;                                                                  \
        }                                                                                                   \
        result->real = sum_real;                                                                            \
        result->imag = sum_imag;                                                                            \
    }

nk_define_dot_(f32, f32, f32, nk_assign_from_to_)            // nk_dot_f32_serial
nk_define_dot_complex_(f32c, f32, f32c, nk_assign_from_to_)  // nk_dot_f32c_serial
nk_define_vdot_complex_(f32c, f32, f32c, nk_assign_from_to_) // nk_vdot_f32c_serial

nk_define_dot_(f16, f32, f32, nk_f16_to_f32_serial)            // nk_dot_f16_serial
nk_define_dot_complex_(f16c, f32, f32c, nk_f16_to_f32_serial)  // nk_dot_f16c_serial
nk_define_vdot_complex_(f16c, f32, f32c, nk_f16_to_f32_serial) // nk_vdot_f16c_serial

nk_define_dot_(bf16, f32, f32, nk_bf16_to_f32_serial)            // nk_dot_bf16_serial
nk_define_dot_complex_(bf16c, f32, f32c, nk_bf16_to_f32_serial)  // nk_dot_bf16c_serial
nk_define_vdot_complex_(bf16c, f32, f32c, nk_bf16_to_f32_serial) // nk_vdot_bf16c_serial

nk_define_dot_(i8, i32, i32, nk_assign_from_to_) // nk_dot_i8_serial
nk_define_dot_(u8, u32, u32, nk_assign_from_to_) // nk_dot_u8_serial

nk_define_dot_(e4m3, f32, f32, nk_e4m3_to_f32_serial) // nk_dot_e4m3_serial
nk_define_dot_(e5m2, f32, f32, nk_e5m2_to_f32_serial) // nk_dot_e5m2_serial

#undef nk_define_dot_
#undef nk_define_dot_complex_
#undef nk_define_vdot_complex_

NK_PUBLIC void nk_dot_i4_serial(nk_i4x2_t const *a, nk_i4x2_t const *b, nk_size_t n, nk_i32_t *result) {
    // i4 values are packed as nibbles: two 4-bit signed values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // Sign extension: (nibble ^ 8) - 8 maps [0,15] to [-8,7]
    nk_size_t n_bytes = (n + 1) / 2;
    nk_i32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        // Extract low nibbles
        nk_i32_t a_lo = (nk_i32_t)((a[i] & 0x0F) ^ 8) - 8;
        nk_i32_t b_lo = (nk_i32_t)((b[i] & 0x0F) ^ 8) - 8;
        sum += a_lo * b_lo;

        // Extract high nibbles - skip if n is odd and this is last byte
        if (2 * i + 1 < n) {
            nk_i32_t a_hi = (nk_i32_t)(((a[i] >> 4) & 0x0F) ^ 8) - 8;
            nk_i32_t b_hi = (nk_i32_t)(((b[i] >> 4) & 0x0F) ^ 8) - 8;
            sum += a_hi * b_hi;
        }
    }
    *result = sum;
}

NK_PUBLIC void nk_dot_u4_serial(nk_u4x2_t const *a, nk_u4x2_t const *b, nk_size_t n, nk_u32_t *result) {
    // u4 values are packed as nibbles: two 4-bit unsigned values per byte.
    // Parameter `n` is the number of 4-bit values (dimensions), not bytes.
    // No sign extension needed - values are ∈ [0,15].
    nk_size_t n_bytes = (n + 1) / 2;
    nk_u32_t sum = 0;
    for (nk_size_t i = 0; i < n_bytes; ++i) {
        // Extract low nibbles
        nk_u32_t a_lo = (nk_u32_t)(a[i] & 0x0F);
        nk_u32_t b_lo = (nk_u32_t)(b[i] & 0x0F);
        sum += a_lo * b_lo;

        // Extract high nibbles - skip if n is odd and this is last byte
        if (2 * i + 1 < n) {
            nk_u32_t a_hi = (nk_u32_t)((a[i] >> 4) & 0x0F);
            nk_u32_t b_hi = (nk_u32_t)((b[i] >> 4) & 0x0F);
            sum += a_hi * b_hi;
        }
    }
    *result = sum;
}

/*  Double-precision dot-produce variants
 *
 *  Implements Neumaier's improved Kahan-Babuška algorithm to minimize floating-point rounding errors.
 *  Unlike Kahan, Neumaier correctly handles the case where the term being added is larger than the
 *  running sum. Achieves O(1) error growth regardless of vector dimension.
 *
 *  Algorithm: For each term, compute t = sum + term, then:
 *    - If ‖sum‖ ≥ ‖term‖: c += (sum - t) + term  (lost low-order bits of term)
 *    - Else:              c += (term - t) + sum  (lost low-order bits of sum)
 *
 *  @see Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
 */
NK_PUBLIC void nk_dot_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_f64_t sum = 0, compensation = 0;
    for (nk_size_t i = 0; i != n; ++i) {
        nk_f64_t ai = a[i], bi = b[i];
        nk_f64_t term = ai * bi, t = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - t) + term) : ((term - t) + sum);
        sum = t;
    }
    *result = (nk_f64_t)(sum + compensation);
}

NK_PUBLIC void nk_dot_f64c_serial(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                  nk_f64c_t *result) {
    nk_f64_t sum_real = 0, sum_imag = 0, compensation_real = 0, compensation_imag = 0;
    for (nk_size_t i = 0; i != count_pairs; ++i) {
        nk_f64_t ar = a_pairs[i].real, br = b_pairs[i].real, ai = a_pairs[i].imag, bi = b_pairs[i].imag;
        nk_f64_t term_real = ar * br - ai * bi, t_real = sum_real + term_real;
        nk_f64_t term_imag = ar * bi + ai * br, t_imag = sum_imag + term_imag;
        compensation_real += (nk_f64_abs_(sum_real) >= nk_f64_abs_(term_real)) ? ((sum_real - t_real) + term_real)
                                                                               : ((term_real - t_real) + sum_real);
        compensation_imag += (nk_f64_abs_(sum_imag) >= nk_f64_abs_(term_imag)) ? ((sum_imag - t_imag) + term_imag)
                                                                               : ((term_imag - t_imag) + sum_imag);
        sum_real = t_real;
        sum_imag = t_imag;
    }
    result->real = sum_real + compensation_real;
    result->imag = sum_imag + compensation_imag;
}

NK_PUBLIC void nk_vdot_f64c_serial(nk_f64c_t const *a_pairs, nk_f64c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f64c_t *result) {
    nk_f64_t sum_real = 0, sum_imag = 0, compensation_real = 0, compensation_imag = 0;
    nk_f64_t ar, br, ai, bi;
    for (nk_size_t i = 0; i != count_pairs; ++i) {
        nk_f64_t ar = a_pairs[i].real, br = b_pairs[i].real, ai = a_pairs[i].imag, bi = b_pairs[i].imag;
        nk_f64_t term_real = ar * br + ai * bi, t_real = sum_real + term_real;
        nk_f64_t term_imag = ar * bi - ai * br, t_imag = sum_imag + term_imag;
        compensation_real += (nk_f64_abs_(sum_real) >= nk_f64_abs_(term_real)) ? ((sum_real - t_real) + term_real)
                                                                               : ((term_real - t_real) + sum_real);
        compensation_imag += (nk_f64_abs_(sum_imag) >= nk_f64_abs_(term_imag)) ? ((sum_imag - t_imag) + term_imag)
                                                                               : ((term_imag - t_imag) + sum_imag);
        sum_real = t_real;
        sum_imag = t_imag;
    }
    result->real = sum_real + compensation_real;
    result->imag = sum_imag + compensation_imag;
}

typedef struct nk_dot_f64x2_state_serial_t {
    nk_f64_t sums[2];
    nk_f64_t compensations[2];
} nk_dot_f64x2_state_serial_t;

NK_INTERNAL void nk_dot_f64x2_init_serial(nk_dot_f64x2_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
    state->compensations[0] = 0, state->compensations[1] = 0;
}

NK_INTERNAL void nk_dot_f64x2_update_serial(nk_dot_f64x2_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f64_t sum0 = state->sums[0], compensation0 = state->compensations[0];
    nk_f64_t sum1 = state->sums[1], compensation1 = state->compensations[1];

    nk_f64_t term0 = a.f64s[0] * b.f64s[0], new_sum0 = sum0 + term0;
    compensation0 += (nk_f64_abs_(sum0) >= nk_f64_abs_(term0)) ? ((sum0 - new_sum0) + term0)
                                                               : ((term0 - new_sum0) + sum0);
    sum0 = new_sum0;

    nk_f64_t term1 = a.f64s[1] * b.f64s[1], new_sum1 = sum1 + term1;
    compensation1 += (nk_f64_abs_(sum1) >= nk_f64_abs_(term1)) ? ((sum1 - new_sum1) + term1)
                                                               : ((term1 - new_sum1) + sum1);
    sum1 = new_sum1;

    state->sums[0] = sum0, state->sums[1] = sum1;
    state->compensations[0] = compensation0, state->compensations[1] = compensation1;
}

NK_INTERNAL void nk_dot_f64x2_finalize_serial(                                              //
    nk_dot_f64x2_state_serial_t const *state_a, nk_dot_f64x2_state_serial_t const *state_b, //
    nk_dot_f64x2_state_serial_t const *state_c, nk_dot_f64x2_state_serial_t const *state_d, //
    nk_b256_vec_t *result) {
    result->f64s[0] = (state_a->sums[0] + state_a->compensations[0]) + (state_a->sums[1] + state_a->compensations[1]);
    result->f64s[1] = (state_b->sums[0] + state_b->compensations[0]) + (state_b->sums[1] + state_b->compensations[1]);
    result->f64s[2] = (state_c->sums[0] + state_c->compensations[0]) + (state_c->sums[1] + state_c->compensations[1]);
    result->f64s[3] = (state_d->sums[0] + state_d->compensations[0]) + (state_d->sums[1] + state_d->compensations[1]);
}

typedef struct nk_dot_f32x4_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_f32x4_state_serial_t;

NK_INTERNAL void nk_dot_f32x4_init_serial(nk_dot_f32x4_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
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
    nk_b128_vec_t *result) {
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_f16x8_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_f16x8_state_serial_t;

NK_INTERNAL void nk_dot_f16x8_init_serial(nk_dot_f16x8_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_f16x8_update_serial(nk_dot_f16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (nk_size_t i = 0; i < 8; i += 4) {
        nk_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        nk_f16_to_f32_serial(a.f16s + i + 0, &a0), nk_f16_to_f32_serial(a.f16s + i + 1, &a1);
        nk_f16_to_f32_serial(a.f16s + i + 2, &a2), nk_f16_to_f32_serial(a.f16s + i + 3, &a3);
        nk_f16_to_f32_serial(b.f16s + i + 0, &b0), nk_f16_to_f32_serial(b.f16s + i + 1, &b1);
        nk_f16_to_f32_serial(b.f16s + i + 2, &b2), nk_f16_to_f32_serial(b.f16s + i + 3, &b3);
        sum0 += a0 * b0, sum1 += a1 * b1, sum2 += a2 * b2, sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_f16x8_finalize_serial(                                              //
    nk_dot_f16x8_state_serial_t const *state_a, nk_dot_f16x8_state_serial_t const *state_b, //
    nk_dot_f16x8_state_serial_t const *state_c, nk_dot_f16x8_state_serial_t const *state_d, //
    nk_b128_vec_t *result) {
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_bf16x8_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_bf16x8_state_serial_t;

NK_INTERNAL void nk_dot_bf16x8_init_serial(nk_dot_bf16x8_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_bf16x8_update_serial(nk_dot_bf16x8_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0], sum1 = state->sums[1], sum2 = state->sums[2], sum3 = state->sums[3];
    for (nk_size_t i = 0; i < 8; i += 4) {
        nk_f32_t a0, a1, a2, a3, b0, b1, b2, b3;
        nk_bf16_to_f32_serial(a.bf16s + i + 0, &a0), nk_bf16_to_f32_serial(a.bf16s + i + 1, &a1);
        nk_bf16_to_f32_serial(a.bf16s + i + 2, &a2), nk_bf16_to_f32_serial(a.bf16s + i + 3, &a3);
        nk_bf16_to_f32_serial(b.bf16s + i + 0, &b0), nk_bf16_to_f32_serial(b.bf16s + i + 1, &b1);
        nk_bf16_to_f32_serial(b.bf16s + i + 2, &b2), nk_bf16_to_f32_serial(b.bf16s + i + 3, &b3);
        sum0 += a0 * b0, sum1 += a1 * b1, sum2 += a2 * b2, sum3 += a3 * b3;
    }
    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_bf16x8_finalize_serial(                                               //
    nk_dot_bf16x8_state_serial_t const *state_a, nk_dot_bf16x8_state_serial_t const *state_b, //
    nk_dot_bf16x8_state_serial_t const *state_c, nk_dot_bf16x8_state_serial_t const *state_d, //
    nk_b128_vec_t *result) {
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_i8x16_state_serial_t {
    nk_i64_t sums[2];
} nk_dot_i8x16_state_serial_t;

NK_INTERNAL void nk_dot_i8x16_init_serial(nk_dot_i8x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
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
    nk_b128_vec_t *result) {
    result->i32s[0] = (nk_i32_t)(state_a->sums[0] + state_a->sums[1]);
    result->i32s[1] = (nk_i32_t)(state_b->sums[0] + state_b->sums[1]);
    result->i32s[2] = (nk_i32_t)(state_c->sums[0] + state_c->sums[1]);
    result->i32s[3] = (nk_i32_t)(state_d->sums[0] + state_d->sums[1]);
}

typedef struct nk_dot_u8x16_state_serial_t {
    nk_u64_t sums[2];
} nk_dot_u8x16_state_serial_t;

NK_INTERNAL void nk_dot_u8x16_init_serial(nk_dot_u8x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0;
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
    nk_b128_vec_t *result) {
    result->u32s[0] = (nk_u32_t)(state_a->sums[0] + state_a->sums[1]);
    result->u32s[1] = (nk_u32_t)(state_b->sums[0] + state_b->sums[1]);
    result->u32s[2] = (nk_u32_t)(state_c->sums[0] + state_c->sums[1]);
    result->u32s[3] = (nk_u32_t)(state_d->sums[0] + state_d->sums[1]);
}

typedef struct nk_dot_e4m3x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e4m3x16_state_serial_t;

NK_INTERNAL void nk_dot_e4m3x16_init_serial(nk_dot_e4m3x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e4m3x16_update_serial(nk_dot_e4m3x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e4m3_to_f32_serial(a.e4m3s + i, &ai0), nk_e4m3_to_f32_serial(b.e4m3s + i, &bi0);
        nk_e4m3_to_f32_serial(a.e4m3s + i + 1, &ai1), nk_e4m3_to_f32_serial(b.e4m3s + i + 1, &bi1);
        nk_e4m3_to_f32_serial(a.e4m3s + i + 2, &ai2), nk_e4m3_to_f32_serial(b.e4m3s + i + 2, &bi2);
        nk_e4m3_to_f32_serial(a.e4m3s + i + 3, &ai3), nk_e4m3_to_f32_serial(b.e4m3s + i + 3, &bi3);
        sum0 += ai0 * bi0, sum1 += ai1 * bi1, sum2 += ai2 * bi2, sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e4m3x16_finalize_serial(                                                //
    nk_dot_e4m3x16_state_serial_t const *state_a, nk_dot_e4m3x16_state_serial_t const *state_b, //
    nk_dot_e4m3x16_state_serial_t const *state_c, nk_dot_e4m3x16_state_serial_t const *state_d, //
    nk_b128_vec_t *result) {
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

typedef struct nk_dot_e5m2x16_state_serial_t {
    nk_f32_t sums[4];
} nk_dot_e5m2x16_state_serial_t;

NK_INTERNAL void nk_dot_e5m2x16_init_serial(nk_dot_e5m2x16_state_serial_t *state) {
    state->sums[0] = 0, state->sums[1] = 0, state->sums[2] = 0, state->sums[3] = 0;
}

NK_INTERNAL void nk_dot_e5m2x16_update_serial(nk_dot_e5m2x16_state_serial_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    nk_f32_t sum0 = state->sums[0];
    nk_f32_t sum1 = state->sums[1];
    nk_f32_t sum2 = state->sums[2];
    nk_f32_t sum3 = state->sums[3];
    nk_f32_t ai0, ai1, ai2, ai3;
    nk_f32_t bi0, bi1, bi2, bi3;
    for (nk_size_t i = 0; i != 16; i += 4) {
        nk_e5m2_to_f32_serial(a.e5m2s + i, &ai0), nk_e5m2_to_f32_serial(b.e5m2s + i, &bi0);
        nk_e5m2_to_f32_serial(a.e5m2s + i + 1, &ai1), nk_e5m2_to_f32_serial(b.e5m2s + i + 1, &bi1);
        nk_e5m2_to_f32_serial(a.e5m2s + i + 2, &ai2), nk_e5m2_to_f32_serial(b.e5m2s + i + 2, &bi2);
        nk_e5m2_to_f32_serial(a.e5m2s + i + 3, &ai3), nk_e5m2_to_f32_serial(b.e5m2s + i + 3, &bi3);
        sum0 += ai0 * bi0, sum1 += ai1 * bi1, sum2 += ai2 * bi2, sum3 += ai3 * bi3;
    }

    state->sums[0] = sum0, state->sums[1] = sum1, state->sums[2] = sum2, state->sums[3] = sum3;
}

NK_INTERNAL void nk_dot_e5m2x16_finalize_serial(                                                //
    nk_dot_e5m2x16_state_serial_t const *state_a, nk_dot_e5m2x16_state_serial_t const *state_b, //
    nk_dot_e5m2x16_state_serial_t const *state_c, nk_dot_e5m2x16_state_serial_t const *state_d, //
    nk_b128_vec_t *result) {
    result->f32s[0] = state_a->sums[0] + state_a->sums[1] + state_a->sums[2] + state_a->sums[3];
    result->f32s[1] = state_b->sums[0] + state_b->sums[1] + state_b->sums[2] + state_b->sums[3];
    result->f32s[2] = state_c->sums[0] + state_c->sums[1] + state_c->sums[2] + state_c->sums[3];
    result->f32s[3] = state_d->sums[0] + state_d->sums[1] + state_d->sums[2] + state_d->sums[3];
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_DOT_SERIAL_H
