/**
 *  @brief SIMD-accelerated Dot Products using FMLAL (FEAT_FHM) for Arm NEON-capable CPUs.
 *  @file include/numkong/dot/neonfhm.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  This implementation uses FMLAL (vfmlalq_low_f16/vfmlalq_high_f16) for widening
 *  fp16->f32 multiply-accumulate, which is 20-48% faster than convert-then-FMA.
 */
#ifndef NK_DOT_NEONFHM_H
#define NK_DOT_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b8x8_serial_`
#include "numkong/cast/neon.h"   // `nk_e4m3x8_to_f16x8_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f16_neonfhm(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                  nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    // Main loop: process 8 elements at a time using FMLAL
    for (; idx + 8 <= count_scalars; idx += 8) {
        float16x8_t a_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(a_scalars + idx));
        float16x8_t b_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(b_scalars + idx));
        // FMLAL: widening multiply-accumulate fp16->f32
        // low: processes elements 0-3, high: processes elements 4-7
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    }

    // Handle remaining elements (0-7)
    if (idx < count_scalars) {
        nk_size_t remaining = count_scalars - idx;
        nk_b128_vec_t a_vec, b_vec;
        nk_partial_load_b16x8_serial_(a_scalars + idx, &a_vec, remaining);
        nk_partial_load_b16x8_serial_(b_scalars + idx, &b_vec, remaining);
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8);
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    }

    *result = vaddvq_f32(sum_f32x4);
}

// State-based API for GEMM integration
typedef struct nk_dot_f16x8_state_neonfhm_t {
    float32x4_t sum_f32x4;
} nk_dot_f16x8_state_neonfhm_t;

NK_INTERNAL void nk_dot_f16x8_init_neonfhm(nk_dot_f16x8_state_neonfhm_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_f16x8_update_neonfhm(nk_dot_f16x8_state_neonfhm_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    float16x8_t a_f16x8 = vreinterpretq_f16_u16(a.u16x8);
    float16x8_t b_f16x8 = vreinterpretq_f16_u16(b.u16x8);
    // FMLAL: widening multiply-accumulate fp16->f32 (faster than convert-then-FMA)
    state->sum_f32x4 = vfmlalq_low_f16(state->sum_f32x4, a_f16x8, b_f16x8);
    state->sum_f32x4 = vfmlalq_high_f16(state->sum_f32x4, a_f16x8, b_f16x8);
}

NK_INTERNAL void nk_dot_f16x8_finalize_neonfhm(                                               //
    nk_dot_f16x8_state_neonfhm_t const *state_a, nk_dot_f16x8_state_neonfhm_t const *state_b, //
    nk_dot_f16x8_state_neonfhm_t const *state_c, nk_dot_f16x8_state_neonfhm_t const *state_d, //
    nk_b128_vec_t *result) {
    float32x4_t sums = {vaddvq_f32(state_a->sum_f32x4), vaddvq_f32(state_b->sum_f32x4), vaddvq_f32(state_c->sum_f32x4),
                        vaddvq_f32(state_d->sum_f32x4)};
    result->u32x4 = vreinterpretq_u32_f32(sums);
}

/**
 *  @brief Complex dot product using FMLAL/FMLSL for widening fp16->f32 operations.
 *  Computes: result = Σ(a[i] * b[i]) where * is complex multiplication.
 *  Real: Σ(a.re * b.re - a.im * b.im)
 *  Imag: Σ(a.re * b.im + a.im * b.re)
 */
NK_PUBLIC void nk_dot_f16c_neonfhm(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                   nk_f32c_t *result) {
    // Accumulate into 4 float32x2_t vectors (low/high for real/imag)
    float32x2_t sum_real_lo = vdup_n_f32(0);
    float32x2_t sum_real_hi = vdup_n_f32(0);
    float32x2_t sum_imag_lo = vdup_n_f32(0);
    float32x2_t sum_imag_hi = vdup_n_f32(0);

    while (count_pairs >= 4) {
        // Load and deinterleave: vld2 loads 4 complex pairs as 2 x float16x4_t
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);

        float16x4_t a_real = vreinterpret_f16_s16(a_i16x4x2.val[0]);
        float16x4_t a_imag = vreinterpret_f16_s16(a_i16x4x2.val[1]);
        float16x4_t b_real = vreinterpret_f16_s16(b_i16x4x2.val[0]);
        float16x4_t b_imag = vreinterpret_f16_s16(b_i16x4x2.val[1]);

        // Real: a.re * b.re - a.im * b.im (FMLAL then FMLSL)
        sum_real_lo = vfmlal_low_f16(sum_real_lo, a_real, b_real);
        sum_real_lo = vfmlsl_low_f16(sum_real_lo, a_imag, b_imag);
        sum_real_hi = vfmlal_high_f16(sum_real_hi, a_real, b_real);
        sum_real_hi = vfmlsl_high_f16(sum_real_hi, a_imag, b_imag);

        // Imag: a.re * b.im + a.im * b.re (FMLAL for both)
        sum_imag_lo = vfmlal_low_f16(sum_imag_lo, a_real, b_imag);
        sum_imag_lo = vfmlal_low_f16(sum_imag_lo, a_imag, b_real);
        sum_imag_hi = vfmlal_high_f16(sum_imag_hi, a_real, b_imag);
        sum_imag_hi = vfmlal_high_f16(sum_imag_hi, a_imag, b_real);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Combine and reduce
    float32x4_t sum_real_f32x4 = vcombine_f32(sum_real_lo, sum_real_hi);
    float32x4_t sum_imag_f32x4 = vcombine_f32(sum_imag_lo, sum_imag_hi);

    // Handle tail with serial fallback
    nk_f32c_t tail_result;
    nk_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = vaddvq_f32(sum_real_f32x4) + tail_result.real;
    result->imag = vaddvq_f32(sum_imag_f32x4) + tail_result.imag;
}

/**
 *  @brief Complex conjugate dot product using FMLAL/FMLSL for widening fp16->f32 operations.
 *  Computes: result = Σ(a[i] * conj(b[i])) where * is complex multiplication.
 *  Real: Σ(a.re * b.re + a.im * b.im)
 *  Imag: Σ(a.im * b.re - a.re * b.im)
 */
NK_PUBLIC void nk_vdot_f16c_neonfhm(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    // Accumulate into 4 float32x2_t vectors (low/high for real/imag)
    float32x2_t sum_real_lo = vdup_n_f32(0);
    float32x2_t sum_real_hi = vdup_n_f32(0);
    float32x2_t sum_imag_lo = vdup_n_f32(0);
    float32x2_t sum_imag_hi = vdup_n_f32(0);

    while (count_pairs >= 4) {
        // Load and deinterleave: vld2 loads 4 complex pairs as 2 x float16x4_t
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);

        float16x4_t a_real = vreinterpret_f16_s16(a_i16x4x2.val[0]);
        float16x4_t a_imag = vreinterpret_f16_s16(a_i16x4x2.val[1]);
        float16x4_t b_real = vreinterpret_f16_s16(b_i16x4x2.val[0]);
        float16x4_t b_imag = vreinterpret_f16_s16(b_i16x4x2.val[1]);

        // Real: a.re * b.re + a.im * b.im (FMLAL for both)
        sum_real_lo = vfmlal_low_f16(sum_real_lo, a_real, b_real);
        sum_real_lo = vfmlal_low_f16(sum_real_lo, a_imag, b_imag);
        sum_real_hi = vfmlal_high_f16(sum_real_hi, a_real, b_real);
        sum_real_hi = vfmlal_high_f16(sum_real_hi, a_imag, b_imag);

        // Imag: a.im * b.re - a.re * b.im (FMLAL then FMLSL)
        sum_imag_lo = vfmlal_low_f16(sum_imag_lo, a_imag, b_real);
        sum_imag_lo = vfmlsl_low_f16(sum_imag_lo, a_real, b_imag);
        sum_imag_hi = vfmlal_high_f16(sum_imag_hi, a_imag, b_real);
        sum_imag_hi = vfmlsl_high_f16(sum_imag_hi, a_real, b_imag);

        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }

    // Combine and reduce
    float32x4_t sum_real_f32x4 = vcombine_f32(sum_real_lo, sum_real_hi);
    float32x4_t sum_imag_f32x4 = vcombine_f32(sum_imag_lo, sum_imag_hi);

    // Handle tail with serial fallback
    nk_f32c_t tail_result;
    nk_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = vaddvq_f32(sum_real_f32x4) + tail_result.real;
    result->imag = vaddvq_f32(sum_imag_f32x4) + tail_result.imag;
}

NK_PUBLIC void nk_dot_e4m3_neonfhm(nk_e4m3_t const *a_scalars, nk_e4m3_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    // Main loop: process 8 elements at a time using FMLAL
    for (; idx + 8 <= count_scalars; idx += 8) {
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(a_scalars + idx));
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(b_scalars + idx));
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    }

    // Handle remaining elements (0-7)
    if (idx < count_scalars) {
        nk_size_t remaining = count_scalars - idx;
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars + idx, &a_vec, remaining);
        nk_partial_load_b8x8_serial_(b_scalars + idx, &b_vec, remaining);
        float16x8_t a_f16x8 = nk_e4m3x8_to_f16x8_neon_(a_vec.u8x8);
        float16x8_t b_f16x8 = nk_e4m3x8_to_f16x8_neon_(b_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    }

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_e5m2_neonfhm(nk_e5m2_t const *a_scalars, nk_e5m2_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    // Main loop: process 8 elements at a time using FMLAL
    for (; idx + 8 <= count_scalars; idx += 8) {
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(a_scalars + idx));
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(b_scalars + idx));
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    }

    // Handle remaining elements (0-7)
    if (idx < count_scalars) {
        nk_size_t remaining = count_scalars - idx;
        nk_b64_vec_t a_vec, b_vec;
        nk_partial_load_b8x8_serial_(a_scalars + idx, &a_vec, remaining);
        nk_partial_load_b8x8_serial_(b_scalars + idx, &b_vec, remaining);
        float16x8_t a_f16x8 = nk_e5m2x8_to_f16x8_neon_(a_vec.u8x8);
        float16x8_t b_f16x8 = nk_e5m2x8_to_f16x8_neon_(b_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8);
    }

    *result = vaddvq_f32(sum_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEONFHM_H
