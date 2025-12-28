/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dot/neon_bf16.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_NEON_BF16_H
#define NK_DOT_NEON_BF16_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL bfloat16x8_t nk_partial_load_bf16x8_neon_(nk_bf16_t const *x, nk_size_t n) {
    nk_b128_vec_t result;
    result.u32x4 = vdupq_n_u32(0);
    nk_size_t i = 0;
    for (; i < n; ++i) result.bf16s[i] = x[i];
    return vreinterpretq_bf16_u32(result.u32x4);
}

NK_PUBLIC void nk_dot_bf16_neon(nk_bf16_t const *a_scalars, nk_bf16_t const *b_scalars, nk_size_t count_scalars,
                                nk_f32_t *result) {
    bfloat16x8_t a_bf16x8, b_bf16x8;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_bf16_neon_cycle:
    if (count_scalars < 8) {
        a_bf16x8 = nk_partial_load_bf16x8_neon_(a_scalars, count_scalars);
        b_bf16x8 = nk_partial_load_bf16x8_neon_(b_scalars, count_scalars);
        count_scalars = 0;
    }
    else {
        a_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)a_scalars);
        b_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)b_scalars);
        a_scalars += 8, b_scalars += 8, count_scalars -= 8;
    }
    sum_f32x4 = vbfdotq_f32(sum_f32x4, a_bf16x8, b_bf16x8);
    if (count_scalars) goto nk_dot_bf16_neon_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_bf16c_neon(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                 nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_dot_bf16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

NK_PUBLIC void nk_vdot_bf16c_neon(nk_bf16c_t const *a_pairs, nk_bf16c_t const *b_pairs, nk_size_t count_pairs,
                                  nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_bf16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_bf16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short const *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short const *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_bf16(vreinterpret_bf16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_vdot_bf16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

/**
 *  @brief Running state for 128-bit dot accumulation over bf16 scalars on NEON.
 */
typedef struct nk_dot_bf16x8_state_neon_t {
    float32x4_t sum_f32x4;
} nk_dot_bf16x8_state_neon_t;

NK_INTERNAL void nk_dot_bf16x8_init_neon(nk_dot_bf16x8_state_neon_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_bf16x8_update_neon(nk_dot_bf16x8_state_neon_t *state, nk_b128_vec_t a, nk_b128_vec_t b) {
    float32x4_t sum_f32x4 = state->sum_f32x4;
    nk_bf16_t const *a_scalars = a.bf16s;
    nk_bf16_t const *b_scalars = b.bf16s;
    sum_f32x4 = vbfdotq_f32(sum_f32x4, vld1q_bf16((nk_bf16_for_arm_simd_t const *)(a_scalars + 0)),
                            vld1q_bf16((nk_bf16_for_arm_simd_t const *)(b_scalars + 0)));
    state->sum_f32x4 = sum_f32x4;
}

NK_INTERNAL void nk_dot_bf16x8_finalize_neon(                                             //
    nk_dot_bf16x8_state_neon_t const *state_a, nk_dot_bf16x8_state_neon_t const *state_b, //
    nk_dot_bf16x8_state_neon_t const *state_c, nk_dot_bf16x8_state_neon_t const *state_d, //
    nk_f32_t *results) {
    results[0] = vaddvq_f32(state_a->sum_f32x4);
    results[1] = vaddvq_f32(state_b->sum_f32x4);
    results[2] = vaddvq_f32(state_c->sum_f32x4);
    results[3] = vaddvq_f32(state_d->sum_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_BF16
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEON_BF16_H