/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dot/neonhalf.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOT_NEONHALF_H
#define NK_DOT_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

#include "numkong/types.h"
#include "numkong/reduce/neonhalf.h" // nk_partial_load_f16x4_to_f32x4_neonhalf_

#if defined(__cplusplus)
extern "C" {
#endif

NK_PUBLIC void nk_dot_f16_neonhalf(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
                                   nk_f32_t *result) {
    float32x4_t a_f32x4, b_f32x4;
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
nk_dot_f16_neonhalf_cycle:
    if (count_scalars < 4) {
        nk_partial_load_f16x4_to_f32x4_neonhalf_(a_scalars, count_scalars, &a_f32x4);
        nk_partial_load_f16x4_to_f32x4_neonhalf_(b_scalars, count_scalars, &b_f32x4);
        count_scalars = 0;
    }
    else {
        a_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)a_scalars));
        b_f32x4 = vcvt_f32_f16(vld1_f16((nk_f16_for_arm_simd_t const *)b_scalars));
        a_scalars += 4, b_scalars += 4, count_scalars -= 4;
    }
    sum_f32x4 = vfmaq_f32(sum_f32x4, a_f32x4, b_f32x4);
    if (count_scalars) goto nk_dot_f16_neonhalf_cycle;
    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_dot_f16c_neonhalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                    nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmsq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_dot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

NK_PUBLIC void nk_vdot_f16c_neonhalf(nk_f16c_t const *a_pairs, nk_f16c_t const *b_pairs, nk_size_t count_pairs,
                                     nk_f32c_t *result) {
    float32x4_t sum_real_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_imag_f32x4 = vdupq_n_f32(0);
    while (count_pairs >= 4) {
        // Unpack the input arrays into real and imaginary parts.
        // MSVC sadly doesn't recognize the `vld2_f16`, so we load the data as signed
        // integers of the same size and reinterpret with `vreinterpret_f16_s16` afterwards.
        int16x4x2_t a_i16x4x2 = vld2_s16((short *)a_pairs);
        int16x4x2_t b_i16x4x2 = vld2_s16((short *)b_pairs);
        float32x4_t a_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[0]));
        float32x4_t a_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(a_i16x4x2.val[1]));
        float32x4_t b_real_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[0]));
        float32x4_t b_imag_f32x4 = vcvt_f32_f16(vreinterpret_f16_s16(b_i16x4x2.val[1]));
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_real_f32x4, b_real_f32x4);
        sum_real_f32x4 = vfmaq_f32(sum_real_f32x4, a_imag_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmaq_f32(sum_imag_f32x4, a_real_f32x4, b_imag_f32x4);
        sum_imag_f32x4 = vfmsq_f32(sum_imag_f32x4, a_imag_f32x4, b_real_f32x4);
        count_pairs -= 4, a_pairs += 4, b_pairs += 4;
    }
    // Reduce horizontal sums and aggregate with the tail:
    nk_f32c_t tail_result;
    nk_vdot_f16c_serial(a_pairs, b_pairs, count_pairs, &tail_result);
    result->real = tail_result.real + vaddvq_f32(sum_real_f32x4);
    result->imag = tail_result.imag + vaddvq_f32(sum_imag_f32x4);
}

/**
 *  @brief Running state for 64-bit dot accumulation over f16 scalars on NEON with FP16 extension.
 *
 *  Processes 4 f16 values at a time (64 bits), converting directly to f32 without
 *  the overhead of vget_low/vget_high operations on 128-bit vectors.
 */
typedef struct nk_dot_f16x4_state_neonhalf_t {
    float32x4_t sum_f32x4;
} nk_dot_f16x4_state_neonhalf_t;

NK_INTERNAL void nk_dot_f16x4_init_neonhalf(nk_dot_f16x4_state_neonhalf_t *state) { state->sum_f32x4 = vdupq_n_f32(0); }

NK_INTERNAL void nk_dot_f16x4_update_neonhalf(nk_dot_f16x4_state_neonhalf_t *state, nk_b64_vec_t a, nk_b64_vec_t b) {
    // 4 f16s = 64 bits, direct conversion without low/high split
    float16x4_t a_f16x4 = vreinterpret_f16_u16(a.u16x4);
    float16x4_t b_f16x4 = vreinterpret_f16_u16(b.u16x4);
    state->sum_f32x4 = vfmaq_f32(state->sum_f32x4, vcvt_f32_f16(a_f16x4), vcvt_f32_f16(b_f16x4));
}

NK_INTERNAL void nk_dot_f16x4_finalize_neonhalf(                                                //
    nk_dot_f16x4_state_neonhalf_t const *state_a, nk_dot_f16x4_state_neonhalf_t const *state_b, //
    nk_dot_f16x4_state_neonhalf_t const *state_c, nk_dot_f16x4_state_neonhalf_t const *state_d, //
    nk_b128_vec_t *result) {
    float32x4_t sums_f32x4 = {vaddvq_f32(state_a->sum_f32x4), vaddvq_f32(state_b->sum_f32x4),
                              vaddvq_f32(state_c->sum_f32x4), vaddvq_f32(state_d->sum_f32x4)};
    result->u32x4 = vreinterpretq_u32_f32(sums_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEONHALF_H
