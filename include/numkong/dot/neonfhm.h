/**
 *  @brief SIMD-accelerated Dot Products using FMLAL (FP16FML) for Arm NEON-capable CPUs.
 *  @file include/numkong/dot/neon_fmlal.h
 *  @sa include/numkong/dot.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  This is an experimental implementation using FMLAL (vfmlalq_low_f16/vfmlalq_high_f16)
 *  for widening fp16->f32 multiply-accumulate, vs the current approach of convert-then-FMA.
 */
#ifndef NK_DOT_NEON_FMLAL_H
#define NK_DOT_NEON_FMLAL_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL float16x8_t nk_partial_load_f16x8_fmlal_(nk_f16_t const *x, nk_size_t n) {
    nk_b128_vec_t result;
    result.u64s[0] = 0;
    result.u64s[1] = 0;
    for (nk_size_t i = 0; i < n && i < 8; ++i) result.f16s[i] = x[i];
    return vreinterpretq_f16_u16(result.u16x8);
}

NK_PUBLIC void nk_dot_f16_fmlal(nk_f16_t const *a_scalars, nk_f16_t const *b_scalars, nk_size_t count_scalars,
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
        float16x8_t a_f16x8 = nk_partial_load_f16x8_fmlal_(a_scalars + idx, remaining);
        float16x8_t b_f16x8 = nk_partial_load_f16x8_fmlal_(b_scalars + idx, remaining);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, a_f16x8, b_f16x8);
        if (remaining > 4) { sum_f32x4 = vfmlalq_high_f16(sum_f32x4, a_f16x8, b_f16x8); }
    }

    *result = vaddvq_f32(sum_f32x4);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEON_FMLAL_H
