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
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/neon.h" // nk_partial_load_b16x8_neon_

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
        nk_partial_load_b16x8_neon_(a_scalars + idx, remaining, &a_vec);
        nk_partial_load_b16x8_neon_(b_scalars + idx, remaining, &b_vec);
        float16x8_t a_f16x8 = vreinterpretq_f16_u16(a_vec.u16x8);
        float16x8_t b_f16x8 = vreinterpretq_f16_u16(b_vec.u16x8);
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
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#endif // NK_DOT_NEONFHM_H
