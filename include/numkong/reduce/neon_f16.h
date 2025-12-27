/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neon_f16.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEON_F16_H
#define NK_REDUCE_NEON_F16_H

#if _NK_TARGET_ARM
#if NK_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Horizontal sum of 8 f16s in a NEON register, returning f32. */
NK_INTERNAL nk_f32_t _nk_reduce_add_f16x8_neon(float16x8_t sum_f16x8) {
    float16x4_t low_f16x4 = vget_low_f16(sum_f16x8);
    float16x4_t high_f16x4 = vget_high_f16(sum_f16x8);
    float16x4_t sum_f16x4 = vadd_f16(low_f16x4, high_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    return vgetq_lane_f32(vcvt_f32_f16(sum_f16x4), 0);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_F16
#endif // _NK_TARGET_ARM

#endif // NK_REDUCE_NEON_F16_H