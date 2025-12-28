/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neon_f16.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEON_F16_H
#define NK_REDUCE_NEON_F16_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Horizontal sum of 8 f16s in a NEON register, returning f32. */
NK_INTERNAL nk_f32_t nk_reduce_add_f16x8_neon_(float16x8_t sum_f16x8) {
    float16x4_t low_f16x4 = vget_low_f16(sum_f16x8);
    float16x4_t high_f16x4 = vget_high_f16(sum_f16x8);
    float16x4_t sum_f16x4 = vadd_f16(low_f16x4, high_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    return vgetq_lane_f32(vcvt_f32_f16(sum_f16x4), 0);
}

NK_INTERNAL void nk_reduce_add_f16_neon_contiguous_( //
    nk_f16_t const *data, nk_size_t count, nk_f32_t *result) {
    // Use native f16 arithmetic for accumulation
    float16x8_t sum_f16x8 = vdupq_n_f16(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data + idx));
        sum_f16x8 = vaddq_f16(sum_f16x8, data_f16x8);
    }

    nk_f32_t sum = nk_reduce_add_f16x8_neon_(sum_f16x8);

    // Scalar tail - convert to f32
    for (; idx < count; ++idx) {
        nk_b128_vec_t tmp;
        tmp.f16s[0] = data[idx];
        sum += vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(tmp.u16x8)))[0];
    }

    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f16_neon_strided_(                     //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    float16x8_t sum_f16x8 = vdupq_n_f16(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            sum_f16x8 = vaddq_f16(sum_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            sum_f16x8 = vaddq_f16(sum_f16x8, data_f16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            sum_f16x8 = vaddq_f16(sum_f16x8, data_f16x8);
        }
    }

    nk_f32_t sum = nk_reduce_add_f16x8_neon_(sum_f16x8);
    for (; idx < count; ++idx) {
        nk_b128_vec_t tmp;
        tmp.f16s[0] = data[idx * stride_elements];
        sum += vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(tmp.u16x8)))[0];
    }

    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f16_neon(                             //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (!aligned) nk_reduce_add_f16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f16_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_f16_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f16_serial(data, count, stride_bytes, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_F16
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEON_F16_H
