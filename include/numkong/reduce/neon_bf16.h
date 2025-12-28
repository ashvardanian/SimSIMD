/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neonbfdot.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEONBFDOT_H
#define NK_REDUCE_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL void nk_reduce_add_bf16_neon_contiguous_( //
    nk_bf16_t const *data, nk_size_t count, nk_f32_t *result) {
    // Use vbfdotq_f32 with ones vector: dot(data, ones) = sum(data)
    // bf16 representation of 1.0 is 0x3F80 (same as upper 16 bits of f32 1.0)
    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        bfloat16x8_t data_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(data + idx));
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
    }

    // Handle tail with type-agnostic partial load
    if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_neon_(data + idx, count - idx, &tail_vec);
        bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(tail_vec.u16x8);
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
    }

    *result = vaddvq_f32(sum_f32x4);
}

NK_INTERNAL void nk_reduce_add_bf16_neon_strided_(                     //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    // For strided bf16, use vld2/vld3/vld4 to de-interleave, then dot with ones
    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            // vld2 loads 16 bf16 values and de-interleaves into two bfloat16x8_t
            // We need to load as u16 and reinterpret since vld2q_bf16 may not exist
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x2.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x3.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x4.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        }
    }

    // Scalar tail
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    for (; idx < count; ++idx) {
        nk_b128_vec_t tmp;
        tmp.bf16s[0] = data[idx * stride_elements];
        tmp.u16s[1] = 0;
        sum += vcvt_f32_bf16(vreinterpret_bf16_u16(vget_low_u16(tmp.u16x8)))[0];
    }

    *result = sum;
}

NK_PUBLIC void nk_reduce_add_bf16_neon(                             //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (!aligned) nk_reduce_add_bf16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_bf16_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_bf16_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_bf16_serial(data, count, stride_bytes, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEONBFDOT_H
