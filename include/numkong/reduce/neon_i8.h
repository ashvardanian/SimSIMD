/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neon_i8.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEON_I8_H
#define NK_REDUCE_NEON_I8_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON_I8
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL void nk_reduce_add_i8_neon_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    // Use vdotq_s32 with ones vector: dot(data, ones) = sum(data)
    // vdotq_s32 computes 4 dot products of 4 i8 pairs each, accumulating into i32
    int8x16_t ones_i8x16 = vdupq_n_s8(1);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;

    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data + idx);
        sum_i32x4 = vdotq_s32(sum_i32x4, data_i8x16, ones_i8x16);
    }

    // Horizontal sum to i64
    nk_i64_t sum = vaddvq_s32(sum_i32x4);

    // Scalar tail
    for (; idx < count; ++idx) sum += data[idx];

    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i8_neon_strided_(                     //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    int8x16_t ones_i8x16 = vdupq_n_s8(1);
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x2_t loaded_i8x16x2 = vld2q_s8(data + idx * 2);
            sum_i32x4 = vdotq_s32(sum_i32x4, loaded_i8x16x2.val[0], ones_i8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x3_t loaded_i8x16x3 = vld3q_s8(data + idx * 3);
            sum_i32x4 = vdotq_s32(sum_i32x4, loaded_i8x16x3.val[0], ones_i8x16);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x4_t loaded_i8x16x4 = vld4q_s8(data + idx * 4);
            sum_i32x4 = vdotq_s32(sum_i32x4, loaded_i8x16x4.val[0], ones_i8x16);
        }
    }

    nk_i64_t sum = vaddvq_s32(sum_i32x4);
    for (; idx < count; ++idx) sum += data[idx * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i8_neon(                             //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (!aligned) nk_reduce_add_i8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i8_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_i8_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_u8_neon_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    // Use vdotq_u32 with ones vector
    uint8x16_t ones_u8x16 = vdupq_n_u8(1);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    nk_size_t idx = 0;

    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data + idx);
        sum_u32x4 = vdotq_u32(sum_u32x4, data_u8x16, ones_u8x16);
    }

    nk_u64_t sum = vaddvq_u32(sum_u32x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u8_neon_strided_(                     //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    uint8x16_t ones_u8x16 = vdupq_n_u8(1);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8(data + idx * 2);
            sum_u32x4 = vdotq_u32(sum_u32x4, loaded_u8x16x2.val[0], ones_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8(data + idx * 3);
            sum_u32x4 = vdotq_u32(sum_u32x4, loaded_u8x16x3.val[0], ones_u8x16);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8(data + idx * 4);
            sum_u32x4 = vdotq_u32(sum_u32x4, loaded_u8x16x4.val[0], ones_u8x16);
        }
    }

    nk_u64_t sum = vaddvq_u32(sum_u32x4);
    for (; idx < count; ++idx) sum += data[idx * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u8_neon(                             //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (!aligned) nk_reduce_add_u8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u8_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_u8_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u8_serial(data, count, stride_bytes, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_I8
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEON_I8_H
