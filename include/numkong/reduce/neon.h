/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neon.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEON_H
#define NK_REDUCE_NEON_H

#if _NK_TARGET_ARM
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Horizontal sum of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t _nk_reduce_add_f32x4_neon(float32x4_t sum_f32x4) { return vaddvq_f32(sum_f32x4); }

/** @brief Horizontal sum of 2 doubles in a NEON register. */
NK_INTERNAL nk_f64_t _nk_reduce_add_f64x2_neon(float64x2_t sum_f64x2) { return vaddvq_f64(sum_f64x2); }

/** @brief Horizontal min of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t _nk_reduce_min_f32x4_neon(float32x4_t min_f32x4) { return vminvq_f32(min_f32x4); }

/** @brief Horizontal max of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t _nk_reduce_max_f32x4_neon(float32x4_t max_f32x4) { return vmaxvq_f32(max_f32x4); }

/** @brief Horizontal sum of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t _nk_reduce_add_i32x4_neon(int32x4_t sum_i32x4) { return vaddvq_s32(sum_i32x4); }

/** @brief Horizontal min of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t _nk_reduce_min_i32x4_neon(int32x4_t min_i32x4) { return vminvq_s32(min_i32x4); }

/** @brief Horizontal max of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t _nk_reduce_max_i32x4_neon(int32x4_t max_i32x4) { return vmaxvq_s32(max_i32x4); }

/** @brief Horizontal sum of 16 u8s in a NEON register, returning u32. */
NK_INTERNAL nk_u32_t _nk_reduce_add_u8x16_neon(uint8x16_t sum_u8x16) {
    uint16x8_t low_u16x8 = vmovl_u8(vget_low_u8(sum_u8x16));
    uint16x8_t high_u16x8 = vmovl_u8(vget_high_u8(sum_u8x16));
    uint16x8_t sum_u16x8 = vaddq_u16(low_u16x8, high_u16x8);
    uint32x4_t sum_u32x4 = vpaddlq_u16(sum_u16x8);
    uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
    return (nk_u32_t)vaddvq_u64(sum_u64x2);
}

NK_INTERNAL void _nk_reduce_add_f32_neon_contiguous( //
    nk_f32_t const *data, nk_size_t count, nk_f64_t *result) {
    // Accumulate in f64 for precision
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        float64x2_t lo_f64x2 = vcvt_f64_f32(vget_low_f32(data_f32x4));
        float64x2_t hi_f64x2 = vcvt_f64_f32(vget_high_f32(data_f32x4));
        sum_f64x2 = vaddq_f64(sum_f64x2, lo_f64x2);
        sum_f64x2 = vaddq_f64(sum_f64x2, hi_f64x2);
    }
    nk_f64_t sum = _nk_reduce_add_f64x2_neon(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    if (stride_bytes == sizeof(nk_f32_t)) return _nk_reduce_add_f32_neon_contiguous(data, count, result);
    nk_reduce_add_f32_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_add_f64_neon_contiguous( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count; idx_scalars += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data + idx_scalars);
        sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2);
    }
    nk_f64_t sum = _nk_reduce_add_f64x2_neon(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_neon(                             //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    if (stride_bytes == sizeof(nk_f64_t)) return _nk_reduce_add_f64_neon_contiguous(data, count, result);
    nk_reduce_add_f64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_min_f32_neon_contiguous( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // First pass: find minimum value using SIMD
    float32x4_t min_f32x4 = vld1q_f32(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        min_f32x4 = vminq_f32(min_f32x4, data_f32x4);
    }
    nk_f32_t min_val = _nk_reduce_min_f32x4_neon(min_f32x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] < min_val) min_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != min_val) continue;
        *min_value = min_val;
        *min_index = idx_scalars;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_f32_t) && count >= 4)
        return _nk_reduce_min_f32_neon_contiguous(data, count, min_value, min_index);
    nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void _nk_reduce_max_f32_neon_contiguous( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *max_value, nk_size_t *max_index) {
    float32x4_t max_f32x4 = vld1q_f32(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        max_f32x4 = vmaxq_f32(max_f32x4, data_f32x4);
    }
    nk_f32_t max_val = _nk_reduce_max_f32x4_neon(max_f32x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] > max_val) max_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != max_val) continue;
        *max_value = max_val;
        *max_index = idx_scalars;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_f32_t) && count >= 4)
        return _nk_reduce_max_f32_neon_contiguous(data, count, max_value, max_index);
    nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // _NK_TARGET_ARM

#endif // NK_REDUCE_NEON_H