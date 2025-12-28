/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neon.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEON_H
#define NK_REDUCE_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Type-agnostic partial load for 32-bit elements (4 elements max) into 128-bit vector (NEON). */
NK_INTERNAL void nk_partial_load_b32x4_neon_(void const *src, nk_size_t n, nk_b128_vec_t *dst) {
    nk_u32_t const *s = (nk_u32_t const *)src;
    dst->u32x4 = vdupq_n_u32(0);
    for (nk_size_t i = 0; i < n && i < 4; ++i) dst->u32s[i] = s[i];
}

/** @brief Type-agnostic partial load for 16-bit elements (8 elements max) into 128-bit vector (NEON). */
NK_INTERNAL void nk_partial_load_b16x8_neon_(void const *src, nk_size_t n, nk_b128_vec_t *dst) {
    nk_u16_t const *s = (nk_u16_t const *)src;
    dst->u16x8 = vdupq_n_u16(0);
    for (nk_size_t i = 0; i < n && i < 8; ++i) dst->u16s[i] = s[i];
}

/** @brief Type-agnostic partial load for 8-bit elements (16 elements max) into 128-bit vector (NEON). */
NK_INTERNAL void nk_partial_load_b8x16_neon_(void const *src, nk_size_t n, nk_b128_vec_t *dst) {
    nk_u8_t const *s = (nk_u8_t const *)src;
    dst->u8x16 = vdupq_n_u8(0);
    for (nk_size_t i = 0; i < n && i < 16; ++i) dst->u8s[i] = s[i];
}

/** @brief Type-agnostic partial store for 32-bit elements (4 elements max) from 128-bit vector (NEON). */
NK_INTERNAL void nk_partial_store_b32x4_neon_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    nk_u32_t *d = (nk_u32_t *)dst;
    for (nk_size_t i = 0; i < n && i < 4; ++i) d[i] = src->u32s[i];
}

/** @brief Horizontal sum of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t nk_reduce_add_f32x4_neon_(float32x4_t sum_f32x4) { return vaddvq_f32(sum_f32x4); }

/** @brief Horizontal sum of 2 doubles in a NEON register. */
NK_INTERNAL nk_f64_t nk_reduce_add_f64x2_neon_(float64x2_t sum_f64x2) { return vaddvq_f64(sum_f64x2); }

/** @brief Horizontal min of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t nk_reduce_min_f32x4_neon_(float32x4_t min_f32x4) { return vminvq_f32(min_f32x4); }

/** @brief Horizontal max of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t nk_reduce_max_f32x4_neon_(float32x4_t max_f32x4) { return vmaxvq_f32(max_f32x4); }

/** @brief Horizontal sum of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t nk_reduce_add_i32x4_neon_(int32x4_t sum_i32x4) { return vaddvq_s32(sum_i32x4); }

/** @brief Horizontal min of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t nk_reduce_min_i32x4_neon_(int32x4_t min_i32x4) { return vminvq_s32(min_i32x4); }

/** @brief Horizontal max of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t nk_reduce_max_i32x4_neon_(int32x4_t max_i32x4) { return vmaxvq_s32(max_i32x4); }

/** @brief Horizontal sum of 16 u8s in a NEON register, returning u32. */
NK_INTERNAL nk_u32_t nk_reduce_add_u8x16_neon_(uint8x16_t sum_u8x16) {
    uint16x8_t low_u16x8 = vmovl_u8(vget_low_u8(sum_u8x16));
    uint16x8_t high_u16x8 = vmovl_u8(vget_high_u8(sum_u8x16));
    uint16x8_t sum_u16x8 = vaddq_u16(low_u16x8, high_u16x8);
    uint32x4_t sum_u32x4 = vpaddlq_u16(sum_u16x8);
    uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
    return (nk_u32_t)vaddvq_u64(sum_u64x2);
}

NK_INTERNAL void nk_reduce_add_f32_neon_contiguous_( //
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
    nk_f64_t sum = nk_reduce_add_f64x2_neon_(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f32_neon_strided_(                     //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // ARM NEON has native structure load instructions for strides 2, 3, and 4.
    // These are more efficient than masked loads as they de-interleave on the fly.
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_logical = 0;

    if (stride_elements == 2) {
        // vld2q_f32 loads 8 floats (4 pairs) and de-interleaves into two float32x4_t
        for (; idx_logical + 4 <= count; idx_logical += 4) {
            float32x4x2_t data_f32x4x2 = vld2q_f32(data + idx_logical * 2);
            float32x4_t strided_f32x4 = data_f32x4x2.val[0]; // Every 2nd element
            float64x2_t lo_f64x2 = vcvt_f64_f32(vget_low_f32(strided_f32x4));
            float64x2_t hi_f64x2 = vcvt_f64_f32(vget_high_f32(strided_f32x4));
            sum_f64x2 = vaddq_f64(sum_f64x2, lo_f64x2);
            sum_f64x2 = vaddq_f64(sum_f64x2, hi_f64x2);
        }
    }
    else if (stride_elements == 3) {
        // vld3q_f32 loads 12 floats (4 triplets) and de-interleaves into three float32x4_t
        for (; idx_logical + 4 <= count; idx_logical += 4) {
            float32x4x3_t data_f32x4x3 = vld3q_f32(data + idx_logical * 3);
            float32x4_t strided_f32x4 = data_f32x4x3.val[0]; // Every 3rd element
            float64x2_t lo_f64x2 = vcvt_f64_f32(vget_low_f32(strided_f32x4));
            float64x2_t hi_f64x2 = vcvt_f64_f32(vget_high_f32(strided_f32x4));
            sum_f64x2 = vaddq_f64(sum_f64x2, lo_f64x2);
            sum_f64x2 = vaddq_f64(sum_f64x2, hi_f64x2);
        }
    }
    else if (stride_elements == 4) {
        // vld4q_f32 loads 16 floats (4 quads) and de-interleaves into four float32x4_t
        for (; idx_logical + 4 <= count; idx_logical += 4) {
            float32x4x4_t data_f32x4x4 = vld4q_f32(data + idx_logical * 4);
            float32x4_t strided_f32x4 = data_f32x4x4.val[0]; // Every 4th element
            float64x2_t lo_f64x2 = vcvt_f64_f32(vget_low_f32(strided_f32x4));
            float64x2_t hi_f64x2 = vcvt_f64_f32(vget_high_f32(strided_f32x4));
            sum_f64x2 = vaddq_f64(sum_f64x2, lo_f64x2);
            sum_f64x2 = vaddq_f64(sum_f64x2, hi_f64x2);
        }
    }

    // Scalar tail for remaining elements
    nk_f64_t sum = nk_reduce_add_f64x2_neon_(sum_f64x2);
    for (; idx_logical < count; ++idx_logical) sum += data[idx_logical * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_add_f32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f32_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_f32_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f32_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_add_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count; idx_scalars += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data + idx_scalars);
        sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2);
    }
    nk_f64_t sum = nk_reduce_add_f64x2_neon_(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_f64_neon_strided_(                     //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // ARM NEON has native structure load instructions for strides 2, 3, and 4.
    // For f64, each 128-bit register holds only 2 doubles.
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_logical = 0;

    if (stride_elements == 2) {
        // vld2q_f64 loads 4 doubles (2 pairs) and de-interleaves into two float64x2_t
        for (; idx_logical + 2 <= count; idx_logical += 2) {
            float64x2x2_t data_f64x2x2 = vld2q_f64(data + idx_logical * 2);
            sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2x2.val[0]); // Every 2nd element
        }
    }
    else if (stride_elements == 3) {
        // vld3q_f64 loads 6 doubles (2 triplets) and de-interleaves into three float64x2_t
        for (; idx_logical + 2 <= count; idx_logical += 2) {
            float64x2x3_t data_f64x2x3 = vld3q_f64(data + idx_logical * 3);
            sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2x3.val[0]); // Every 3rd element
        }
    }
    else if (stride_elements == 4) {
        // vld4q_f64 loads 8 doubles (2 quads) and de-interleaves into four float64x2_t
        for (; idx_logical + 2 <= count; idx_logical += 2) {
            float64x2x4_t data_f64x2x4 = vld4q_f64(data + idx_logical * 4);
            sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2x4.val[0]); // Every 4th element
        }
    }

    // Scalar tail for remaining elements
    nk_f64_t sum = nk_reduce_add_f64x2_neon_(sum_f64x2);
    for (; idx_logical < count; ++idx_logical) sum += data[idx_logical * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_neon(                             //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_add_f64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f64_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_f64_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_f32_neon_contiguous_( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    // Single-pass: track both min value and index in SIMD
    float32x4_t min_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx);
        uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
        min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
        min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    // Horizontal reduction using union
    nk_b128_vec_t min_vec, idx_vec;
    min_vec.f32x4 = min_f32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_f32_t best_val = min_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.f32s[i] < best_val) {
            best_val = min_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        if (data[idx] < best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_f32_neon_strided_(                     //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    // Single-pass: track both min value and index in SIMD
    float32x4_t min_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x2_t loaded_f32x4x2 = vld2q_f32(data + idx * 2);
            float32x4_t data_f32x4 = loaded_f32x4x2.val[0];
            uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x3_t loaded_f32x4x3 = vld3q_f32(data + idx * 3);
            float32x4_t data_f32x4 = loaded_f32x4x3.val[0];
            uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x4_t loaded_f32x4x4 = vld4q_f32(data + idx * 4);
            float32x4_t data_f32x4 = loaded_f32x4x4.val[0];
            uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    // Horizontal reduction using union
    nk_b128_vec_t min_vec, idx_vec;
    min_vec.f32x4 = min_f32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_f32_t best_val = min_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.f32s[i] < best_val) {
            best_val = min_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val = data[idx * stride_elements];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_min_f32_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_f32_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f32_neon_contiguous_( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    // Single-pass: track both max value and index in SIMD
    float32x4_t max_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx);
        uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
        max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
        max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    // Horizontal reduction using union
    nk_b128_vec_t max_vec, idx_vec;
    max_vec.f32x4 = max_f32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_f32_t best_val = max_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.f32s[i] > best_val) {
            best_val = max_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_f32_neon_strided_(                     //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    // Single-pass: track both max value and index in SIMD
    float32x4_t max_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x2_t loaded_f32x4x2 = vld2q_f32(data + idx * 2);
            float32x4_t data_f32x4 = loaded_f32x4x2.val[0];
            uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x3_t loaded_f32x4x3 = vld3q_f32(data + idx * 3);
            float32x4_t data_f32x4 = loaded_f32x4x3.val[0];
            uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x4_t loaded_f32x4x4 = vld4q_f32(data + idx * 4);
            float32x4_t data_f32x4 = loaded_f32x4x4.val[0];
            uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    // Horizontal reduction using union
    nk_b128_vec_t max_vec, idx_vec;
    max_vec.f32x4 = max_f32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_f32_t best_val = max_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.f32s[i] > best_val) {
            best_val = max_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val = data[idx * stride_elements];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_max_f32_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_f32_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count,           //
    nk_f64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    // Single-pass: track both min value and index in SIMD
    float64x2_t min_f64x2 = vdupq_n_f64(__builtin_huge_val());
    int64x2_t min_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    for (; idx + 2 <= count; idx += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data + idx);
        uint64x2_t lt_u64x2 = vcltq_f64(data_f64x2, min_f64x2);
        min_f64x2 = vbslq_f64(lt_u64x2, data_f64x2, min_f64x2);
        min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
        idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
    }

    // Horizontal reduction using union
    nk_b128_vec_t min_vec, idx_vec;
    min_vec.f64x2 = min_f64x2;
    idx_vec.i64x2 = min_idx_i64x2;

    nk_f64_t best_val = min_vec.f64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (min_vec.f64s[1] < best_val) {
        best_val = min_vec.f64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        if (data[idx] < best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_f64_neon_strided_(                     //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    // Single-pass: track both min value and index in SIMD
    float64x2_t min_f64x2 = vdupq_n_f64(__builtin_huge_val());
    int64x2_t min_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            float64x2x2_t loaded_f64x2x2 = vld2q_f64(data + idx * 2);
            float64x2_t data_f64x2 = loaded_f64x2x2.val[0];
            uint64x2_t lt_u64x2 = vcltq_f64(data_f64x2, min_f64x2);
            min_f64x2 = vbslq_f64(lt_u64x2, data_f64x2, min_f64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            float64x2x3_t loaded_f64x2x3 = vld3q_f64(data + idx * 3);
            float64x2_t data_f64x2 = loaded_f64x2x3.val[0];
            uint64x2_t lt_u64x2 = vcltq_f64(data_f64x2, min_f64x2);
            min_f64x2 = vbslq_f64(lt_u64x2, data_f64x2, min_f64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            float64x2x4_t loaded_f64x2x4 = vld4q_f64(data + idx * 4);
            float64x2_t data_f64x2 = loaded_f64x2x4.val[0];
            uint64x2_t lt_u64x2 = vcltq_f64(data_f64x2, min_f64x2);
            min_f64x2 = vbslq_f64(lt_u64x2, data_f64x2, min_f64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }

    // Horizontal reduction using union
    nk_b128_vec_t min_vec, idx_vec;
    min_vec.f64x2 = min_f64x2;
    idx_vec.i64x2 = min_idx_i64x2;

    nk_f64_t best_val = min_vec.f64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (min_vec.f64s[1] < best_val) {
        best_val = min_vec.f64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f64_t val = data[idx * stride_elements];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_f64_neon(                             //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_min_f64_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_f64_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count,           //
    nk_f64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    // Single-pass: track both max value and index in SIMD
    float64x2_t max_f64x2 = vdupq_n_f64(-__builtin_huge_val());
    int64x2_t max_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    for (; idx + 2 <= count; idx += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data + idx);
        uint64x2_t gt_u64x2 = vcgtq_f64(data_f64x2, max_f64x2);
        max_f64x2 = vbslq_f64(gt_u64x2, data_f64x2, max_f64x2);
        max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
        idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
    }

    // Horizontal reduction using union
    nk_b128_vec_t max_vec, idx_vec;
    max_vec.f64x2 = max_f64x2;
    idx_vec.i64x2 = max_idx_i64x2;

    nk_f64_t best_val = max_vec.f64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (max_vec.f64s[1] > best_val) {
        best_val = max_vec.f64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_f64_neon_strided_(                     //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    // Single-pass: track both max value and index in SIMD
    float64x2_t max_f64x2 = vdupq_n_f64(-__builtin_huge_val());
    int64x2_t max_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            float64x2x2_t loaded_f64x2x2 = vld2q_f64(data + idx * 2);
            float64x2_t data_f64x2 = loaded_f64x2x2.val[0];
            uint64x2_t gt_u64x2 = vcgtq_f64(data_f64x2, max_f64x2);
            max_f64x2 = vbslq_f64(gt_u64x2, data_f64x2, max_f64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            float64x2x3_t loaded_f64x2x3 = vld3q_f64(data + idx * 3);
            float64x2_t data_f64x2 = loaded_f64x2x3.val[0];
            uint64x2_t gt_u64x2 = vcgtq_f64(data_f64x2, max_f64x2);
            max_f64x2 = vbslq_f64(gt_u64x2, data_f64x2, max_f64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            float64x2x4_t loaded_f64x2x4 = vld4q_f64(data + idx * 4);
            float64x2_t data_f64x2 = loaded_f64x2x4.val[0];
            uint64x2_t gt_u64x2 = vcgtq_f64(data_f64x2, max_f64x2);
            max_f64x2 = vbslq_f64(gt_u64x2, data_f64x2, max_f64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }

    // Horizontal reduction using union
    nk_b128_vec_t max_vec, idx_vec;
    max_vec.f64x2 = max_f64x2;
    idx_vec.i64x2 = max_idx_i64x2;

    nk_f64_t best_val = max_vec.f64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (max_vec.f64s[1] > best_val) {
        best_val = max_vec.f64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f64_t val = data[idx * stride_elements];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_f64_neon(                             //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_max_f64_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_f64_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i32_neon_contiguous_( //
    nk_i32_t const *data, nk_size_t count, nk_i64_t *result) {
    // Accumulate in i64 to avoid overflow
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        int32x4_t data_i32x4 = vld1q_s32(data + idx);
        int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
        int64x2_t hi_i64x2 = vmovl_s32(vget_high_s32(data_i32x4));
        sum_i64x2 = vaddq_s64(sum_i64x2, lo_i64x2);
        sum_i64x2 = vaddq_s64(sum_i64x2, hi_i64x2);
    }
    nk_i64_t sum = vaddvq_s64(sum_i64x2);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i32_neon_strided_(                     //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x2_t loaded_i32x4x2 = vld2q_s32(data + idx * 2);
            int32x4_t data_i32x4 = loaded_i32x4x2.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            int64x2_t hi_i64x2 = vmovl_s32(vget_high_s32(data_i32x4));
            sum_i64x2 = vaddq_s64(sum_i64x2, lo_i64x2);
            sum_i64x2 = vaddq_s64(sum_i64x2, hi_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x3_t loaded_i32x4x3 = vld3q_s32(data + idx * 3);
            int32x4_t data_i32x4 = loaded_i32x4x3.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            int64x2_t hi_i64x2 = vmovl_s32(vget_high_s32(data_i32x4));
            sum_i64x2 = vaddq_s64(sum_i64x2, lo_i64x2);
            sum_i64x2 = vaddq_s64(sum_i64x2, hi_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x4_t loaded_i32x4x4 = vld4q_s32(data + idx * 4);
            int32x4_t data_i32x4 = loaded_i32x4x4.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            int64x2_t hi_i64x2 = vmovl_s32(vget_high_s32(data_i32x4));
            sum_i64x2 = vaddq_s64(sum_i64x2, lo_i64x2);
            sum_i64x2 = vaddq_s64(sum_i64x2, hi_i64x2);
        }
    }

    nk_i64_t sum = vaddvq_s64(sum_i64x2);
    for (; idx < count; ++idx) sum += data[idx * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i32_neon(                             //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (!aligned) nk_reduce_add_i32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i32_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_i32_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i32_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_i32_neon_contiguous_( //
    nk_i32_t const *data, nk_size_t count,           //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    int32x4_t min_i32x4 = vdupq_n_s32(INT32_MAX);
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        int32x4_t data_i32x4 = vld1q_s32(data + idx);
        uint32x4_t lt_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
        min_i32x4 = vbslq_s32(lt_u32x4, data_i32x4, min_i32x4);
        min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.i32x4 = min_i32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_i32_t best_val = min_vec.i32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.i32s[i] < best_val) {
            best_val = min_vec.i32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        if (data[idx] < best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_i32_neon_strided_(                     //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    int32x4_t min_i32x4 = vdupq_n_s32(INT32_MAX);
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x2_t loaded_i32x4x2 = vld2q_s32(data + idx * 2);
            int32x4_t data_i32x4 = loaded_i32x4x2.val[0];
            uint32x4_t lt_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
            min_i32x4 = vbslq_s32(lt_u32x4, data_i32x4, min_i32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x3_t loaded_i32x4x3 = vld3q_s32(data + idx * 3);
            int32x4_t data_i32x4 = loaded_i32x4x3.val[0];
            uint32x4_t lt_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
            min_i32x4 = vbslq_s32(lt_u32x4, data_i32x4, min_i32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x4_t loaded_i32x4x4 = vld4q_s32(data + idx * 4);
            int32x4_t data_i32x4 = loaded_i32x4x4.val[0];
            uint32x4_t lt_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
            min_i32x4 = vbslq_s32(lt_u32x4, data_i32x4, min_i32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.i32x4 = min_i32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_i32_t best_val = min_vec.i32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.i32s[i] < best_val) {
            best_val = min_vec.i32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        nk_i32_t val = data[idx * stride_elements];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_i32_neon(                             //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (!aligned) nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_min_i32_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_i32_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i32_neon_contiguous_( //
    nk_i32_t const *data, nk_size_t count,           //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    int32x4_t max_i32x4 = vdupq_n_s32(INT32_MIN);
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        int32x4_t data_i32x4 = vld1q_s32(data + idx);
        uint32x4_t gt_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
        max_i32x4 = vbslq_s32(gt_u32x4, data_i32x4, max_i32x4);
        max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.i32x4 = max_i32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_i32_t best_val = max_vec.i32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.i32s[i] > best_val) {
            best_val = max_vec.i32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_i32_neon_strided_(                     //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    int32x4_t max_i32x4 = vdupq_n_s32(INT32_MIN);
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x2_t loaded_i32x4x2 = vld2q_s32(data + idx * 2);
            int32x4_t data_i32x4 = loaded_i32x4x2.val[0];
            uint32x4_t gt_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
            max_i32x4 = vbslq_s32(gt_u32x4, data_i32x4, max_i32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x3_t loaded_i32x4x3 = vld3q_s32(data + idx * 3);
            int32x4_t data_i32x4 = loaded_i32x4x3.val[0];
            uint32x4_t gt_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
            max_i32x4 = vbslq_s32(gt_u32x4, data_i32x4, max_i32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x4_t loaded_i32x4x4 = vld4q_s32(data + idx * 4);
            int32x4_t data_i32x4 = loaded_i32x4x4.val[0];
            uint32x4_t gt_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
            max_i32x4 = vbslq_s32(gt_u32x4, data_i32x4, max_i32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.i32x4 = max_i32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_i32_t best_val = max_vec.i32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.i32s[i] > best_val) {
            best_val = max_vec.i32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        nk_i32_t val = data[idx * stride_elements];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_i32_neon(                             //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (!aligned) nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_max_i32_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_i32_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_u32_neon_contiguous_( //
    nk_u32_t const *data, nk_size_t count, nk_u64_t *result) {
    // Accumulate in u64 to avoid overflow
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t data_u32x4 = vld1q_u32(data + idx);
        uint64x2_t lo_u64x2 = vmovl_u32(vget_low_u32(data_u32x4));
        uint64x2_t hi_u64x2 = vmovl_u32(vget_high_u32(data_u32x4));
        sum_u64x2 = vaddq_u64(sum_u64x2, lo_u64x2);
        sum_u64x2 = vaddq_u64(sum_u64x2, hi_u64x2);
    }
    nk_u64_t sum = vaddvq_u64(sum_u64x2);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u32_neon_strided_(                     //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x2_t loaded_u32x4x2 = vld2q_u32(data + idx * 2);
            uint32x4_t data_u32x4 = loaded_u32x4x2.val[0];
            uint64x2_t lo_u64x2 = vmovl_u32(vget_low_u32(data_u32x4));
            uint64x2_t hi_u64x2 = vmovl_u32(vget_high_u32(data_u32x4));
            sum_u64x2 = vaddq_u64(sum_u64x2, lo_u64x2);
            sum_u64x2 = vaddq_u64(sum_u64x2, hi_u64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x3_t loaded_u32x4x3 = vld3q_u32(data + idx * 3);
            uint32x4_t data_u32x4 = loaded_u32x4x3.val[0];
            uint64x2_t lo_u64x2 = vmovl_u32(vget_low_u32(data_u32x4));
            uint64x2_t hi_u64x2 = vmovl_u32(vget_high_u32(data_u32x4));
            sum_u64x2 = vaddq_u64(sum_u64x2, lo_u64x2);
            sum_u64x2 = vaddq_u64(sum_u64x2, hi_u64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x4_t loaded_u32x4x4 = vld4q_u32(data + idx * 4);
            uint32x4_t data_u32x4 = loaded_u32x4x4.val[0];
            uint64x2_t lo_u64x2 = vmovl_u32(vget_low_u32(data_u32x4));
            uint64x2_t hi_u64x2 = vmovl_u32(vget_high_u32(data_u32x4));
            sum_u64x2 = vaddq_u64(sum_u64x2, lo_u64x2);
            sum_u64x2 = vaddq_u64(sum_u64x2, hi_u64x2);
        }
    }

    nk_u64_t sum = vaddvq_u64(sum_u64x2);
    for (; idx < count; ++idx) sum += data[idx * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u32_neon(                             //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (!aligned) nk_reduce_add_u32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u32_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_u32_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u32_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_u32_neon_contiguous_( //
    nk_u32_t const *data, nk_size_t count,           //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    uint32x4_t min_u32x4 = vdupq_n_u32(UINT32_MAX);
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t data_u32x4 = vld1q_u32(data + idx);
        uint32x4_t lt_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
        min_u32x4 = vbslq_u32(lt_u32x4, data_u32x4, min_u32x4);
        min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.u32x4 = min_u32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_u32_t best_val = min_vec.u32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.u32s[i] < best_val) {
            best_val = min_vec.u32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        if (data[idx] < best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_u32_neon_strided_(                     //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    uint32x4_t min_u32x4 = vdupq_n_u32(UINT32_MAX);
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x2_t loaded_u32x4x2 = vld2q_u32(data + idx * 2);
            uint32x4_t data_u32x4 = loaded_u32x4x2.val[0];
            uint32x4_t lt_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
            min_u32x4 = vbslq_u32(lt_u32x4, data_u32x4, min_u32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x3_t loaded_u32x4x3 = vld3q_u32(data + idx * 3);
            uint32x4_t data_u32x4 = loaded_u32x4x3.val[0];
            uint32x4_t lt_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
            min_u32x4 = vbslq_u32(lt_u32x4, data_u32x4, min_u32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x4_t loaded_u32x4x4 = vld4q_u32(data + idx * 4);
            uint32x4_t data_u32x4 = loaded_u32x4x4.val[0];
            uint32x4_t lt_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
            min_u32x4 = vbslq_u32(lt_u32x4, data_u32x4, min_u32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.u32x4 = min_u32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_u32_t best_val = min_vec.u32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.u32s[i] < best_val) {
            best_val = min_vec.u32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        nk_u32_t val = data[idx * stride_elements];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_u32_neon(                             //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (!aligned) nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_min_u32_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_u32_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u32_neon_contiguous_( //
    nk_u32_t const *data, nk_size_t count,           //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    uint32x4_t max_u32x4 = vdupq_n_u32(0);
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t data_u32x4 = vld1q_u32(data + idx);
        uint32x4_t gt_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
        max_u32x4 = vbslq_u32(gt_u32x4, data_u32x4, max_u32x4);
        max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.u32x4 = max_u32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_u32_t best_val = max_vec.u32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.u32s[i] > best_val) {
            best_val = max_vec.u32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_u32_neon_strided_(                     //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    uint32x4_t max_u32x4 = vdupq_n_u32(0);
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x2_t loaded_u32x4x2 = vld2q_u32(data + idx * 2);
            uint32x4_t data_u32x4 = loaded_u32x4x2.val[0];
            uint32x4_t gt_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
            max_u32x4 = vbslq_u32(gt_u32x4, data_u32x4, max_u32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x3_t loaded_u32x4x3 = vld3q_u32(data + idx * 3);
            uint32x4_t data_u32x4 = loaded_u32x4x3.val[0];
            uint32x4_t gt_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
            max_u32x4 = vbslq_u32(gt_u32x4, data_u32x4, max_u32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x4_t loaded_u32x4x4 = vld4q_u32(data + idx * 4);
            uint32x4_t data_u32x4 = loaded_u32x4x4.val[0];
            uint32x4_t gt_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
            max_u32x4 = vbslq_u32(gt_u32x4, data_u32x4, max_u32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.u32x4 = max_u32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_u32_t best_val = max_vec.u32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.u32s[i] > best_val) {
            best_val = max_vec.u32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    for (; idx < count; ++idx) {
        nk_u32_t val = data[idx * stride_elements];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_u32_neon(                             //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (!aligned) nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_max_u32_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_u32_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i64_neon_contiguous_( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *result) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        int64x2_t data_i64x2 = vld1q_s64(data + idx);
        sum_i64x2 = vaddq_s64(sum_i64x2, data_i64x2);
    }
    nk_i64_t sum = vaddvq_s64(sum_i64x2);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_i64_neon_strided_(                     //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *result) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x2_t loaded_i64x2x2 = vld2q_s64(data + idx * 2);
            sum_i64x2 = vaddq_s64(sum_i64x2, loaded_i64x2x2.val[0]);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x3_t loaded_i64x2x3 = vld3q_s64(data + idx * 3);
            sum_i64x2 = vaddq_s64(sum_i64x2, loaded_i64x2x3.val[0]);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x4_t loaded_i64x2x4 = vld4q_s64(data + idx * 4);
            sum_i64x2 = vaddq_s64(sum_i64x2, loaded_i64x2x4.val[0]);
        }
    }

    nk_i64_t sum = vaddvq_s64(sum_i64x2);
    for (; idx < count; ++idx) sum += data[idx * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i64_neon(                             //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (!aligned) nk_reduce_add_i64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i64_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_i64_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_i64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_i64_neon_contiguous_( //
    nk_i64_t const *data, nk_size_t count,           //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    int64x2_t min_i64x2 = vdupq_n_s64(INT64_MAX);
    int64x2_t min_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    for (; idx + 2 <= count; idx += 2) {
        int64x2_t data_i64x2 = vld1q_s64(data + idx);
        uint64x2_t lt_u64x2 = vcltq_s64(data_i64x2, min_i64x2);
        min_i64x2 = vbslq_s64(lt_u64x2, data_i64x2, min_i64x2);
        min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
        idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.i64x2 = min_i64x2;
    idx_vec.i64x2 = min_idx_i64x2;

    nk_i64_t best_val = min_vec.i64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (min_vec.i64s[1] < best_val) {
        best_val = min_vec.i64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        if (data[idx] < best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_i64_neon_strided_(                     //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    int64x2_t min_i64x2 = vdupq_n_s64(INT64_MAX);
    int64x2_t min_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x2_t loaded_i64x2x2 = vld2q_s64(data + idx * 2);
            int64x2_t data_i64x2 = loaded_i64x2x2.val[0];
            uint64x2_t lt_u64x2 = vcltq_s64(data_i64x2, min_i64x2);
            min_i64x2 = vbslq_s64(lt_u64x2, data_i64x2, min_i64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x3_t loaded_i64x2x3 = vld3q_s64(data + idx * 3);
            int64x2_t data_i64x2 = loaded_i64x2x3.val[0];
            uint64x2_t lt_u64x2 = vcltq_s64(data_i64x2, min_i64x2);
            min_i64x2 = vbslq_s64(lt_u64x2, data_i64x2, min_i64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x4_t loaded_i64x2x4 = vld4q_s64(data + idx * 4);
            int64x2_t data_i64x2 = loaded_i64x2x4.val[0];
            uint64x2_t lt_u64x2 = vcltq_s64(data_i64x2, min_i64x2);
            min_i64x2 = vbslq_s64(lt_u64x2, data_i64x2, min_i64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.i64x2 = min_i64x2;
    idx_vec.i64x2 = min_idx_i64x2;

    nk_i64_t best_val = min_vec.i64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (min_vec.i64s[1] < best_val) {
        best_val = min_vec.i64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx * stride_elements];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_i64_neon(                             //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (!aligned) nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_min_i64_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_i64_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i64_neon_contiguous_( //
    nk_i64_t const *data, nk_size_t count,           //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    int64x2_t max_i64x2 = vdupq_n_s64(INT64_MIN);
    int64x2_t max_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    for (; idx + 2 <= count; idx += 2) {
        int64x2_t data_i64x2 = vld1q_s64(data + idx);
        uint64x2_t gt_u64x2 = vcgtq_s64(data_i64x2, max_i64x2);
        max_i64x2 = vbslq_s64(gt_u64x2, data_i64x2, max_i64x2);
        max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
        idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.i64x2 = max_i64x2;
    idx_vec.i64x2 = max_idx_i64x2;

    nk_i64_t best_val = max_vec.i64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (max_vec.i64s[1] > best_val) {
        best_val = max_vec.i64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_i64_neon_strided_(                     //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    int64x2_t max_i64x2 = vdupq_n_s64(INT64_MIN);
    int64x2_t max_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x2_t loaded_i64x2x2 = vld2q_s64(data + idx * 2);
            int64x2_t data_i64x2 = loaded_i64x2x2.val[0];
            uint64x2_t gt_u64x2 = vcgtq_s64(data_i64x2, max_i64x2);
            max_i64x2 = vbslq_s64(gt_u64x2, data_i64x2, max_i64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x3_t loaded_i64x2x3 = vld3q_s64(data + idx * 3);
            int64x2_t data_i64x2 = loaded_i64x2x3.val[0];
            uint64x2_t gt_u64x2 = vcgtq_s64(data_i64x2, max_i64x2);
            max_i64x2 = vbslq_s64(gt_u64x2, data_i64x2, max_i64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            int64x2x4_t loaded_i64x2x4 = vld4q_s64(data + idx * 4);
            int64x2_t data_i64x2 = loaded_i64x2x4.val[0];
            uint64x2_t gt_u64x2 = vcgtq_s64(data_i64x2, max_i64x2);
            max_i64x2 = vbslq_s64(gt_u64x2, data_i64x2, max_i64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.i64x2 = max_i64x2;
    idx_vec.i64x2 = max_idx_i64x2;

    nk_i64_t best_val = max_vec.i64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (max_vec.i64s[1] > best_val) {
        best_val = max_vec.i64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx * stride_elements];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_i64_neon(                             //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (!aligned) nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_max_i64_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_i64_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_u64_neon_contiguous_( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *result) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t data_u64x2 = vld1q_u64(data + idx);
        sum_u64x2 = vaddq_u64(sum_u64x2, data_u64x2);
    }
    nk_u64_t sum = vaddvq_u64(sum_u64x2);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void nk_reduce_add_u64_neon_strided_(                     //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *result) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x2_t loaded_u64x2x2 = vld2q_u64(data + idx * 2);
            sum_u64x2 = vaddq_u64(sum_u64x2, loaded_u64x2x2.val[0]);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x3_t loaded_u64x2x3 = vld3q_u64(data + idx * 3);
            sum_u64x2 = vaddq_u64(sum_u64x2, loaded_u64x2x3.val[0]);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x4_t loaded_u64x2x4 = vld4q_u64(data + idx * 4);
            sum_u64x2 = vaddq_u64(sum_u64x2, loaded_u64x2x4.val[0]);
        }
    }

    nk_u64_t sum = vaddvq_u64(sum_u64x2);
    for (; idx < count; ++idx) sum += data[idx * stride_elements];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u64_neon(                             //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (!aligned) nk_reduce_add_u64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u64_neon_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_u64_neon_strided_(data, count, stride_elements, result);
    else nk_reduce_add_u64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_u64_neon_contiguous_( //
    nk_u64_t const *data, nk_size_t count,           //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    uint64x2_t min_u64x2 = vdupq_n_u64(UINT64_MAX);
    int64x2_t min_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t data_u64x2 = vld1q_u64(data + idx);
        uint64x2_t lt_u64x2 = vcltq_u64(data_u64x2, min_u64x2);
        min_u64x2 = vbslq_u64(lt_u64x2, data_u64x2, min_u64x2);
        min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
        idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.u64x2 = min_u64x2;
    idx_vec.i64x2 = min_idx_i64x2;

    nk_u64_t best_val = min_vec.u64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (min_vec.u64s[1] < best_val) {
        best_val = min_vec.u64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        if (data[idx] < best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_u64_neon_strided_(                     //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    uint64x2_t min_u64x2 = vdupq_n_u64(UINT64_MAX);
    int64x2_t min_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x2_t loaded_u64x2x2 = vld2q_u64(data + idx * 2);
            uint64x2_t data_u64x2 = loaded_u64x2x2.val[0];
            uint64x2_t lt_u64x2 = vcltq_u64(data_u64x2, min_u64x2);
            min_u64x2 = vbslq_u64(lt_u64x2, data_u64x2, min_u64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x3_t loaded_u64x2x3 = vld3q_u64(data + idx * 3);
            uint64x2_t data_u64x2 = loaded_u64x2x3.val[0];
            uint64x2_t lt_u64x2 = vcltq_u64(data_u64x2, min_u64x2);
            min_u64x2 = vbslq_u64(lt_u64x2, data_u64x2, min_u64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x4_t loaded_u64x2x4 = vld4q_u64(data + idx * 4);
            uint64x2_t data_u64x2 = loaded_u64x2x4.val[0];
            uint64x2_t lt_u64x2 = vcltq_u64(data_u64x2, min_u64x2);
            min_u64x2 = vbslq_u64(lt_u64x2, data_u64x2, min_u64x2);
            min_idx_i64x2 = vbslq_s64(lt_u64x2, idx_i64x2, min_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }

    nk_b128_vec_t min_vec, idx_vec;
    min_vec.u64x2 = min_u64x2;
    idx_vec.i64x2 = min_idx_i64x2;

    nk_u64_t best_val = min_vec.u64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (min_vec.u64s[1] < best_val) {
        best_val = min_vec.u64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx * stride_elements];
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_u64_neon(                             //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (!aligned) nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_min_u64_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_u64_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u64_neon_contiguous_( //
    nk_u64_t const *data, nk_size_t count,           //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    uint64x2_t max_u64x2 = vdupq_n_u64(0);
    int64x2_t max_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t data_u64x2 = vld1q_u64(data + idx);
        uint64x2_t gt_u64x2 = vcgtq_u64(data_u64x2, max_u64x2);
        max_u64x2 = vbslq_u64(gt_u64x2, data_u64x2, max_u64x2);
        max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
        idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.u64x2 = max_u64x2;
    idx_vec.i64x2 = max_idx_i64x2;

    nk_u64_t best_val = max_vec.u64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (max_vec.u64s[1] > best_val) {
        best_val = max_vec.u64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        if (data[idx] > best_val) {
            best_val = data[idx];
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_u64_neon_strided_(                     //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    uint64x2_t max_u64x2 = vdupq_n_u64(0);
    int64x2_t max_idx_i64x2 = vdupq_n_s64(0);
    int64x2_t idx_i64x2 = {0, 1};
    int64x2_t step_i64x2 = vdupq_n_s64(2);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x2_t loaded_u64x2x2 = vld2q_u64(data + idx * 2);
            uint64x2_t data_u64x2 = loaded_u64x2x2.val[0];
            uint64x2_t gt_u64x2 = vcgtq_u64(data_u64x2, max_u64x2);
            max_u64x2 = vbslq_u64(gt_u64x2, data_u64x2, max_u64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x3_t loaded_u64x2x3 = vld3q_u64(data + idx * 3);
            uint64x2_t data_u64x2 = loaded_u64x2x3.val[0];
            uint64x2_t gt_u64x2 = vcgtq_u64(data_u64x2, max_u64x2);
            max_u64x2 = vbslq_u64(gt_u64x2, data_u64x2, max_u64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 2 <= count; idx += 2) {
            uint64x2x4_t loaded_u64x2x4 = vld4q_u64(data + idx * 4);
            uint64x2_t data_u64x2 = loaded_u64x2x4.val[0];
            uint64x2_t gt_u64x2 = vcgtq_u64(data_u64x2, max_u64x2);
            max_u64x2 = vbslq_u64(gt_u64x2, data_u64x2, max_u64x2);
            max_idx_i64x2 = vbslq_s64(gt_u64x2, idx_i64x2, max_idx_i64x2);
            idx_i64x2 = vaddq_s64(idx_i64x2, step_i64x2);
        }
    }

    nk_b128_vec_t max_vec, idx_vec;
    max_vec.u64x2 = max_u64x2;
    idx_vec.i64x2 = max_idx_i64x2;

    nk_u64_t best_val = max_vec.u64s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i64s[0];
    if (max_vec.u64s[1] > best_val) {
        best_val = max_vec.u64s[1];
        best_idx = (nk_size_t)idx_vec.i64s[1];
    }

    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx * stride_elements];
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_u64_neon(                             //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (!aligned) nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_max_u64_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_u64_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEON_H