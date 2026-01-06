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
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

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
    float64x2_t compensation_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count; idx_scalars += 2) {
        float64x2_t term_f64x2 = vld1q_f64(data + idx_scalars);
        float64x2_t tentative_f64x2 = vaddq_f64(sum_f64x2, term_f64x2);
        float64x2_t absolute_sum_f64x2 = vabsq_f64(sum_f64x2);
        float64x2_t absolute_term_f64x2 = vabsq_f64(term_f64x2);
        uint64x2_t sum_bigger_u64x2 = vcgeq_f64(absolute_sum_f64x2, absolute_term_f64x2);
        float64x2_t correction_sum_bigger_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, tentative_f64x2), term_f64x2);
        float64x2_t correction_term_bigger_f64x2 = vaddq_f64(vsubq_f64(term_f64x2, tentative_f64x2), sum_f64x2);
        float64x2_t correction_f64x2 = vbslq_f64(sum_bigger_u64x2, correction_sum_bigger_f64x2,
                                                 correction_term_bigger_f64x2);
        compensation_f64x2 = vaddq_f64(compensation_f64x2, correction_f64x2);
        sum_f64x2 = tentative_f64x2;
    }
    float64x2_t total_f64x2 = vaddq_f64(sum_f64x2, compensation_f64x2);
    nk_f64_t sum = nk_reduce_add_f64x2_neon_(total_f64x2);
    nk_f64_t compensation = 0;
    for (; idx_scalars < count; ++idx_scalars) {
        nk_f64_t term = data[idx_scalars], tentative = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - tentative) + term)
                                                                : ((term - tentative) + sum);
        sum = tentative;
    }
    *result = sum + compensation;
}

NK_INTERNAL void nk_reduce_add_f64_neon_strided_(                     //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // ARM NEON has native structure load instructions for strides 2, 3, and 4.
    // For f64, each 128-bit register holds only 2 doubles.
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    float64x2_t compensation_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_logical = 0;

    if (stride_elements == 2) {
        for (; idx_logical + 2 <= count; idx_logical += 2) {
            float64x2x2_t data_f64x2x2 = vld2q_f64(data + idx_logical * 2);
            float64x2_t term_f64x2 = data_f64x2x2.val[0];
            float64x2_t tentative_f64x2 = vaddq_f64(sum_f64x2, term_f64x2);
            float64x2_t absolute_sum_f64x2 = vabsq_f64(sum_f64x2);
            float64x2_t absolute_term_f64x2 = vabsq_f64(term_f64x2);
            uint64x2_t sum_bigger_u64x2 = vcgeq_f64(absolute_sum_f64x2, absolute_term_f64x2);
            float64x2_t correction_sum_bigger_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, tentative_f64x2), term_f64x2);
            float64x2_t correction_term_bigger_f64x2 = vaddq_f64(vsubq_f64(term_f64x2, tentative_f64x2), sum_f64x2);
            float64x2_t correction_f64x2 = vbslq_f64(sum_bigger_u64x2, correction_sum_bigger_f64x2,
                                                     correction_term_bigger_f64x2);
            compensation_f64x2 = vaddq_f64(compensation_f64x2, correction_f64x2);
            sum_f64x2 = tentative_f64x2;
        }
    }
    else if (stride_elements == 3) {
        for (; idx_logical + 2 <= count; idx_logical += 2) {
            float64x2x3_t data_f64x2x3 = vld3q_f64(data + idx_logical * 3);
            float64x2_t term_f64x2 = data_f64x2x3.val[0];
            float64x2_t tentative_f64x2 = vaddq_f64(sum_f64x2, term_f64x2);
            float64x2_t absolute_sum_f64x2 = vabsq_f64(sum_f64x2);
            float64x2_t absolute_term_f64x2 = vabsq_f64(term_f64x2);
            uint64x2_t sum_bigger_u64x2 = vcgeq_f64(absolute_sum_f64x2, absolute_term_f64x2);
            float64x2_t correction_sum_bigger_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, tentative_f64x2), term_f64x2);
            float64x2_t correction_term_bigger_f64x2 = vaddq_f64(vsubq_f64(term_f64x2, tentative_f64x2), sum_f64x2);
            float64x2_t correction_f64x2 = vbslq_f64(sum_bigger_u64x2, correction_sum_bigger_f64x2,
                                                     correction_term_bigger_f64x2);
            compensation_f64x2 = vaddq_f64(compensation_f64x2, correction_f64x2);
            sum_f64x2 = tentative_f64x2;
        }
    }
    else if (stride_elements == 4) {
        for (; idx_logical + 2 <= count; idx_logical += 2) {
            float64x2x4_t data_f64x2x4 = vld4q_f64(data + idx_logical * 4);
            float64x2_t term_f64x2 = data_f64x2x4.val[0];
            float64x2_t tentative_f64x2 = vaddq_f64(sum_f64x2, term_f64x2);
            float64x2_t absolute_sum_f64x2 = vabsq_f64(sum_f64x2);
            float64x2_t absolute_term_f64x2 = vabsq_f64(term_f64x2);
            uint64x2_t sum_bigger_u64x2 = vcgeq_f64(absolute_sum_f64x2, absolute_term_f64x2);
            float64x2_t correction_sum_bigger_f64x2 = vaddq_f64(vsubq_f64(sum_f64x2, tentative_f64x2), term_f64x2);
            float64x2_t correction_term_bigger_f64x2 = vaddq_f64(vsubq_f64(term_f64x2, tentative_f64x2), sum_f64x2);
            float64x2_t correction_f64x2 = vbslq_f64(sum_bigger_u64x2, correction_sum_bigger_f64x2,
                                                     correction_term_bigger_f64x2);
            compensation_f64x2 = vaddq_f64(compensation_f64x2, correction_f64x2);
            sum_f64x2 = tentative_f64x2;
        }
    }

    // Scalar tail with Neumaier
    float64x2_t total_f64x2 = vaddq_f64(sum_f64x2, compensation_f64x2);
    nk_f64_t sum = nk_reduce_add_f64x2_neon_(total_f64x2);
    nk_f64_t compensation = 0;
    for (; idx_logical < count; ++idx_logical) {
        nk_f64_t term = data[idx_logical * stride_elements], tentative = sum + term;
        compensation += (nk_f64_abs_(sum) >= nk_f64_abs_(term)) ? ((sum - tentative) + term)
                                                                : ((term - tentative) + sum);
        sum = tentative;
    }
    *result = sum + compensation;
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
    if (count == 0 || !aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_min_f32_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_f32_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f32_neon_contiguous_( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *max_value, nk_size_t *max_index) {

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
    if (count == 0 || !aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_max_f32_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_f32_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_min_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count,           //
    nk_f64_t *min_value, nk_size_t *min_index) {

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
    if (count == 0 || !aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_min_f64_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_f64_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count,           //
    nk_f64_t *max_value, nk_size_t *max_index) {

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
    if (count == 0 || !aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
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

    int32x4_t min_i32x4 = vdupq_n_s32(NK_I32_MAX);
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

    int32x4_t min_i32x4 = vdupq_n_s32(NK_I32_MAX);
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
    if (count == 0 || !aligned) nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_min_i32_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_i32_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i32_neon_contiguous_( //
    nk_i32_t const *data, nk_size_t count,           //
    nk_i32_t *max_value, nk_size_t *max_index) {

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
    if (count == 0 || !aligned) nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
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

    uint32x4_t min_u32x4 = vdupq_n_u32(NK_U32_MAX);
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

    uint32x4_t min_u32x4 = vdupq_n_u32(NK_U32_MAX);
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
    if (count == 0 || !aligned) nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4) nk_reduce_min_u32_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_u32_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u32_neon_contiguous_( //
    nk_u32_t const *data, nk_size_t count,           //
    nk_u32_t *max_value, nk_size_t *max_index) {

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
    if (count == 0 || !aligned) nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
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

    int64x2_t min_i64x2 = vdupq_n_s64(NK_I64_MAX);
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

    int64x2_t min_i64x2 = vdupq_n_s64(NK_I64_MAX);
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
    if (count == 0 || !aligned) nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_min_i64_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_i64_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i64_neon_contiguous_( //
    nk_i64_t const *data, nk_size_t count,           //
    nk_i64_t *max_value, nk_size_t *max_index) {

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
    if (count == 0 || !aligned) nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
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

    uint64x2_t min_u64x2 = vdupq_n_u64(NK_U64_MAX);
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

    uint64x2_t min_u64x2 = vdupq_n_u64(NK_U64_MAX);
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
    if (count == 0 || !aligned) nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_min_u64_neon_contiguous_(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_min_u64_neon_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u64_neon_contiguous_( //
    nk_u64_t const *data, nk_size_t count,           //
    nk_u64_t *max_value, nk_size_t *max_index) {

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
    if (count == 0 || !aligned) nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 2) nk_reduce_max_u64_neon_contiguous_(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        nk_reduce_max_u64_neon_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i8_neon_contiguous_( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data + idx);
        int16x8_t lo_i16x8 = vmovl_s8(vget_low_s8(data_i8x16));
        int16x8_t hi_i16x8 = vmovl_s8(vget_high_s8(data_i8x16));
        sum_i32x4 = vaddq_s32(sum_i32x4, vaddl_s16(vget_low_s16(lo_i16x8), vget_high_s16(lo_i16x8)));
        sum_i32x4 = vaddq_s32(sum_i32x4, vaddl_s16(vget_low_s16(hi_i16x8), vget_high_s16(hi_i16x8)));
    }
    nk_i64_t sum = vaddvq_s32(sum_i32x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i8_neon(                             //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (!aligned) nk_reduce_add_i8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i8_neon_contiguous_(data, count, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_i8_neon_contiguous_( //
    nk_i8_t const *data, nk_size_t count,           //
    nk_i8_t *min_value, nk_size_t *min_index) {
    int8x16_t min_i8x16 = vdupq_n_s8(127);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data + idx);
        min_i8x16 = vminq_s8(min_i8x16, data_i8x16);
    }
    nk_i8_t best_val = vminvq_s8(min_i8x16);
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] < best_val || (data[i] == best_val && best_idx == 0)) {
            if (data[i] <= best_val) {
                best_val = data[i];
                best_idx = i;
                break;
            }
        }
    }
    // Find first occurrence
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_i8_neon(                             //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 16) nk_reduce_min_i8_neon_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i8_neon_contiguous_( //
    nk_i8_t const *data, nk_size_t count,           //
    nk_i8_t *max_value, nk_size_t *max_index) {
    int8x16_t max_i8x16 = vdupq_n_s8(-128);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data + idx);
        max_i8x16 = vmaxq_s8(max_i8x16, data_i8x16);
    }
    nk_i8_t best_val = vmaxvq_s8(max_i8x16);
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_i8_neon(                             //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 16) nk_reduce_max_i8_neon_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_u8_neon_contiguous_( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data + idx);
        uint16x8_t lo_u16x8 = vmovl_u8(vget_low_u8(data_u8x16));
        uint16x8_t hi_u16x8 = vmovl_u8(vget_high_u8(data_u8x16));
        sum_u32x4 = vaddq_u32(sum_u32x4, vaddl_u16(vget_low_u16(lo_u16x8), vget_high_u16(lo_u16x8)));
        sum_u32x4 = vaddq_u32(sum_u32x4, vaddl_u16(vget_low_u16(hi_u16x8), vget_high_u16(hi_u16x8)));
    }
    nk_u64_t sum = vaddvq_u32(sum_u32x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u8_neon(                             //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (!aligned) nk_reduce_add_u8_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u8_neon_contiguous_(data, count, result);
    else nk_reduce_add_u8_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_u8_neon_contiguous_( //
    nk_u8_t const *data, nk_size_t count,           //
    nk_u8_t *min_value, nk_size_t *min_index) {
    uint8x16_t min_u8x16 = vdupq_n_u8(255);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data + idx);
        min_u8x16 = vminq_u8(min_u8x16, data_u8x16);
    }
    nk_u8_t best_val = vminvq_u8(min_u8x16);
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_u8_neon(                             //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 16) nk_reduce_min_u8_neon_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u8_neon_contiguous_( //
    nk_u8_t const *data, nk_size_t count,           //
    nk_u8_t *max_value, nk_size_t *max_index) {
    uint8x16_t max_u8x16 = vdupq_n_u8(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data + idx);
        max_u8x16 = vmaxq_u8(max_u8x16, data_u8x16);
    }
    nk_u8_t best_val = vmaxvq_u8(max_u8x16);
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_u8_neon(                             //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 16) nk_reduce_max_u8_neon_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_i16_neon_contiguous_( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    int32x4_t sum_i32x4 = vdupq_n_s32(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t data_i16x8 = vld1q_s16(data + idx);
        sum_i32x4 = vaddq_s32(sum_i32x4, vaddl_s16(vget_low_s16(data_i16x8), vget_high_s16(data_i16x8)));
    }
    nk_i64_t sum = vaddvq_s32(sum_i32x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i16_neon(                             //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (!aligned) nk_reduce_add_i16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_i16_neon_contiguous_(data, count, result);
    else nk_reduce_add_i16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_i16_neon_contiguous_( //
    nk_i16_t const *data, nk_size_t count,           //
    nk_i16_t *min_value, nk_size_t *min_index) {
    int16x8_t min_i16x8 = vdupq_n_s16(32767);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t data_i16x8 = vld1q_s16(data + idx);
        min_i16x8 = vminq_s16(min_i16x8, data_i16x8);
    }
    nk_i16_t best_val = vminvq_s16(min_i16x8);
    // Handle tail
    for (; idx < count; ++idx) {
        if (data[idx] < best_val) best_val = data[idx];
    }
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_i16_neon(                             //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8) nk_reduce_min_i16_neon_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_i16_neon_contiguous_( //
    nk_i16_t const *data, nk_size_t count,           //
    nk_i16_t *max_value, nk_size_t *max_index) {
    int16x8_t max_i16x8 = vdupq_n_s16(-32768);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t data_i16x8 = vld1q_s16(data + idx);
        max_i16x8 = vmaxq_s16(max_i16x8, data_i16x8);
    }
    nk_i16_t best_val = vmaxvq_s16(max_i16x8);
    // Handle tail
    for (; idx < count; ++idx) {
        if (data[idx] > best_val) best_val = data[idx];
    }
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_i16_neon(                             //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8) nk_reduce_max_i16_neon_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void nk_reduce_add_u16_neon_contiguous_( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    uint32x4_t sum_u32x4 = vdupq_n_u32(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t data_u16x8 = vld1q_u16(data + idx);
        sum_u32x4 = vaddq_u32(sum_u32x4, vaddl_u16(vget_low_u16(data_u16x8), vget_high_u16(data_u16x8)));
    }
    nk_u64_t sum = vaddvq_u32(sum_u32x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u16_neon(                             //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (!aligned) nk_reduce_add_u16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_u16_neon_contiguous_(data, count, result);
    else nk_reduce_add_u16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_u16_neon_contiguous_( //
    nk_u16_t const *data, nk_size_t count,           //
    nk_u16_t *min_value, nk_size_t *min_index) {
    uint16x8_t min_u16x8 = vdupq_n_u16(65535);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t data_u16x8 = vld1q_u16(data + idx);
        min_u16x8 = vminq_u16(min_u16x8, data_u16x8);
    }
    nk_u16_t best_val = vminvq_u16(min_u16x8);
    // Handle tail
    for (; idx < count; ++idx) {
        if (data[idx] < best_val) best_val = data[idx];
    }
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_u16_neon(                             //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8) nk_reduce_min_u16_neon_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_u16_neon_contiguous_( //
    nk_u16_t const *data, nk_size_t count,           //
    nk_u16_t *max_value, nk_size_t *max_index) {
    uint16x8_t max_u16x8 = vdupq_n_u16(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t data_u16x8 = vld1q_u16(data + idx);
        max_u16x8 = vmaxq_u16(max_u16x8, data_u16x8);
    }
    nk_u16_t best_val = vmaxvq_u16(max_u16x8);
    // Handle tail
    for (; idx < count; ++idx) {
        if (data[idx] > best_val) best_val = data[idx];
    }
    // Find first occurrence
    nk_size_t best_idx = 0;
    for (nk_size_t i = 0; i < count; ++i) {
        if (data[i] == best_val) {
            best_idx = i;
            break;
        }
    }
    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_u16_neon(                             //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8) nk_reduce_max_u16_neon_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEON_H
