/**
 *  @brief SIMD-accelerated Vector Reductions for NEON FP16.
 *  @file include/numkong/reduce/neonhalf.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section reduce_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vld1q_f16                   LD1 (V.8H)                      4cy         2/cy        3/cy
 *      vaddq_f16                   FADD (V.8H, V.8H, V.8H)         2cy         2/cy        4/cy
 *      vadd_f16                    FADD (V.4H, V.4H, V.4H)         2cy         2/cy        4/cy
 *      vpadd_f16                   FADDP (V.4H, V.4H, V.4H)        2cy         2/cy        4/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy        4/cy
 *      vcltq_f16                   FCMLT (V.8H, V.8H, V.8H)        2cy         2/cy        4/cy
 *      vcgtq_f16                   FCMGT (V.8H, V.8H, V.8H)        2cy         2/cy        4/cy
 *      vbslq_f16                   BSL (V.16B, V.16B, V.16B)       2cy         2/cy        4/cy
 *      vmovl_u16                   UXTL (V.4S, V.4H)               2cy         2/cy        4/cy
 *      vbslq_u32                   BSL (V.16B, V.16B, V.16B)       2cy         2/cy        4/cy
 *
 *  The ARMv8.2-FP16 extension enables native half-precision reductions, processing 8 F16 elements
 *  per vector operation. For sum reductions, accumulation is performed in F16 then converted to F32
 *  for the final result to avoid overflow in large arrays.
 *
 *  Min/max reductions track both values and indices using F16 comparisons with 32-bit index vectors.
 *  The BSL (bitwise select) instruction enables branchless conditional updates of both value and
 *  index lanes based on comparison masks.
 */
#ifndef NK_REDUCE_NEONHALF_H
#define NK_REDUCE_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

/** @brief Horizontal sum of 8 f16s in a NEON register, returning f32. */
NK_INTERNAL nk_f32_t nk_reduce_add_f16x8_neonhalf_(float16x8_t sum_f16x8) {
    float16x4_t low_f16x4 = vget_low_f16(sum_f16x8);
    float16x4_t high_f16x4 = vget_high_f16(sum_f16x8);
    float16x4_t sum_f16x4 = vadd_f16(low_f16x4, high_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    return vgetq_lane_f32(vcvt_f32_f16(sum_f16x4), 0);
}

NK_INTERNAL void nk_reduce_add_f16_neonhalf_contiguous_( //
    nk_f16_t const *data, nk_size_t count, nk_f32_t *result) {
    float16x8_t sum_f16x8 = vdupq_n_f16(0);
    float16x8_t data_f16x8;
    nk_size_t idx = 0;

nk_reduce_add_f16_neonhalf_contiguous_cycle:
    if (idx + 8 <= count) {
        data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data + idx));
        idx += 8;
    }
    else {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, count - idx);
        data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        idx = count;
    }
    sum_f16x8 = vaddq_f16(sum_f16x8, data_f16x8);
    if (idx < count) goto nk_reduce_add_f16_neonhalf_contiguous_cycle;

    *result = nk_reduce_add_f16x8_neonhalf_(sum_f16x8);
}

NK_INTERNAL void nk_reduce_add_f16_neonhalf_strided_(                 //
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

    if (idx < count) {
        nk_b128_vec_t tail_vec = {{0}};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data + (idx + k) * stride_elements);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        sum_f16x8 = vaddq_f16(sum_f16x8, data_f16x8);
    }
    *result = nk_reduce_add_f16x8_neonhalf_(sum_f16x8);
}

NK_PUBLIC void nk_reduce_add_f16_neonhalf(                         //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (!aligned) nk_reduce_add_f16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_f16_neonhalf_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_f16_neonhalf_strided_(data, count, stride_elements, result);
    else nk_reduce_add_f16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_f16_neonhalf_contiguous_( //
    nk_f16_t const *data, nk_size_t count,               //
    nk_f32_t *min_value, nk_size_t *min_index) {

    float16x8_t min_f16x8 = vdupq_n_f16(NK_F32_MAX);
    uint32x4_t min_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    float16x8_t data_f16x8;
    nk_size_t idx = 0;

nk_reduce_min_f16_neonhalf_contiguous_cycle:
    if (idx + 8 <= count) {
        data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data + idx));
        idx += 8;
    }
    else {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, count - idx);
        data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        // Mask invalid lanes to current min (so they can't win comparison)
        nk_size_t valid = count - idx;
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)valid));
        data_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        idx = count;
    }
    uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
    uint16x4_t lt_low_u16x4 = vget_low_u16(lt_u16x8);
    uint16x4_t lt_high_u16x4 = vget_high_u16(lt_u16x8);
    uint32x4_t lt_low_u32x4 = vmovl_u16(lt_low_u16x4);
    uint32x4_t lt_high_u32x4 = vmovl_u16(lt_high_u16x4);
    min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
    min_idx_low_u32x4 = vbslq_u32(lt_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
    min_idx_high_u32x4 = vbslq_u32(lt_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
    idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
    idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    if (idx < count) goto nk_reduce_min_f16_neonhalf_contiguous_cycle;

    // Horizontal reduction: convert f16x8 to f32 and find best lane
    float32x4_t min_low_f32x4 = vcvt_f32_f16(vget_low_f16(min_f16x8));
    float32x4_t min_high_f32x4 = vcvt_f32_f16(vget_high_f16(min_f16x8));
    nk_b128_vec_t min_low_vec, min_high_vec, idx_low_vec, idx_high_vec;
    min_low_vec.f32x4 = min_low_f32x4;
    min_high_vec.f32x4 = min_high_f32x4;
    idx_low_vec.u32x4 = min_idx_low_u32x4;
    idx_high_vec.u32x4 = min_idx_high_u32x4;

    nk_f32_t best_val = min_low_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_low_vec.u32s[0];
    for (int i = 1; i < 4; ++i)
        if (min_low_vec.f32s[i] < best_val) best_val = min_low_vec.f32s[i], best_idx = (nk_size_t)idx_low_vec.u32s[i];
    for (int i = 0; i < 4; ++i)
        if (min_high_vec.f32s[i] < best_val)
            best_val = min_high_vec.f32s[i], best_idx = (nk_size_t)idx_high_vec.u32s[i];

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_f16_neonhalf_strided_(                 //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    float16x8_t min_f16x8 = vdupq_n_f16(NK_F32_MAX);
    uint32x4_t min_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            uint16x4_t lt_low_u16x4 = vget_low_u16(lt_u16x8);
            uint16x4_t lt_high_u16x4 = vget_high_u16(lt_u16x8);
            uint32x4_t lt_low_u32x4 = vmovl_u16(lt_low_u16x4);
            uint32x4_t lt_high_u32x4 = vmovl_u16(lt_high_u16x4);
            min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
            min_idx_low_u32x4 = vbslq_u32(lt_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
            min_idx_high_u32x4 = vbslq_u32(lt_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
            idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
            idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            uint16x4_t lt_low_u16x4 = vget_low_u16(lt_u16x8);
            uint16x4_t lt_high_u16x4 = vget_high_u16(lt_u16x8);
            uint32x4_t lt_low_u32x4 = vmovl_u16(lt_low_u16x4);
            uint32x4_t lt_high_u32x4 = vmovl_u16(lt_high_u16x4);
            min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
            min_idx_low_u32x4 = vbslq_u32(lt_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
            min_idx_high_u32x4 = vbslq_u32(lt_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
            idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
            idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            uint16x4_t lt_low_u16x4 = vget_low_u16(lt_u16x8);
            uint16x4_t lt_high_u16x4 = vget_high_u16(lt_u16x8);
            uint32x4_t lt_low_u32x4 = vmovl_u16(lt_low_u16x4);
            uint32x4_t lt_high_u32x4 = vmovl_u16(lt_high_u16x4);
            min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
            min_idx_low_u32x4 = vbslq_u32(lt_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
            min_idx_high_u32x4 = vbslq_u32(lt_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
            idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
            idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
        }
    }

    // Handle tail with gather-into-buffer
    if (idx < count) {
        nk_b128_vec_t tail_vec = {{0}};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data + (idx + k) * stride_elements);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        // Mask invalid lanes
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        data_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);

        uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
        uint16x4_t lt_low_u16x4 = vget_low_u16(lt_u16x8);
        uint16x4_t lt_high_u16x4 = vget_high_u16(lt_u16x8);
        uint32x4_t lt_low_u32x4 = vmovl_u16(lt_low_u16x4);
        uint32x4_t lt_high_u32x4 = vmovl_u16(lt_high_u16x4);
        min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
        min_idx_low_u32x4 = vbslq_u32(lt_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(lt_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
    }

    // Horizontal reduction
    float32x4_t min_low_f32x4 = vcvt_f32_f16(vget_low_f16(min_f16x8));
    float32x4_t min_high_f32x4 = vcvt_f32_f16(vget_high_f16(min_f16x8));
    nk_b128_vec_t min_low_vec, min_high_vec, idx_low_vec, idx_high_vec;
    min_low_vec.f32x4 = min_low_f32x4;
    min_high_vec.f32x4 = min_high_f32x4;
    idx_low_vec.u32x4 = min_idx_low_u32x4;
    idx_high_vec.u32x4 = min_idx_high_u32x4;

    nk_f32_t best_val = min_low_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_low_vec.u32s[0];
    for (int i = 1; i < 4; ++i)
        if (min_low_vec.f32s[i] < best_val) best_val = min_low_vec.f32s[i], best_idx = (nk_size_t)idx_low_vec.u32s[i];
    for (int i = 0; i < 4; ++i)
        if (min_high_vec.f32s[i] < best_val)
            best_val = min_high_vec.f32s[i], best_idx = (nk_size_t)idx_high_vec.u32s[i];

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_f16_neonhalf(                         //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_f16_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_f16_neonhalf_contiguous_(data, count, min_value, min_index);
    else if (stride_elements <= 4)
        nk_reduce_min_f16_neonhalf_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_f16_neonhalf_contiguous_( //
    nk_f16_t const *data, nk_size_t count,               //
    nk_f32_t *max_value, nk_size_t *max_index) {

    float16x8_t max_f16x8 = vdupq_n_f16(NK_F32_MIN);
    uint32x4_t max_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t max_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    float16x8_t data_f16x8;
    nk_size_t idx = 0;

nk_reduce_max_f16_neonhalf_contiguous_cycle:
    if (idx + 8 <= count) {
        data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data + idx));
        idx += 8;
    }
    else {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, count - idx);
        data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        // Mask invalid lanes to current max (so they can't win comparison)
        nk_size_t valid = count - idx;
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)valid));
        data_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);
        idx = count;
    }
    uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
    uint16x4_t gt_low_u16x4 = vget_low_u16(gt_u16x8);
    uint16x4_t gt_high_u16x4 = vget_high_u16(gt_u16x8);
    uint32x4_t gt_low_u32x4 = vmovl_u16(gt_low_u16x4);
    uint32x4_t gt_high_u32x4 = vmovl_u16(gt_high_u16x4);
    max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
    max_idx_low_u32x4 = vbslq_u32(gt_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
    max_idx_high_u32x4 = vbslq_u32(gt_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
    idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
    idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    if (idx < count) goto nk_reduce_max_f16_neonhalf_contiguous_cycle;

    // Horizontal reduction: convert f16x8 to f32 and find best lane
    float32x4_t max_low_f32x4 = vcvt_f32_f16(vget_low_f16(max_f16x8));
    float32x4_t max_high_f32x4 = vcvt_f32_f16(vget_high_f16(max_f16x8));
    nk_b128_vec_t max_low_vec, max_high_vec, idx_low_vec, idx_high_vec;
    max_low_vec.f32x4 = max_low_f32x4;
    max_high_vec.f32x4 = max_high_f32x4;
    idx_low_vec.u32x4 = max_idx_low_u32x4;
    idx_high_vec.u32x4 = max_idx_high_u32x4;

    nk_f32_t best_val = max_low_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_low_vec.u32s[0];
    for (int i = 1; i < 4; ++i)
        if (max_low_vec.f32s[i] > best_val) best_val = max_low_vec.f32s[i], best_idx = (nk_size_t)idx_low_vec.u32s[i];
    for (int i = 0; i < 4; ++i)
        if (max_high_vec.f32s[i] > best_val)
            best_val = max_high_vec.f32s[i], best_idx = (nk_size_t)idx_high_vec.u32s[i];

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_f16_neonhalf_strided_(                 //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    float16x8_t max_f16x8 = vdupq_n_f16(NK_F32_MIN);
    uint32x4_t max_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t max_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            uint16x4_t gt_low_u16x4 = vget_low_u16(gt_u16x8);
            uint16x4_t gt_high_u16x4 = vget_high_u16(gt_u16x8);
            uint32x4_t gt_low_u32x4 = vmovl_u16(gt_low_u16x4);
            uint32x4_t gt_high_u32x4 = vmovl_u16(gt_high_u16x4);
            max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
            max_idx_low_u32x4 = vbslq_u32(gt_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
            max_idx_high_u32x4 = vbslq_u32(gt_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
            idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
            idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            uint16x4_t gt_low_u16x4 = vget_low_u16(gt_u16x8);
            uint16x4_t gt_high_u16x4 = vget_high_u16(gt_u16x8);
            uint32x4_t gt_low_u32x4 = vmovl_u16(gt_low_u16x4);
            uint32x4_t gt_high_u32x4 = vmovl_u16(gt_high_u16x4);
            max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
            max_idx_low_u32x4 = vbslq_u32(gt_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
            max_idx_high_u32x4 = vbslq_u32(gt_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
            idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
            idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            uint16x4_t gt_low_u16x4 = vget_low_u16(gt_u16x8);
            uint16x4_t gt_high_u16x4 = vget_high_u16(gt_u16x8);
            uint32x4_t gt_low_u32x4 = vmovl_u16(gt_low_u16x4);
            uint32x4_t gt_high_u32x4 = vmovl_u16(gt_high_u16x4);
            max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
            max_idx_low_u32x4 = vbslq_u32(gt_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
            max_idx_high_u32x4 = vbslq_u32(gt_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
            idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
            idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
        }
    }

    // Handle tail with gather-into-buffer
    if (idx < count) {
        nk_b128_vec_t tail_vec = {{0}};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data + (idx + k) * stride_elements);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        // Mask invalid lanes
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        data_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);

        uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
        uint16x4_t gt_low_u16x4 = vget_low_u16(gt_u16x8);
        uint16x4_t gt_high_u16x4 = vget_high_u16(gt_u16x8);
        uint32x4_t gt_low_u32x4 = vmovl_u16(gt_low_u16x4);
        uint32x4_t gt_high_u32x4 = vmovl_u16(gt_high_u16x4);
        max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
        max_idx_low_u32x4 = vbslq_u32(gt_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(gt_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
    }

    // Horizontal reduction
    float32x4_t max_low_f32x4 = vcvt_f32_f16(vget_low_f16(max_f16x8));
    float32x4_t max_high_f32x4 = vcvt_f32_f16(vget_high_f16(max_f16x8));
    nk_b128_vec_t max_low_vec, max_high_vec, idx_low_vec, idx_high_vec;
    max_low_vec.f32x4 = max_low_f32x4;
    max_high_vec.f32x4 = max_high_f32x4;
    idx_low_vec.u32x4 = max_idx_low_u32x4;
    idx_high_vec.u32x4 = max_idx_high_u32x4;

    nk_f32_t best_val = max_low_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_low_vec.u32s[0];
    for (int i = 1; i < 4; ++i)
        if (max_low_vec.f32s[i] > best_val) best_val = max_low_vec.f32s[i], best_idx = (nk_size_t)idx_low_vec.u32s[i];
    for (int i = 0; i < 4; ++i)
        if (max_high_vec.f32s[i] > best_val)
            best_val = max_high_vec.f32s[i], best_idx = (nk_size_t)idx_high_vec.u32s[i];

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_f16_neonhalf(                         //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_f16_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_f16_neonhalf_contiguous_(data, count, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_max_f16_neonhalf_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f16_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONHALF_H
