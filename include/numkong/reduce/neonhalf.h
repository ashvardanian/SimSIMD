/**
 *  @brief NEON FP16 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonhalf.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section reduce_neonhalf_new_design Design Notes
 *
 *  Moments (sum + sum-of-squares) accumulate in f32 via vcvt_f32_f16 widening, giving
 *  full f32 precision. The contiguous path processes 8 f16 elements per iteration, widening
 *  to two f32x4 halves and using vfmaq_f32 for fused multiply-accumulate of squares.
 *
 *  Minmax tracks min/max values as native f16x8 with u16x8 iteration counters (same width
 *  as f16). The u16 counters wrap at 65536, so the dispatcher splits arrays larger than
 *  65536 * 8 = 524288 elements via recursive halving.
 */
#ifndef NK_REDUCE_NEONHALF_H
#define NK_REDUCE_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_INTERNAL void nk_reduce_moments_f16_neonhalf_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,               //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data_ptr + idx));
        float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
        sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
        sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
    }

    // Scalar tail
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    nk_f32_t sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_f16_to_f32_serial(data_ptr + idx, &value_f32);
        sum += value_f32, sumsq += value_f32 * value_f32;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_INTERNAL void nk_reduce_moments_f16_neonhalf_strided_(                 //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            float32x4_t low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, low_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, high_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, low_f32x4, low_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, high_f32x4, high_f32x4);
        }
    }

    // Scalar tail for remaining elements
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    nk_f32_t sumsq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t value_f32;
        nk_f16_to_f32_serial((nk_f16_t const *)(data_ptr + idx * stride_elements), &value_f32);
        sum += value_f32, sumsq += value_f32 * value_f32;
    }
    *sum_ptr = sum, *sumsq_ptr = sumsq;
}

NK_PUBLIC void nk_reduce_moments_f16_neonhalf(                         //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_f16_neonhalf(data_ptr, left_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_f16_neonhalf(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                       &right_sum_value, &right_sumsq_value);
        *sum_ptr = left_sum_value + right_sum_value, *sumsq_ptr = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_neonhalf_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else if (stride_elements <= 4)
        nk_reduce_moments_f16_neonhalf_strided_(data_ptr, count, stride_elements, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f16_neonhalf_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,              //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,      //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {

    float16x8_t min_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(NK_F16_MAX));
    float16x8_t max_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(NK_F16_MIN));
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t one_u16x8 = vdupq_n_u16(1);

    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data_ptr + idx));
        uint16x8_t less_b16x8 = vcltq_f16(data_f16x8, min_f16x8);
        min_f16x8 = vbslq_f16(less_b16x8, data_f16x8, min_f16x8);
        min_iter_u16x8 = vbslq_u16(less_b16x8, iter_u16x8, min_iter_u16x8);
        uint16x8_t greater_b16x8 = vcgtq_f16(data_f16x8, max_f16x8);
        max_f16x8 = vbslq_f16(greater_b16x8, data_f16x8, max_f16x8);
        max_iter_u16x8 = vbslq_u16(greater_b16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }

    // Partial-load tail with identity masking
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data_ptr + idx, &tail_vec, remaining);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        float16x8_t data_for_min = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        float16x8_t data_for_max = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);
        uint16x8_t less_b16x8 = vcltq_f16(data_for_min, min_f16x8);
        min_f16x8 = vbslq_f16(less_b16x8, data_for_min, min_f16x8);
        min_iter_u16x8 = vbslq_u16(less_b16x8, iter_u16x8, min_iter_u16x8);
        uint16x8_t greater_b16x8 = vcgtq_f16(data_for_max, max_f16x8);
        max_f16x8 = vbslq_f16(greater_b16x8, data_for_max, max_f16x8);
        max_iter_u16x8 = vbslq_u16(greater_b16x8, iter_u16x8, max_iter_u16x8);
    }

    // Horizontal reduction: extract via union, scalar compare 8 lanes
    nk_b128_vec_t min_values_f16x8, max_values_f16x8, min_iteration_indices_u16x8, max_iteration_indices_u16x8;
    min_values_f16x8.u16x8 = vreinterpretq_u16_f16(min_f16x8);
    max_values_f16x8.u16x8 = vreinterpretq_u16_f16(max_f16x8);
    min_iteration_indices_u16x8.u16x8 = min_iter_u16x8;
    max_iteration_indices_u16x8.u16x8 = max_iter_u16x8;

    nk_f16_t best_min = min_values_f16x8.f16s[0];
    nk_size_t min_index = (nk_size_t)min_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(min_values_f16x8.f16s[i], best_min) < 0) {
            best_min = min_values_f16x8.f16s[i];
            min_index = (nk_size_t)min_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    nk_f16_t best_max = max_values_f16x8.f16s[0];
    nk_size_t max_index = (nk_size_t)max_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(max_values_f16x8.f16s[i], best_max) > 0) {
            best_max = max_values_f16x8.f16s[i];
            max_index = (nk_size_t)max_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    *min_value_ptr = best_min, *min_index_ptr = min_index;
    *max_value_ptr = best_max, *max_index_ptr = max_index;
}

NK_INTERNAL void nk_reduce_minmax_f16_neonhalf_strided_(                  //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_elements, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                    //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {

    float16x8_t min_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(NK_F16_MAX));
    float16x8_t max_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(NK_F16_MIN));
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data_ptr + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            uint16x8_t less_b16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(less_b16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(less_b16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t greater_b16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(greater_b16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(greater_b16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            uint16x8_t less_b16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(less_b16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(less_b16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t greater_b16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(greater_b16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(greater_b16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            uint16x8_t less_b16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(less_b16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(less_b16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t greater_b16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(greater_b16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(greater_b16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }

    // Handle tail with gather-into-buffer
    if (idx < count) {
        nk_b128_vec_t tail_vec = {{0}};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data_ptr + (idx + k) * stride_elements);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        // Mask invalid lanes
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        float16x8_t data_for_min = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        float16x8_t data_for_max = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);

        uint16x8_t less_b16x8 = vcltq_f16(data_for_min, min_f16x8);
        min_f16x8 = vbslq_f16(less_b16x8, data_for_min, min_f16x8);
        min_iter_u16x8 = vbslq_u16(less_b16x8, iter_u16x8, min_iter_u16x8);

        uint16x8_t greater_b16x8 = vcgtq_f16(data_for_max, max_f16x8);
        max_f16x8 = vbslq_f16(greater_b16x8, data_for_max, max_f16x8);
        max_iter_u16x8 = vbslq_u16(greater_b16x8, iter_u16x8, max_iter_u16x8);
    }

    // Horizontal reduction
    nk_b128_vec_t min_values_f16x8, max_values_f16x8, min_iteration_indices_u16x8, max_iteration_indices_u16x8;
    min_values_f16x8.u16x8 = vreinterpretq_u16_f16(min_f16x8);
    max_values_f16x8.u16x8 = vreinterpretq_u16_f16(max_f16x8);
    min_iteration_indices_u16x8.u16x8 = min_iter_u16x8;
    max_iteration_indices_u16x8.u16x8 = max_iter_u16x8;

    nk_f16_t best_min = min_values_f16x8.f16s[0];
    nk_size_t min_index = (nk_size_t)min_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(min_values_f16x8.f16s[i], best_min) < 0) {
            best_min = min_values_f16x8.f16s[i];
            min_index = (nk_size_t)min_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    nk_f16_t best_max = max_values_f16x8.f16s[0];
    nk_size_t max_index = (nk_size_t)max_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(max_values_f16x8.f16s[i], best_max) > 0) {
            best_max = max_values_f16x8.f16s[i];
            max_index = (nk_size_t)max_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    *min_value_ptr = best_min, *min_index_ptr = min_index;
    *max_value_ptr = best_max, *max_index_ptr = max_index;
}

NK_PUBLIC void nk_reduce_minmax_f16_neonhalf(                          //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0)
        *min_value_ptr = nk_f16_from_u16_(NK_F16_MAX), *min_index_ptr = NK_SIZE_MAX,
        *max_value_ptr = nk_f16_from_u16_(NK_F16_MIN), *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_count = count / 2;
        nk_f16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f16_neonhalf(data_ptr, left_count, stride_bytes, &left_min_value, &left_min_index,
                                      &left_max_value, &left_max_index);
        nk_reduce_minmax_f16_neonhalf(data_ptr + left_count * stride_elements, count - left_count, stride_bytes,
                                      &right_min_value, &right_min_index, &right_max_value, &right_max_index);
        if (nk_f16_compare_(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_f16_compare_(right_max_value, left_max_value) > 0)
            *max_value_ptr = right_max_value, *max_index_ptr = left_count + right_max_index;
        else *max_value_ptr = left_max_value, *max_index_ptr = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_neonhalf_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                                  max_index_ptr);
    else if (stride_elements <= 4)
        nk_reduce_minmax_f16_neonhalf_strided_(data_ptr, count, stride_elements, min_value_ptr, min_index_ptr,
                                               max_value_ptr, max_index_ptr);
    else
        nk_reduce_minmax_f16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
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
