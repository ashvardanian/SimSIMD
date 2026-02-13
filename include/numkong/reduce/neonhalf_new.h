/**
 *  @brief NEON FP16 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonhalf_new.h
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
#ifndef NK_REDUCE_NEONHALF_NEW_H
#define NK_REDUCE_NEONHALF_NEW_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial_new.h"

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
    nk_f16_t const *data, nk_size_t count,                   //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data + idx));
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
        sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4);
        sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, hi_f32x4, hi_f32x4);
    }

    // Scalar tail
    nk_f32_t s = vaddvq_f32(sum_f32x4);
    nk_f32_t sq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_f16_to_f32_serial(data + idx, &val);
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_f16_neonhalf_strided_(             //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, hi_f32x4, hi_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, hi_f32x4, hi_f32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
            float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
            sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
            sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4);
            sumsq_f32x4 = vfmaq_f32(sumsq_f32x4, hi_f32x4, hi_f32x4);
        }
    }

    // Scalar tail for remaining elements
    nk_f32_t s = vaddvq_f32(sum_f32x4);
    nk_f32_t sq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_f16_to_f32_serial((nk_f16_t const *)(data + idx * stride_elements), &val);
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_f16_neonhalf(                     //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_f16_neonhalf(data, left_partition_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_f16_neonhalf(data + left_partition_count * stride_elements, count - left_partition_count,
                                       stride_bytes, &right_sum_value, &right_sumsq_value);
        *sum = left_sum_value + right_sum_value, *sumsq = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_f16_neonhalf_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_f16_neonhalf_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_f16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f16_neonhalf_contiguous_( //
    nk_f16_t const *data, nk_size_t count,                  //
    nk_f16_t *min_value, nk_size_t *min_index,              //
    nk_f16_t *max_value, nk_size_t *max_index) {

    float16x8_t min_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)data);
    float16x8_t max_f16x8 = min_f16x8;
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(1);
    uint16x8_t one_u16x8 = vdupq_n_u16(1);

    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = vld1q_f16((nk_f16_for_arm_simd_t const *)(data + idx));
        uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
        min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
        min_iter_u16x8 = vbslq_u16(lt_u16x8, iter_u16x8, min_iter_u16x8);
        uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
        max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
        max_iter_u16x8 = vbslq_u16(gt_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }

    // Partial-load tail with identity masking
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, remaining);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        uint16x8_t lane_u16x8 = {0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        float16x8_t data_for_min = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        float16x8_t data_for_max = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);
        uint16x8_t lt_u16x8 = vcltq_f16(data_for_min, min_f16x8);
        min_f16x8 = vbslq_f16(lt_u16x8, data_for_min, min_f16x8);
        min_iter_u16x8 = vbslq_u16(lt_u16x8, iter_u16x8, min_iter_u16x8);
        uint16x8_t gt_u16x8 = vcgtq_f16(data_for_max, max_f16x8);
        max_f16x8 = vbslq_f16(gt_u16x8, data_for_max, max_f16x8);
        max_iter_u16x8 = vbslq_u16(gt_u16x8, iter_u16x8, max_iter_u16x8);
    }

    // Horizontal reduction: extract via union, scalar compare 8 lanes
    nk_b128_vec_t minimum_values_f16x8, maximum_values_f16x8, minimum_iteration_indices_u16x8,
        maximum_iteration_indices_u16x8;
    minimum_values_f16x8.u16x8 = vreinterpretq_u16_f16(min_f16x8);
    maximum_values_f16x8.u16x8 = vreinterpretq_u16_f16(max_f16x8);
    minimum_iteration_indices_u16x8.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_u16x8.u16x8 = max_iter_u16x8;

    nk_f16_t best_min = minimum_values_f16x8.f16s[0];
    nk_size_t best_min_idx = (nk_size_t)minimum_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(minimum_values_f16x8.f16s[i], best_min) < 0) {
            best_min = minimum_values_f16x8.f16s[i];
            best_min_idx = (nk_size_t)minimum_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    nk_f16_t best_max = maximum_values_f16x8.f16s[0];
    nk_size_t best_max_idx = (nk_size_t)maximum_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(maximum_values_f16x8.f16s[i], best_max) > 0) {
            best_max = maximum_values_f16x8.f16s[i];
            best_max_idx = (nk_size_t)maximum_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    *min_value = best_min, *min_index = best_min_idx;
    *max_value = best_max, *max_index = best_max_idx;
}

NK_INTERNAL void nk_reduce_minmax_f16_neonhalf_strided_(              //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f16_t *min_value, nk_size_t *min_index,                        //
    nk_f16_t *max_value, nk_size_t *max_index) {

    float16x8_t min_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(NK_F16_MAX));
    float16x8_t max_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(NK_F16_MIN));
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x2.val[0]);
            uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(lt_u16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(gt_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(lt_u16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(gt_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            uint16x8_t lt_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(lt_u16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(lt_u16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t gt_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(gt_u16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(gt_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
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
        float16x8_t data_for_min = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        float16x8_t data_for_max = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);

        uint16x8_t lt_u16x8 = vcltq_f16(data_for_min, min_f16x8);
        min_f16x8 = vbslq_f16(lt_u16x8, data_for_min, min_f16x8);
        min_iter_u16x8 = vbslq_u16(lt_u16x8, iter_u16x8, min_iter_u16x8);

        uint16x8_t gt_u16x8 = vcgtq_f16(data_for_max, max_f16x8);
        max_f16x8 = vbslq_f16(gt_u16x8, data_for_max, max_f16x8);
        max_iter_u16x8 = vbslq_u16(gt_u16x8, iter_u16x8, max_iter_u16x8);
    }

    // Horizontal reduction
    nk_b128_vec_t minimum_values_f16x8, maximum_values_f16x8, minimum_iteration_indices_u16x8,
        maximum_iteration_indices_u16x8;
    minimum_values_f16x8.u16x8 = vreinterpretq_u16_f16(min_f16x8);
    maximum_values_f16x8.u16x8 = vreinterpretq_u16_f16(max_f16x8);
    minimum_iteration_indices_u16x8.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_u16x8.u16x8 = max_iter_u16x8;

    nk_f16_t best_min = minimum_values_f16x8.f16s[0];
    nk_size_t best_min_idx = (nk_size_t)minimum_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(minimum_values_f16x8.f16s[i], best_min) < 0) {
            best_min = minimum_values_f16x8.f16s[i];
            best_min_idx = (nk_size_t)minimum_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    nk_f16_t best_max = maximum_values_f16x8.f16s[0];
    nk_size_t best_max_idx = (nk_size_t)maximum_iteration_indices_u16x8.u16s[0] * 8 + 0;
    for (int i = 1; i < 8; ++i) {
        if (nk_f16_compare_(maximum_values_f16x8.f16s[i], best_max) > 0) {
            best_max = maximum_values_f16x8.f16s[i];
            best_max_idx = (nk_size_t)maximum_iteration_indices_u16x8.u16s[i] * 8 + (nk_size_t)i;
        }
    }

    *min_value = best_min, *min_index = best_min_idx;
    *max_value = best_max, *max_index = best_max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f16_neonhalf(                      //
    nk_f16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value, nk_size_t *min_index,                     //
    nk_f16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);
    if (count == 0)
        *min_value = nk_f16_from_u16_(NK_F16_MAX), *min_index = NK_SIZE_MAX, *max_value = nk_f16_from_u16_(NK_F16_MIN),
        *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_f16_t left_min_value, right_min_value, left_max_value, right_max_value;
        nk_size_t left_min_index, right_min_index, left_max_index, right_max_index;
        nk_reduce_minmax_f16_neonhalf(data, left_partition_count, stride_bytes, &left_min_value, &left_min_index,
                                      &left_max_value, &left_max_index);
        nk_reduce_minmax_f16_neonhalf(data + left_partition_count * stride_elements, count - left_partition_count,
                                      stride_bytes, &right_min_value, &right_min_index, &right_max_value,
                                      &right_max_index);
        if (nk_f16_compare_(right_min_value, left_min_value) < 0)
            *min_value = right_min_value, *min_index = left_partition_count + right_min_index;
        else *min_value = left_min_value, *min_index = left_min_index;
        if (nk_f16_compare_(right_max_value, left_max_value) > 0)
            *max_value = right_max_value, *max_index = left_partition_count + right_max_index;
        else *max_value = left_max_value, *max_index = left_max_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_neonhalf_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_f16_neonhalf_strided_(data, count, stride_elements, min_value, min_index, max_value,
                                               max_index);
    else nk_reduce_minmax_f16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
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
#endif // NK_REDUCE_NEONHALF_NEW_H
