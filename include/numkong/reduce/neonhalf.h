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
        uint16x8_t less_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
        min_f16x8 = vbslq_f16(less_u16x8, data_f16x8, min_f16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        uint16x8_t greater_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
        max_f16x8 = vbslq_f16(greater_u16x8, data_f16x8, max_f16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }

    // Partial-load tail with identity masking
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data_ptr + idx, &tail_vec, remaining);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        uint16x8_t lane_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                             vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        float16x8_t data_for_min_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        float16x8_t data_for_max_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);
        uint16x8_t less_u16x8 = vcltq_f16(data_for_min_f16x8, min_f16x8);
        min_f16x8 = vbslq_f16(less_u16x8, data_for_min_f16x8, min_f16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        uint16x8_t greater_u16x8 = vcgtq_f16(data_for_max_f16x8, max_f16x8);
        max_f16x8 = vbslq_f16(greater_u16x8, data_for_max_f16x8, max_f16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }

    // Horizontal reduction: pairwise min/max across 8 f16 lanes to find the scalar extrema
    float16x4_t min_low_f16x4 = vget_low_f16(min_f16x8), min_high_f16x4 = vget_high_f16(min_f16x8);
    float16x4_t max_low_f16x4 = vget_low_f16(max_f16x8), max_high_f16x4 = vget_high_f16(max_f16x8);
    float16x4_t min_pairs_f16x4 = vpmin_f16(min_low_f16x4, min_high_f16x4);
    float16x4_t max_pairs_f16x4 = vpmax_f16(max_low_f16x4, max_high_f16x4);
    float16x4_t min_quads_f16x4 = vpmin_f16(min_pairs_f16x4, min_pairs_f16x4);
    float16x4_t max_quads_f16x4 = vpmax_f16(max_pairs_f16x4, max_pairs_f16x4);
    float16x4_t min_final_f16x4 = vpmin_f16(min_quads_f16x4, min_quads_f16x4);
    float16x4_t max_final_f16x4 = vpmax_f16(max_quads_f16x4, max_quads_f16x4);
    nk_f16_t min_value = vget_lane_f16(min_final_f16x4, 0), max_value = vget_lane_f16(max_final_f16x4, 0);

    // All-NaN / sentinel check: sentinels remain unchanged when all data is NaN.
    if (*(nk_u16_t *)&min_value == NK_F16_MAX && *(nk_u16_t *)&max_value == NK_F16_MIN) {
        *min_value_ptr = min_value, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = max_value,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    // Step 1: find all lanes matching the extremal value
    uint16x8_t min_value_match_u16x8 = vceqq_f16(min_f16x8, vcombine_f16(min_final_f16x4, min_final_f16x4));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_f16(max_f16x8, vcombine_f16(max_final_f16x4, max_final_f16x4));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);

    // Step 2: among lanes matching value AND earliest iteration, find smallest lane offset
    uint16x8_t lane_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                         vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
    uint16x8_t min_cycle_match_u16x8 = vceqq_u16(min_iter_u16x8, vdupq_n_u16(earliest_min_cycle));
    uint16x8_t min_both_match_u16x8 = vandq_u16(min_value_match_u16x8, min_cycle_match_u16x8);
    uint16x8_t min_masked_lanes_u16x8 = vbslq_u16(min_both_match_u16x8, lane_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t min_lane_offset = vminvq_u16(min_masked_lanes_u16x8);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)min_lane_offset;
    uint16x8_t max_cycle_match_u16x8 = vceqq_u16(max_iter_u16x8, vdupq_n_u16(earliest_max_cycle));
    uint16x8_t max_both_match_u16x8 = vandq_u16(max_value_match_u16x8, max_cycle_match_u16x8);
    uint16x8_t max_masked_lanes_u16x8 = vbslq_u16(max_both_match_u16x8, lane_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t max_lane_offset = vminvq_u16(max_masked_lanes_u16x8);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)max_lane_offset;

    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
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
            uint16x8_t less_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(less_u16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t greater_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(greater_u16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data_ptr + idx * 3));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x3.val[0]);
            uint16x8_t less_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(less_u16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t greater_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(greater_u16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data_ptr + idx * 4));
            float16x8_t data_f16x8 = vreinterpretq_f16_u16(loaded_u16x8x4.val[0]);
            uint16x8_t less_u16x8 = vcltq_f16(data_f16x8, min_f16x8);
            min_f16x8 = vbslq_f16(less_u16x8, data_f16x8, min_f16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            uint16x8_t greater_u16x8 = vcgtq_f16(data_f16x8, max_f16x8);
            max_f16x8 = vbslq_f16(greater_u16x8, data_f16x8, max_f16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }

    // Handle tail with gather-into-buffer
    if (idx < count) {
        nk_b128_vec_t tail_vec = {0};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data_ptr + (idx + k) * stride_elements);
        float16x8_t data_f16x8 = vreinterpretq_f16_u16(tail_vec.u16x8);
        // Mask invalid lanes
        uint16x8_t lane_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                             vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
        uint16x8_t valid_u16x8 = vcltq_u16(lane_u16x8, vdupq_n_u16((uint16_t)remaining));
        float16x8_t data_for_min_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, min_f16x8);
        float16x8_t data_for_max_f16x8 = vbslq_f16(valid_u16x8, data_f16x8, max_f16x8);

        uint16x8_t less_u16x8 = vcltq_f16(data_for_min_f16x8, min_f16x8);
        min_f16x8 = vbslq_f16(less_u16x8, data_for_min_f16x8, min_f16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);

        uint16x8_t greater_u16x8 = vcgtq_f16(data_for_max_f16x8, max_f16x8);
        max_f16x8 = vbslq_f16(greater_u16x8, data_for_max_f16x8, max_f16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }

    // Horizontal reduction: pairwise min/max across 8 f16 lanes to find the scalar extrema
    float16x4_t min_low_f16x4 = vget_low_f16(min_f16x8), min_high_f16x4 = vget_high_f16(min_f16x8);
    float16x4_t max_low_f16x4 = vget_low_f16(max_f16x8), max_high_f16x4 = vget_high_f16(max_f16x8);
    float16x4_t min_pairs_f16x4 = vpmin_f16(min_low_f16x4, min_high_f16x4);
    float16x4_t max_pairs_f16x4 = vpmax_f16(max_low_f16x4, max_high_f16x4);
    float16x4_t min_quads_f16x4 = vpmin_f16(min_pairs_f16x4, min_pairs_f16x4);
    float16x4_t max_quads_f16x4 = vpmax_f16(max_pairs_f16x4, max_pairs_f16x4);
    float16x4_t min_final_f16x4 = vpmin_f16(min_quads_f16x4, min_quads_f16x4);
    float16x4_t max_final_f16x4 = vpmax_f16(max_quads_f16x4, max_quads_f16x4);
    nk_f16_t min_value = vget_lane_f16(min_final_f16x4, 0), max_value = vget_lane_f16(max_final_f16x4, 0);

    // All-NaN / sentinel check: sentinels remain unchanged when all data is NaN.
    if (*(nk_u16_t *)&min_value == NK_F16_MAX && *(nk_u16_t *)&max_value == NK_F16_MIN) {
        *min_value_ptr = min_value, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = max_value,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    // Step 1: find all lanes matching the extremal value
    uint16x8_t min_value_match_u16x8 = vceqq_f16(min_f16x8, vcombine_f16(min_final_f16x4, min_final_f16x4));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_f16(max_f16x8, vcombine_f16(max_final_f16x4, max_final_f16x4));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);

    // Step 2: among lanes matching value AND earliest iteration, find smallest lane offset
    uint16x8_t lane_u16x8 = vcombine_u16(vreinterpret_u16_u64(vcreate_u64(0x0003000200010000ULL)),
                                         vreinterpret_u16_u64(vcreate_u64(0x0007000600050004ULL)));
    uint16x8_t min_cycle_match_u16x8 = vceqq_u16(min_iter_u16x8, vdupq_n_u16(earliest_min_cycle));
    uint16x8_t min_both_match_u16x8 = vandq_u16(min_value_match_u16x8, min_cycle_match_u16x8);
    uint16x8_t min_masked_lanes_u16x8 = vbslq_u16(min_both_match_u16x8, lane_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t min_lane_offset = vminvq_u16(min_masked_lanes_u16x8);
    nk_size_t min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)min_lane_offset;
    uint16x8_t max_cycle_match_u16x8 = vceqq_u16(max_iter_u16x8, vdupq_n_u16(earliest_max_cycle));
    uint16x8_t max_both_match_u16x8 = vandq_u16(max_value_match_u16x8, max_cycle_match_u16x8);
    uint16x8_t max_masked_lanes_u16x8 = vbslq_u16(max_both_match_u16x8, lane_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t max_lane_offset = vminvq_u16(max_masked_lanes_u16x8);
    nk_size_t max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)max_lane_offset;

    *min_value_ptr = min_value, *min_index_ptr = min_idx;
    *max_value_ptr = max_value, *max_index_ptr = max_idx;
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
        if (nk_f16_order_serial(right_min_value, left_min_value) < 0)
            *min_value_ptr = right_min_value, *min_index_ptr = left_count + right_min_index;
        else *min_value_ptr = left_min_value, *min_index_ptr = left_min_index;
        if (nk_f16_order_serial(right_max_value, left_max_value) > 0)
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
