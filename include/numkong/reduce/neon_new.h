/**
 *  @brief Base NEON (ARMv8-A) implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neon_new.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEON_NEW_H
#define NK_REDUCE_NEON_NEW_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_INTERNAL nk_u64_t nk_reduce_sadd_u64x2_neon_(uint64x2_t v) {
    uint64x2_t swapped_u64x2 = vextq_u64(v, v, 1);
    return vgetq_lane_u64(vqaddq_u64(v, swapped_u64x2), 0);
}

/** @brief Saturating square of each i64 lane → u64. If |a| >= 2^32, a² overflows u64 → saturate. */
NK_INTERNAL uint64x2_t nk_i64_smul_sq_i64x2_neon_(int64x2_t val) {
    uint64x2_t absolute_u64x2 = vreinterpretq_u64_s64(vabsq_s64(val));
    uint32x2_t low_halves_u32x2 = vmovn_u64(absolute_u64x2);
    uint64x2_t high_bits_u64x2 = vshrq_n_u64(absolute_u64x2, 32);
    uint64x2_t low_squared_u64x2 = vmull_u32(low_halves_u32x2, low_halves_u32x2);
    uint64x2_t is_small_u64x2 = vceqq_u64(high_bits_u64x2, vdupq_n_u64(0));
    return vbslq_u64(is_small_u64x2, low_squared_u64x2, vdupq_n_u64(NK_U64_MAX));
}

/** @brief Saturating square of each u64 lane → u64. If a >= 2^32, a² overflows u64 → saturate. */
NK_INTERNAL uint64x2_t nk_u64_smul_sq_u64x2_neon_(uint64x2_t val) {
    uint32x2_t low_halves_u32x2 = vmovn_u64(val);
    uint64x2_t high_bits_u64x2 = vshrq_n_u64(val, 32);
    uint64x2_t low_squared_u64x2 = vmull_u32(low_halves_u32x2, low_halves_u32x2);
    uint64x2_t is_small_u64x2 = vceqq_u64(high_bits_u64x2, vdupq_n_u64(0));
    return vbslq_u64(is_small_u64x2, low_squared_u64x2, vdupq_n_u64(NK_U64_MAX));
}

NK_INTERNAL void nk_reduce_moments_f32_neon_contiguous_( //
    nk_f32_t const *data, nk_size_t count,               //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0), sumsq_f64x2 = vdupq_n_f64(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        float32x4_t d = vld1q_f32(data + idx);
        float64x2_t lo = vcvt_f64_f32(vget_low_f32(d));
        float64x2_t hi = vcvt_f64_f32(vget_high_f32(d));
        sum_f64x2 = vaddq_f64(sum_f64x2, lo);
        sum_f64x2 = vaddq_f64(sum_f64x2, hi);
        sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, lo, lo);
        sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, hi, hi);
    }
    nk_f64_t s = vaddvq_f64(sum_f64x2), sq = vaddvq_f64(sumsq_f64x2);
    for (; idx < count; ++idx) {
        nk_f64_t val = (nk_f64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_f32_neon_strided_(                 //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0), sumsq_f64x2 = vdupq_n_f64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x2_t ld = vld2q_f32(data + idx * 2);
            float64x2_t lo = vcvt_f64_f32(vget_low_f32(ld.val[0]));
            float64x2_t hi = vcvt_f64_f32(vget_high_f32(ld.val[0]));
            sum_f64x2 = vaddq_f64(sum_f64x2, lo);
            sum_f64x2 = vaddq_f64(sum_f64x2, hi);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, lo, lo);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, hi, hi);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x3_t ld = vld3q_f32(data + idx * 3);
            float64x2_t lo = vcvt_f64_f32(vget_low_f32(ld.val[0]));
            float64x2_t hi = vcvt_f64_f32(vget_high_f32(ld.val[0]));
            sum_f64x2 = vaddq_f64(sum_f64x2, lo);
            sum_f64x2 = vaddq_f64(sum_f64x2, hi);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, lo, lo);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, hi, hi);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x4_t ld = vld4q_f32(data + idx * 4);
            float64x2_t lo = vcvt_f64_f32(vget_low_f32(ld.val[0]));
            float64x2_t hi = vcvt_f64_f32(vget_high_f32(ld.val[0]));
            sum_f64x2 = vaddq_f64(sum_f64x2, lo);
            sum_f64x2 = vaddq_f64(sum_f64x2, hi);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, lo, lo);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, hi, hi);
        }
    }
    nk_f64_t s = vaddvq_f64(sum_f64x2), sq = vaddvq_f64(sumsq_f64x2);
    nk_f32_t const *ptr = data + idx * stride_elements;
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f64_t val = (nk_f64_t)(*ptr);
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_f32_neon(                         //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_partition_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f32_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f32_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f32_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_f32_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f32_neon_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index,          //
    nk_f32_t *max_value, nk_size_t *max_index) {
    float32x4_t min_f32x4 = vld1q_f32(data), max_f32x4 = min_f32x4;
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(1), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 4;
    for (; idx + 4 <= count; idx += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx);
        uint32x4_t less_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
        uint32x4_t greater_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
        min_f32x4 = vbslq_f32(less_u32x4, data_f32x4, min_f32x4);
        max_f32x4 = vbslq_f32(greater_u32x4, data_f32x4, max_f32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
        iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b32x4_serial_(data + idx, &tail_vec, remaining);
        uint32x4_t lane_u32x4 = {0, 1, 2, 3};
        uint32x4_t valid_u32x4 = vcltq_u32(lane_u32x4, vdupq_n_u32((uint32_t)remaining));
        float32x4_t data_min_f32x4 = vbslq_f32(valid_u32x4, tail_vec.f32x4, min_f32x4);
        float32x4_t data_max_f32x4 = vbslq_f32(valid_u32x4, tail_vec.f32x4, max_f32x4);
        uint32x4_t less_u32x4 = vcltq_f32(data_min_f32x4, min_f32x4);
        uint32x4_t greater_u32x4 = vcgtq_f32(data_max_f32x4, max_f32x4);
        min_f32x4 = vbslq_f32(less_u32x4, data_min_f32x4, min_f32x4);
        max_f32x4 = vbslq_f32(greater_u32x4, data_max_f32x4, max_f32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    }
    nk_f32_t minimum_scalar = vminvq_f32(min_f32x4), maximum_scalar = vmaxvq_f32(max_f32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_f32(min_f32x4, vdupq_n_f32(minimum_scalar));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_f32(max_f32x4, vdupq_n_f32(maximum_scalar));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.f32x4 = min_f32x4;
    maximum_values_vec.f32x4 = max_f32x4;
    minimum_iteration_indices_vec.u32x4 = min_iter_u32x4;
    maximum_iteration_indices_vec.u32x4 = max_iter_u32x4;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 4; ++i)
        if (minimum_values_vec.f32s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u32s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 4; ++i)
        if (maximum_values_vec.f32s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u32s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_f32_neon_strided_(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index,                        //
    nk_f32_t *max_value, nk_size_t *max_index) {
    float32x4_t min_f32x4 = vdupq_n_f32(NK_F32_MAX), max_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x2_t loaded_f32x4x2 = vld2q_f32(data + idx * 2);
            float32x4_t data_f32x4 = loaded_f32x4x2.val[0];
            uint32x4_t less_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            uint32x4_t greater_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            min_f32x4 = vbslq_f32(less_u32x4, data_f32x4, min_f32x4);
            max_f32x4 = vbslq_f32(greater_u32x4, data_f32x4, max_f32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x3_t loaded_f32x4x3 = vld3q_f32(data + idx * 3);
            float32x4_t data_f32x4 = loaded_f32x4x3.val[0];
            uint32x4_t less_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            uint32x4_t greater_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            min_f32x4 = vbslq_f32(less_u32x4, data_f32x4, min_f32x4);
            max_f32x4 = vbslq_f32(greater_u32x4, data_f32x4, max_f32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            float32x4x4_t loaded_f32x4x4 = vld4q_f32(data + idx * 4);
            float32x4_t data_f32x4 = loaded_f32x4x4.val[0];
            uint32x4_t less_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            uint32x4_t greater_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            min_f32x4 = vbslq_f32(less_u32x4, data_f32x4, min_f32x4);
            max_f32x4 = vbslq_f32(greater_u32x4, data_f32x4, max_f32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    nk_f32_t minimum_scalar = vminvq_f32(min_f32x4), maximum_scalar = vmaxvq_f32(max_f32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_f32(min_f32x4, vdupq_n_f32(minimum_scalar));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_f32(max_f32x4, vdupq_n_f32(maximum_scalar));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.f32x4 = min_f32x4;
    maximum_values_vec.f32x4 = max_f32x4;
    minimum_iteration_indices_vec.u32x4 = min_iter_u32x4;
    maximum_iteration_indices_vec.u32x4 = max_iter_u32x4;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 4; ++i)
        if (minimum_values_vec.f32s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u32s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 4; ++i)
        if (maximum_values_vec.f32s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u32s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_f32_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f32_neon(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value = NK_F32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F32_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_f32_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                  &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_f32_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                  &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_f32_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count,               //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    float64x2_t s = vdupq_n_f64(0), sc = vdupq_n_f64(0);
    float64x2_t sq = vdupq_n_f64(0), sqc = vdupq_n_f64(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        float64x2_t v = vld1q_f64(data + idx);
        float64x2_t t = vaddq_f64(s, v);
        float64x2_t r = vsubq_f64(t, s);
        sc = vaddq_f64(sc, vaddq_f64(vsubq_f64(s, vsubq_f64(t, r)), vsubq_f64(v, r)));
        s = t;
        float64x2_t v2 = vmulq_f64(v, v);
        float64x2_t ts = vaddq_f64(sq, v2);
        float64x2_t right_sum = vsubq_f64(ts, sq);
        sqc = vaddq_f64(sqc, vaddq_f64(vsubq_f64(sq, vsubq_f64(ts, right_sum)), vsubq_f64(v2, right_sum)));
        sq = ts;
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tv;
        nk_partial_load_b64x2_serial_(data + idx, &tv, remaining);
        float64x2_t v = tv.f64x2;
        float64x2_t t = vaddq_f64(s, v);
        float64x2_t r = vsubq_f64(t, s);
        sc = vaddq_f64(sc, vaddq_f64(vsubq_f64(s, vsubq_f64(t, r)), vsubq_f64(v, r)));
        s = t;
        float64x2_t v2 = vmulq_f64(v, v);
        float64x2_t ts = vaddq_f64(sq, v2);
        float64x2_t right_sum = vsubq_f64(ts, sq);
        sqc = vaddq_f64(sqc, vaddq_f64(vsubq_f64(sq, vsubq_f64(ts, right_sum)), vsubq_f64(v2, right_sum)));
        sq = ts;
    }
    *sum = vaddvq_f64(vaddq_f64(s, sc));
    *sumsq = vaddvq_f64(vaddq_f64(sq, sqc));
}

NK_PUBLIC void nk_reduce_moments_f64_neon(                         //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 2) {
        nk_size_t left_partition_count = count / 2;
        nk_f64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_f64_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_f64_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_f64_neon_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f64_neon_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *min_value, nk_size_t *min_index,          //
    nk_f64_t *max_value, nk_size_t *max_index) {
    float64x2_t min_f64x2 = vld1q_f64(data), max_f64x2 = min_f64x2;
    uint64x2_t min_iter = vdupq_n_u64(0), max_iter = vdupq_n_u64(0);
    uint64x2_t iter = vdupq_n_u64(1), one = vdupq_n_u64(1);
    nk_size_t idx = 2;
    for (; idx + 2 <= count; idx += 2) {
        float64x2_t d = vld1q_f64(data + idx);
        uint64x2_t lt = vcltq_f64(d, min_f64x2), gt = vcgtq_f64(d, max_f64x2);
        min_f64x2 = vbslq_f64(lt, d, min_f64x2);
        max_f64x2 = vbslq_f64(gt, d, max_f64x2);
        min_iter = vbslq_u64(lt, iter, min_iter);
        max_iter = vbslq_u64(gt, iter, max_iter);
        iter = vaddq_u64(iter, one);
    }
    nk_b128_vec_t mv, xv, mi, xi;
    mv.f64x2 = min_f64x2;
    mi.u64x2 = min_iter;
    xv.f64x2 = max_f64x2;
    xi.u64x2 = max_iter;
    nk_f64_t min_s, max_s;
    nk_size_t min_idx, max_idx;
    if (mv.f64s[0] <= mv.f64s[1]) min_s = mv.f64s[0], min_idx = (nk_size_t)mi.u64s[0] * 2;
    else min_s = mv.f64s[1], min_idx = (nk_size_t)mi.u64s[1] * 2 + 1;
    if (xv.f64s[0] >= xv.f64s[1]) max_s = xv.f64s[0], max_idx = (nk_size_t)xi.u64s[0] * 2;
    else max_s = xv.f64s[1], max_idx = (nk_size_t)xi.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_f64_t val = data[idx];
        if (val < min_s) min_s = val, min_idx = idx;
        if (val > max_s) max_s = val, max_idx = idx;
    }
    *min_value = min_s, *min_index = min_idx;
    *max_value = max_s, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_f64_neon(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0)
        *min_value = NK_F64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F64_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i8_neon_contiguous_( //
    nk_i8_t const *data, nk_size_t count,               //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data + idx);
        int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
        sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
        int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
        int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
    }
    nk_i64_t s = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        s += val, sq += (nk_u64_t)(val * val);
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_i8_neon_strided_(                 //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x2_t loaded_i8x16x2 = vld2q_s8(data + idx * 2);
            int8x16_t data_i8x16 = loaded_i8x16x2.val[0];
            int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x3_t loaded_i8x16x3 = vld3q_s8(data + idx * 3);
            int8x16_t data_i8x16 = loaded_i8x16x3.val[0];
            int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x4_t loaded_i8x16x4 = vld4q_s8(data + idx * 4);
            int8x16_t data_i8x16 = loaded_i8x16x4.val[0];
            int16x8_t pairwise_i16x8 = vpaddlq_s8(data_i8x16);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(data_i8x16), vget_low_s8(data_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(data_i8x16, data_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    nk_i64_t s = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx * stride_elements];
        s += val, sq += (nk_u64_t)(val * val);
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i8_neon(                         //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i8_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i8_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_sum, &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i8_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_i8_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i8_neon_contiguous_( //
    nk_i8_t const *data, nk_size_t count,              //
    nk_i8_t *min_value, nk_size_t *min_index,          //
    nk_i8_t *max_value, nk_size_t *max_index) {
    int8x16_t min_i8x16 = vld1q_s8(data), max_i8x16 = min_i8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        int8x16_t data_i8x16 = vld1q_s8(data + idx);
        uint8x16_t less_u8x16 = vcltq_s8(data_i8x16, min_i8x16);
        uint8x16_t greater_u8x16 = vcgtq_s8(data_i8x16, max_i8x16);
        min_i8x16 = vbslq_s8(less_u8x16, data_i8x16, min_i8x16);
        max_i8x16 = vbslq_s8(greater_u8x16, data_i8x16, max_i8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data + idx, &tail_vec, remaining);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u8x16 = vdupq_n_u8(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u8s[i] = 0xFF;
        int8x16_t data_for_min_i8x16 = vbslq_s8(valid_mask_vec.u8x16, tail_vec.i8x16, vdupq_n_s8(NK_I8_MAX));
        int8x16_t data_for_max_i8x16 = vbslq_s8(valid_mask_vec.u8x16, tail_vec.i8x16, vdupq_n_s8(NK_I8_MIN));
        uint8x16_t less_u8x16 = vcltq_s8(data_for_min_i8x16, min_i8x16);
        uint8x16_t greater_u8x16 = vcgtq_s8(data_for_max_i8x16, max_i8x16);
        min_i8x16 = vbslq_s8(less_u8x16, data_for_min_i8x16, min_i8x16);
        max_i8x16 = vbslq_s8(greater_u8x16, data_for_max_i8x16, max_i8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_i8_t minimum_scalar = vminvq_s8(min_i8x16), maximum_scalar = vmaxvq_s8(max_i8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_s8(min_i8x16, vdupq_n_s8(minimum_scalar));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_s8(max_i8x16, vdupq_n_s8(maximum_scalar));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i8x16 = min_i8x16;
    maximum_values_vec.i8x16 = max_i8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.i8s[i] == minimum_scalar && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.i8s[i] == maximum_scalar && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_i8_neon_strided_(                  //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i8_t *min_value, nk_size_t *min_index,                        //
    nk_i8_t *max_value, nk_size_t *max_index) {
    int8x16_t min_i8x16 = vdupq_n_s8(NK_I8_MAX), max_i8x16 = vdupq_n_s8(NK_I8_MIN);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x2_t loaded_i8x16x2 = vld2q_s8(data + idx * 2);
            int8x16_t data_i8x16 = loaded_i8x16x2.val[0];
            uint8x16_t less_u8x16 = vcltq_s8(data_i8x16, min_i8x16);
            uint8x16_t greater_u8x16 = vcgtq_s8(data_i8x16, max_i8x16);
            min_i8x16 = vbslq_s8(less_u8x16, data_i8x16, min_i8x16);
            max_i8x16 = vbslq_s8(greater_u8x16, data_i8x16, max_i8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x3_t loaded_i8x16x3 = vld3q_s8(data + idx * 3);
            int8x16_t data_i8x16 = loaded_i8x16x3.val[0];
            uint8x16_t less_u8x16 = vcltq_s8(data_i8x16, min_i8x16);
            uint8x16_t greater_u8x16 = vcgtq_s8(data_i8x16, max_i8x16);
            min_i8x16 = vbslq_s8(less_u8x16, data_i8x16, min_i8x16);
            max_i8x16 = vbslq_s8(greater_u8x16, data_i8x16, max_i8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            int8x16x4_t loaded_i8x16x4 = vld4q_s8(data + idx * 4);
            int8x16_t data_i8x16 = loaded_i8x16x4.val[0];
            uint8x16_t less_u8x16 = vcltq_s8(data_i8x16, min_i8x16);
            uint8x16_t greater_u8x16 = vcgtq_s8(data_i8x16, max_i8x16);
            min_i8x16 = vbslq_s8(less_u8x16, data_i8x16, min_i8x16);
            max_i8x16 = vbslq_s8(greater_u8x16, data_i8x16, max_i8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    nk_i8_t minimum_scalar = vminvq_s8(min_i8x16), maximum_scalar = vmaxvq_s8(max_i8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_s8(min_i8x16, vdupq_n_s8(minimum_scalar));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_s8(max_i8x16, vdupq_n_s8(maximum_scalar));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i8x16 = min_i8x16;
    maximum_values_vec.i8x16 = max_i8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.i8s[i] == minimum_scalar && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.i8s[i] == maximum_scalar && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_i8_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i8_neon(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index,                     //
    nk_i8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);
    if (count == 0) *min_value = NK_I8_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I8_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_i8_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_i8_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                 &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_i8_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                 stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                 &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_i8_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u8_neon_contiguous_( //
    nk_u8_t const *data, nk_size_t count,               //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t d = vld1q_u8(data + idx);
        uint16x8_t sum16 = vpaddlq_u8(d);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(sum32));
        uint16x8_t sq_lo = vmull_u8(vget_low_u8(d), vget_low_u8(d));
        uint16x8_t sq_hi = vmull_high_u8(d, d);
        uint32x4_t sq32_lo = vpaddlq_u16(sq_lo);
        uint32x4_t sq32_hi = vpaddlq_u16(sq_hi);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq32_lo));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq32_hi));
    }
    nk_u64_t s = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_u8_neon_strided_(                 //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8(data + idx * 2);
            uint8x16_t data_u8x16 = loaded_u8x16x2.val[0];
            uint16x8_t pairwise_u16x8 = vpaddlq_u8(data_u8x16);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(vpaddlq_u16(pairwise_u16x8)));
            uint16x8_t squares_lo_u16x8 = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
            uint16x8_t squares_hi_u16x8 = vmull_high_u8(data_u8x16, data_u8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_lo_u16x8)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_hi_u16x8)));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8(data + idx * 3);
            uint8x16_t data_u8x16 = loaded_u8x16x3.val[0];
            uint16x8_t pairwise_u16x8 = vpaddlq_u8(data_u8x16);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(vpaddlq_u16(pairwise_u16x8)));
            uint16x8_t squares_lo_u16x8 = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
            uint16x8_t squares_hi_u16x8 = vmull_high_u8(data_u8x16, data_u8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_lo_u16x8)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_hi_u16x8)));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8(data + idx * 4);
            uint8x16_t data_u8x16 = loaded_u8x16x4.val[0];
            uint16x8_t pairwise_u16x8 = vpaddlq_u8(data_u8x16);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(vpaddlq_u16(pairwise_u16x8)));
            uint16x8_t squares_lo_u16x8 = vmull_u8(vget_low_u8(data_u8x16), vget_low_u8(data_u8x16));
            uint16x8_t squares_hi_u16x8 = vmull_high_u8(data_u8x16, data_u8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_lo_u16x8)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(squares_hi_u16x8)));
        }
    }
    nk_u64_t s = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx * stride_elements];
        s += val, sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u8_neon(                         //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u8_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u8_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_sum, &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u8_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_u8_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u8_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u8_neon_contiguous_( //
    nk_u8_t const *data, nk_size_t count,              //
    nk_u8_t *min_value, nk_size_t *min_index,          //
    nk_u8_t *max_value, nk_size_t *max_index) {
    uint8x16_t min_u8x16 = vld1q_u8(data), max_u8x16 = min_u8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t data_u8x16 = vld1q_u8(data + idx);
        uint8x16_t less_u8x16 = vcltq_u8(data_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data + idx, &tail_vec, remaining);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u8x16 = vdupq_n_u8(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u8s[i] = 0xFF;
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_mask_vec.u8x16, tail_vec.u8x16, vdupq_n_u8(NK_U8_MAX));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_mask_vec.u8x16, tail_vec.u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t minimum_scalar = vminvq_u8(min_u8x16), maximum_scalar = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(minimum_scalar));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(maximum_scalar));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == minimum_scalar && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == maximum_scalar && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_u8_neon_strided_(                  //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u8_t *min_value, nk_size_t *min_index,                        //
    nk_u8_t *max_value, nk_size_t *max_index) {
    uint8x16_t min_u8x16 = vdupq_n_u8(NK_U8_MAX), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)data + idx * 2);
            uint8x16_t data_u8x16 = loaded_u8x16x2.val[0];
            uint8x16_t less_u8x16 = vcltq_u8(data_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(data_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, data_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, data_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)data + idx * 3);
            uint8x16_t data_u8x16 = loaded_u8x16x3.val[0];
            uint8x16_t less_u8x16 = vcltq_u8(data_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(data_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, data_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, data_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)data + idx * 4);
            uint8x16_t data_u8x16 = loaded_u8x16x4.val[0];
            uint8x16_t less_u8x16 = vcltq_u8(data_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(data_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, data_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, data_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    nk_u8_t minimum_scalar = vminvq_u8(min_u8x16), maximum_scalar = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(minimum_scalar));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(maximum_scalar));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == minimum_scalar && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == maximum_scalar && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u8_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u8_neon(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index,                     //
    nk_u8_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);
    if (count == 0) *min_value = NK_U8_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U8_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_u8_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_u8_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                 &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_u8_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                 stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                 &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_u8_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u8_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i16_neon_contiguous_( //
    nk_i16_t const *data, nk_size_t count,               //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t d = vld1q_s16(data + idx);
        int32x4_t sum32 = vpaddlq_s16(d);
        sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(sum32));
        // sumsq: widening multiply i16*i16 -> i32, then widen to u64
        int32x4_t sq_lo = vmull_s16(vget_low_s16(d), vget_low_s16(d));
        int32x4_t sq_hi = vmull_high_s16(d, d);
        // i16*i16 squares are always non-negative, safe to reinterpret as u32
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(sq_lo)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(sq_hi)));
    }
    nk_i64_t s = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        s += val;
        sq += (nk_u64_t)(val * val);
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_i16_neon_strided_(                 //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x2_t loaded_i16x8x2 = vld2q_s16(data + idx * 2);
            int16x8_t data_i16x8 = loaded_i16x8x2.val[0];
            int32x4_t pairwise_i32x4 = vpaddlq_s16(data_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(pairwise_i32x4));
            int32x4_t squares_lo_i32x4 = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
            int32x4_t squares_hi_i32x4 = vmull_high_s16(data_i16x8, data_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_i32x4)));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x3_t loaded_i16x8x3 = vld3q_s16(data + idx * 3);
            int16x8_t data_i16x8 = loaded_i16x8x3.val[0];
            int32x4_t pairwise_i32x4 = vpaddlq_s16(data_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(pairwise_i32x4));
            int32x4_t squares_lo_i32x4 = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
            int32x4_t squares_hi_i32x4 = vmull_high_s16(data_i16x8, data_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_i32x4)));
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x4_t loaded_i16x8x4 = vld4q_s16(data + idx * 4);
            int16x8_t data_i16x8 = loaded_i16x8x4.val[0];
            int32x4_t pairwise_i32x4 = vpaddlq_s16(data_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(pairwise_i32x4));
            int32x4_t squares_lo_i32x4 = vmull_s16(vget_low_s16(data_i16x8), vget_low_s16(data_i16x8));
            int32x4_t squares_hi_i32x4 = vmull_high_s16(data_i16x8, data_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_i32x4)));
        }
    }
    nk_i64_t s = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx * stride_elements];
        s += val, sq += (nk_u64_t)(val * val);
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i16_neon(                         //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_i64_t left_sum, right_sum;
        nk_u64_t left_sumsq, right_sumsq;
        nk_reduce_moments_i16_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_i16_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_sum, &right_sumsq);
        nk_i64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_i16_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_i16_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i16_neon_contiguous_( //
    nk_i16_t const *data, nk_size_t count,              //
    nk_i16_t *min_value, nk_size_t *min_index,          //
    nk_i16_t *max_value, nk_size_t *max_index) {
    int16x8_t min_i16x8 = vld1q_s16(data), max_i16x8 = min_i16x8;
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(1), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        int16x8_t data_i16x8 = vld1q_s16(data + idx);
        uint16x8_t less_u16x8 = vcltq_s16(data_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(data_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, data_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, data_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, remaining);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u16x8 = vdupq_n_u16(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u16s[i] = 0xFFFF;
        int16x8_t data_for_min_i16x8 = vbslq_s16(valid_mask_vec.u16x8, tail_vec.i16x8, vdupq_n_s16(NK_I16_MAX));
        int16x8_t data_for_max_i16x8 = vbslq_s16(valid_mask_vec.u16x8, tail_vec.i16x8, vdupq_n_s16(NK_I16_MIN));
        uint16x8_t less_u16x8 = vcltq_s16(data_for_min_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(data_for_max_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, data_for_min_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, data_for_max_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }
    nk_i16_t minimum_scalar = vminvq_s16(min_i16x8), maximum_scalar = vmaxvq_s16(max_i16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(minimum_scalar));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(maximum_scalar));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i16x8 = min_i16x8;
    maximum_values_vec.i16x8 = max_i16x8;
    minimum_iteration_indices_vec.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_vec.u16x8 = max_iter_u16x8;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 8; ++i)
        if (minimum_values_vec.i16s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u16s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 8; ++i)
        if (maximum_values_vec.i16s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u16s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_i16_neon_strided_(                  //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i16_t *min_value, nk_size_t *min_index,                        //
    nk_i16_t *max_value, nk_size_t *max_index) {
    int16x8_t min_i16x8 = vdupq_n_s16(NK_I16_MAX), max_i16x8 = vdupq_n_s16(NK_I16_MIN);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x2_t loaded_i16x8x2 = vld2q_s16(data + idx * 2);
            int16x8_t data_i16x8 = loaded_i16x8x2.val[0];
            uint16x8_t less_u16x8 = vcltq_s16(data_i16x8, min_i16x8);
            uint16x8_t greater_u16x8 = vcgtq_s16(data_i16x8, max_i16x8);
            min_i16x8 = vbslq_s16(less_u16x8, data_i16x8, min_i16x8);
            max_i16x8 = vbslq_s16(greater_u16x8, data_i16x8, max_i16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x3_t loaded_i16x8x3 = vld3q_s16(data + idx * 3);
            int16x8_t data_i16x8 = loaded_i16x8x3.val[0];
            uint16x8_t less_u16x8 = vcltq_s16(data_i16x8, min_i16x8);
            uint16x8_t greater_u16x8 = vcgtq_s16(data_i16x8, max_i16x8);
            min_i16x8 = vbslq_s16(less_u16x8, data_i16x8, min_i16x8);
            max_i16x8 = vbslq_s16(greater_u16x8, data_i16x8, max_i16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            int16x8x4_t loaded_i16x8x4 = vld4q_s16(data + idx * 4);
            int16x8_t data_i16x8 = loaded_i16x8x4.val[0];
            uint16x8_t less_u16x8 = vcltq_s16(data_i16x8, min_i16x8);
            uint16x8_t greater_u16x8 = vcgtq_s16(data_i16x8, max_i16x8);
            min_i16x8 = vbslq_s16(less_u16x8, data_i16x8, min_i16x8);
            max_i16x8 = vbslq_s16(greater_u16x8, data_i16x8, max_i16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    nk_i16_t minimum_scalar = vminvq_s16(min_i16x8), maximum_scalar = vmaxvq_s16(max_i16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(minimum_scalar));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(maximum_scalar));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i16x8 = min_i16x8;
    maximum_values_vec.i16x8 = max_i16x8;
    minimum_iteration_indices_vec.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_vec.u16x8 = max_iter_u16x8;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 8; ++i)
        if (minimum_values_vec.i16s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u16s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 8; ++i)
        if (maximum_values_vec.i16s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u16s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_i16_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i16_neon(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index,                     //
    nk_i16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);
    if (count == 0)
        *min_value = NK_I16_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I16_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_i16_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_i16_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                  &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_i16_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                  &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_i16_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u16_neon_contiguous_( //
    nk_u16_t const *data, nk_size_t count,               //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t d = vld1q_u16(data + idx);
        uint32x4_t sum32 = vpaddlq_u16(d);
        sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(sum32));
        uint32x4_t sq_lo = vmull_u16(vget_low_u16(d), vget_low_u16(d));
        uint32x4_t sq_hi = vmull_high_u16(d, d);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi));
    }
    nk_u64_t s = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_u16_neon_strided_(                 //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16(data + idx * 2);
            uint16x8_t data_u16x8 = loaded_u16x8x2.val[0];
            uint32x4_t widened_sum_u32x4 = vpaddlq_u16(data_u16x8);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(widened_sum_u32x4));
            uint32x4_t sq_lo_u32x4 = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
            uint32x4_t sq_hi_u32x4 = vmull_high_u16(data_u16x8, data_u16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo_u32x4));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi_u32x4));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16(data + idx * 3);
            uint16x8_t data_u16x8 = loaded_u16x8x3.val[0];
            uint32x4_t widened_sum_u32x4 = vpaddlq_u16(data_u16x8);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(widened_sum_u32x4));
            uint32x4_t sq_lo_u32x4 = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
            uint32x4_t sq_hi_u32x4 = vmull_high_u16(data_u16x8, data_u16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo_u32x4));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi_u32x4));
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16(data + idx * 4);
            uint16x8_t data_u16x8 = loaded_u16x8x4.val[0];
            uint32x4_t widened_sum_u32x4 = vpaddlq_u16(data_u16x8);
            sum_u64x2 = vaddq_u64(sum_u64x2, vpaddlq_u32(widened_sum_u32x4));
            uint32x4_t sq_lo_u32x4 = vmull_u16(vget_low_u16(data_u16x8), vget_low_u16(data_u16x8));
            uint32x4_t sq_hi_u32x4 = vmull_high_u16(data_u16x8, data_u16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_lo_u32x4));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(sq_hi_u32x4));
        }
    }

    nk_u64_t s = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sq = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx * stride_elements];
        s += val;
        sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u16_neon(                         //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u16_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u16_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_sum, &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u16_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_u16_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u16_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u16_neon_contiguous_( //
    nk_u16_t const *data, nk_size_t count,              //
    nk_u16_t *min_value, nk_size_t *min_index,          //
    nk_u16_t *max_value, nk_size_t *max_index) {
    uint16x8_t min_u16x8 = vld1q_u16(data), max_u16x8 = min_u16x8;
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(1), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t data_u16x8 = vld1q_u16(data + idx);
        uint16x8_t less_u16x8 = vcltq_u16(data_u16x8, min_u16x8);
        uint16x8_t greater_u16x8 = vcgtq_u16(data_u16x8, max_u16x8);
        min_u16x8 = vbslq_u16(less_u16x8, data_u16x8, min_u16x8);
        max_u16x8 = vbslq_u16(greater_u16x8, data_u16x8, max_u16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, remaining);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u16x8 = vdupq_n_u16(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u16s[i] = 0xFFFF;
        uint16x8_t data_for_min_u16x8 = vbslq_u16(valid_mask_vec.u16x8, tail_vec.u16x8, vdupq_n_u16(NK_U16_MAX));
        uint16x8_t data_for_max_u16x8 = vbslq_u16(valid_mask_vec.u16x8, tail_vec.u16x8, vdupq_n_u16(0));
        uint16x8_t less_u16x8 = vcltq_u16(data_for_min_u16x8, min_u16x8);
        uint16x8_t greater_u16x8 = vcgtq_u16(data_for_max_u16x8, max_u16x8);
        min_u16x8 = vbslq_u16(less_u16x8, data_for_min_u16x8, min_u16x8);
        max_u16x8 = vbslq_u16(greater_u16x8, data_for_max_u16x8, max_u16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }
    nk_u16_t minimum_scalar = vminvq_u16(min_u16x8), maximum_scalar = vmaxvq_u16(max_u16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_u16(min_u16x8, vdupq_n_u16(minimum_scalar));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_u16(max_u16x8, vdupq_n_u16(maximum_scalar));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u16x8 = min_u16x8;
    maximum_values_vec.u16x8 = max_u16x8;
    minimum_iteration_indices_vec.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_vec.u16x8 = max_iter_u16x8;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 8; ++i)
        if (minimum_values_vec.u16s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u16s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 8; ++i)
        if (maximum_values_vec.u16s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u16s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_u16_neon_strided_(                  //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u16_t *min_value, nk_size_t *min_index,                        //
    nk_u16_t *max_value, nk_size_t *max_index) {
    uint16x8_t min_u16x8 = vdupq_n_u16(NK_U16_MAX), max_u16x8 = vdupq_n_u16(0);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((nk_u16_t const *)data + idx * 2);
            uint16x8_t data_u16x8 = loaded_u16x8x2.val[0];
            uint16x8_t less_u16x8 = vcltq_u16(data_u16x8, min_u16x8);
            uint16x8_t greater_u16x8 = vcgtq_u16(data_u16x8, max_u16x8);
            min_u16x8 = vbslq_u16(less_u16x8, data_u16x8, min_u16x8);
            max_u16x8 = vbslq_u16(greater_u16x8, data_u16x8, max_u16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((nk_u16_t const *)data + idx * 3);
            uint16x8_t data_u16x8 = loaded_u16x8x3.val[0];
            uint16x8_t less_u16x8 = vcltq_u16(data_u16x8, min_u16x8);
            uint16x8_t greater_u16x8 = vcgtq_u16(data_u16x8, max_u16x8);
            min_u16x8 = vbslq_u16(less_u16x8, data_u16x8, min_u16x8);
            max_u16x8 = vbslq_u16(greater_u16x8, data_u16x8, max_u16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((nk_u16_t const *)data + idx * 4);
            uint16x8_t data_u16x8 = loaded_u16x8x4.val[0];
            uint16x8_t less_u16x8 = vcltq_u16(data_u16x8, min_u16x8);
            uint16x8_t greater_u16x8 = vcgtq_u16(data_u16x8, max_u16x8);
            min_u16x8 = vbslq_u16(less_u16x8, data_u16x8, min_u16x8);
            max_u16x8 = vbslq_u16(greater_u16x8, data_u16x8, max_u16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    nk_u16_t minimum_scalar = vminvq_u16(min_u16x8), maximum_scalar = vmaxvq_u16(max_u16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_u16(min_u16x8, vdupq_n_u16(minimum_scalar));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_u16(max_u16x8, vdupq_n_u16(maximum_scalar));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u16x8 = min_u16x8;
    maximum_values_vec.u16x8 = max_u16x8;
    minimum_iteration_indices_vec.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_vec.u16x8 = max_iter_u16x8;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 8; ++i)
        if (minimum_values_vec.u16s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u16s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 8; ++i)
        if (maximum_values_vec.u16s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u16s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u16_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u16_neon(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index,                     //
    nk_u16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);
    if (count == 0) *min_value = NK_U16_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_u16_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_u16_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                  &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_u16_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                  &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_u16_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i32_neon_contiguous_( //
    nk_i32_t const *data, nk_size_t count,               //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    // 128-bit accumulation: lower (u64) + upper (i64) per lane
    uint64x2_t sum_lower_u64x2 = vdupq_n_u64(0);
    int64x2_t sum_upper_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    int sumsq_overflow = 0;
    // XOR sign-bit trick for unsigned u64 compare on NEON
    int64x2_t sign_bit_i64x2 = vdupq_n_s64((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        int32x4_t d = vld1q_s32(data + idx);
        // Sum: widen i32->i64 and accumulate with carry detection
        int64x2_t lo = vmovl_s32(vget_low_s32(d));
        uint64x2_t before = sum_lower_u64x2;
        sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo));
        int64x2_t result_biased = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
        int64x2_t before_biased = veorq_s64(vreinterpretq_s64_u64(before), sign_bit_i64x2);
        uint64x2_t carry = vcgtq_s64(before_biased, result_biased);
        sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry));
        sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo, 63));

        int64x2_t hi = vmovl_high_s32(d);
        before = sum_lower_u64x2;
        sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi));
        result_biased = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
        before_biased = veorq_s64(vreinterpretq_s64_u64(before), sign_bit_i64x2);
        carry = vcgtq_s64(before_biased, result_biased);
        sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry));
        sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi, 63));

        // Sumsq: widening multiply i32*i32 -> i64 (always non-negative for squares)
        int64x2_t sq_lo = vmull_s32(vget_low_s32(d), vget_low_s32(d));
        int64x2_t sq_hi = vmull_high_s32(d, d);
        uint64x2_t sq_before = sumsq_u64x2;
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(sq_lo));
        result_biased = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
        before_biased = veorq_s64(vreinterpretq_s64_u64(sq_before), sign_bit_i64x2);
        sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased, result_biased), 0) |
                           vgetq_lane_s64(vcgtq_s64(before_biased, result_biased), 1));
        sq_before = sumsq_u64x2;
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(sq_hi));
        result_biased = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
        before_biased = veorq_s64(vreinterpretq_s64_u64(sq_before), sign_bit_i64x2);
        sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased, result_biased), 0) |
                           vgetq_lane_s64(vcgtq_s64(before_biased, result_biased), 1));
    }
    // Sumsq horizontal saturating reduction
    nk_u64_t sq;
    if (sumsq_overflow) sq = NK_U64_MAX;
    else sq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    // Sum: horizontal 128-bit reduction (2 lanes -> scalar)
    nk_b128_vec_t lower_vec, upper_vec;
    lower_vec.u64x2 = sum_lower_u64x2;
    upper_vec.i64x2 = sum_upper_i64x2;
    nk_u64_t s_lower = 0;
    nk_i64_t s_upper = 0;
    for (int i = 0; i < 2; i++) {
        nk_u64_t before = s_lower;
        s_lower += lower_vec.u64s[i];
        if (s_lower < before) s_upper++;
        s_upper += upper_vec.i64s[i];
    }
    // Scalar tail
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t)data[idx];
        nk_u64_t before = s_lower;
        s_lower += (nk_u64_t)val;
        if (s_lower < before) s_upper++;
        s_upper += (val >> 63);
        nk_i64_t product;
        nk_i64_smul_(&val, &val, &product);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        nk_u64_sadd_(&sq, &unsigned_product, &sq);
    }
    // Clamp 128-bit sum to i64 range
    nk_i64_t s_lower_signed = (nk_i64_t)s_lower;
    if (s_upper == (s_lower_signed >> 63)) *sum = s_lower_signed;
    else if (s_upper >= 0) *sum = NK_I64_MAX;
    else *sum = NK_I64_MIN;
    *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_i32_neon_strided_(                 //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_lower_u64x2 = vdupq_n_u64(0);
    int64x2_t sum_upper_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    int sumsq_overflow = 0;
    int64x2_t sign_bit_i64x2 = vdupq_n_s64((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x2_t loaded_i32x4x2 = vld2q_s32(data + idx * 2);
            int32x4_t data_i32x4 = loaded_i32x4x2.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            uint64x2_t before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo_i64x2));
            int64x2_t result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            int64x2_t before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            uint64x2_t carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo_i64x2, 63));
            int64x2_t hi_i64x2 = vmovl_high_s32(data_i32x4);
            before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi_i64x2, 63));
            int64x2_t squares_lo_i64x2 = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
            int64x2_t squares_hi_i64x2 = vmull_high_s32(data_i32x4, data_i32x4);
            uint64x2_t sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_lo_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 0) |
                               vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 1));
            sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 0) |
                               vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 1));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x3_t loaded_i32x4x3 = vld3q_s32(data + idx * 3);
            int32x4_t data_i32x4 = loaded_i32x4x3.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            uint64x2_t before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo_i64x2));
            int64x2_t result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            int64x2_t before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            uint64x2_t carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo_i64x2, 63));
            int64x2_t hi_i64x2 = vmovl_high_s32(data_i32x4);
            before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi_i64x2, 63));
            int64x2_t squares_lo_i64x2 = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
            int64x2_t squares_hi_i64x2 = vmull_high_s32(data_i32x4, data_i32x4);
            uint64x2_t sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_lo_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 0) |
                               vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 1));
            sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 0) |
                               vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 1));
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x4_t loaded_i32x4x4 = vld4q_s32(data + idx * 4);
            int32x4_t data_i32x4 = loaded_i32x4x4.val[0];
            int64x2_t lo_i64x2 = vmovl_s32(vget_low_s32(data_i32x4));
            uint64x2_t before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(lo_i64x2));
            int64x2_t result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            int64x2_t before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            uint64x2_t carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(lo_i64x2, 63));
            int64x2_t hi_i64x2 = vmovl_high_s32(data_i32x4);
            before_u64x2 = sum_lower_u64x2;
            sum_lower_u64x2 = vaddq_u64(sum_lower_u64x2, vreinterpretq_u64_s64(hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sum_lower_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(before_u64x2), sign_bit_i64x2);
            carry_u64x2 = vcgtq_s64(before_biased_i64x2, result_biased_i64x2);
            sum_upper_i64x2 = vsubq_s64(sum_upper_i64x2, vreinterpretq_s64_u64(carry_u64x2));
            sum_upper_i64x2 = vaddq_s64(sum_upper_i64x2, vshrq_n_s64(hi_i64x2, 63));
            int64x2_t squares_lo_i64x2 = vmull_s32(vget_low_s32(data_i32x4), vget_low_s32(data_i32x4));
            int64x2_t squares_hi_i64x2 = vmull_high_s32(data_i32x4, data_i32x4);
            uint64x2_t sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_lo_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 0) |
                               vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 1));
            sq_before_u64x2 = sumsq_u64x2;
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vreinterpretq_u64_s64(squares_hi_i64x2));
            result_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
            before_biased_i64x2 = veorq_s64(vreinterpretq_s64_u64(sq_before_u64x2), sign_bit_i64x2);
            sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 0) |
                               vgetq_lane_s64(vcgtq_s64(before_biased_i64x2, result_biased_i64x2), 1));
        }
    }
    nk_u64_t sq;
    if (sumsq_overflow) sq = NK_U64_MAX;
    else sq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    nk_b128_vec_t lower_vec, upper_vec;
    lower_vec.u64x2 = sum_lower_u64x2;
    upper_vec.i64x2 = sum_upper_i64x2;
    nk_u64_t s_lower = 0;
    nk_i64_t s_upper = 0;
    for (int i = 0; i < 2; i++) {
        nk_u64_t before = s_lower;
        s_lower += lower_vec.u64s[i];
        if (s_lower < before) s_upper++;
        s_upper += upper_vec.i64s[i];
    }
    for (; idx < count; ++idx) {
        nk_i64_t val = (nk_i64_t) * (data + idx * stride_elements);
        nk_u64_t before = s_lower;
        s_lower += (nk_u64_t)val;
        if (s_lower < before) s_upper++;
        s_upper += (val >> 63);
        nk_i64_t product;
        nk_i64_smul_(&val, &val, &product);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        nk_u64_sadd_(&sq, &unsigned_product, &sq);
    }
    nk_i64_t s_lower_signed = (nk_i64_t)s_lower;
    if (s_upper == (s_lower_signed >> 63)) *sum = s_lower_signed;
    else if (s_upper >= 0) *sum = NK_I64_MAX;
    else *sum = NK_I64_MIN;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i32_neon(                         //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i32_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_i32_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_i32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i32_neon_contiguous_( //
    nk_i32_t const *data, nk_size_t count,              //
    nk_i32_t *min_value, nk_size_t *min_index,          //
    nk_i32_t *max_value, nk_size_t *max_index) {
    int32x4_t min_i32x4 = vld1q_s32(data), max_i32x4 = min_i32x4;
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(1), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 4;
    for (; idx + 4 <= count; idx += 4) {
        int32x4_t data_i32x4 = vld1q_s32(data + idx);
        uint32x4_t less_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
        uint32x4_t greater_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
        min_i32x4 = vbslq_s32(less_u32x4, data_i32x4, min_i32x4);
        max_i32x4 = vbslq_s32(greater_u32x4, data_i32x4, max_i32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
        iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b32x4_serial_(data + idx, &tail_vec, remaining);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u32x4 = vdupq_n_u32(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u32s[i] = 0xFFFFFFFF;
        int32x4_t data_min_i32x4 = vbslq_s32(valid_mask_vec.u32x4, tail_vec.i32x4, vdupq_n_s32(NK_I32_MAX));
        int32x4_t data_max_i32x4 = vbslq_s32(valid_mask_vec.u32x4, tail_vec.i32x4, vdupq_n_s32(NK_I32_MIN));
        uint32x4_t less_u32x4 = vcltq_s32(data_min_i32x4, min_i32x4);
        uint32x4_t greater_u32x4 = vcgtq_s32(data_max_i32x4, max_i32x4);
        min_i32x4 = vbslq_s32(less_u32x4, data_min_i32x4, min_i32x4);
        max_i32x4 = vbslq_s32(greater_u32x4, data_max_i32x4, max_i32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    }
    nk_i32_t minimum_scalar = vminvq_s32(min_i32x4), maximum_scalar = vmaxvq_s32(max_i32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_s32(min_i32x4, vdupq_n_s32(minimum_scalar));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_s32(max_i32x4, vdupq_n_s32(maximum_scalar));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i32x4 = min_i32x4;
    maximum_values_vec.i32x4 = max_i32x4;
    minimum_iteration_indices_vec.u32x4 = min_iter_u32x4;
    maximum_iteration_indices_vec.u32x4 = max_iter_u32x4;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 4; ++i)
        if (minimum_values_vec.i32s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u32s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 4; ++i)
        if (maximum_values_vec.i32s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u32s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_i32_neon_strided_(                  //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_i32_t *min_value, nk_size_t *min_index,                        //
    nk_i32_t *max_value, nk_size_t *max_index) {
    int32x4_t min_i32x4 = vdupq_n_s32(NK_I32_MAX), max_i32x4 = vdupq_n_s32(NK_I32_MIN);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x2_t loaded_i32x4x2 = vld2q_s32(data + idx * 2);
            int32x4_t data_i32x4 = loaded_i32x4x2.val[0];
            uint32x4_t less_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
            uint32x4_t greater_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
            min_i32x4 = vbslq_s32(less_u32x4, data_i32x4, min_i32x4);
            max_i32x4 = vbslq_s32(greater_u32x4, data_i32x4, max_i32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x3_t loaded_i32x4x3 = vld3q_s32(data + idx * 3);
            int32x4_t data_i32x4 = loaded_i32x4x3.val[0];
            uint32x4_t less_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
            uint32x4_t greater_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
            min_i32x4 = vbslq_s32(less_u32x4, data_i32x4, min_i32x4);
            max_i32x4 = vbslq_s32(greater_u32x4, data_i32x4, max_i32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            int32x4x4_t loaded_i32x4x4 = vld4q_s32(data + idx * 4);
            int32x4_t data_i32x4 = loaded_i32x4x4.val[0];
            uint32x4_t less_u32x4 = vcltq_s32(data_i32x4, min_i32x4);
            uint32x4_t greater_u32x4 = vcgtq_s32(data_i32x4, max_i32x4);
            min_i32x4 = vbslq_s32(less_u32x4, data_i32x4, min_i32x4);
            max_i32x4 = vbslq_s32(greater_u32x4, data_i32x4, max_i32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    nk_i32_t minimum_scalar = vminvq_s32(min_i32x4), maximum_scalar = vmaxvq_s32(max_i32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_s32(min_i32x4, vdupq_n_s32(minimum_scalar));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_s32(max_i32x4, vdupq_n_s32(maximum_scalar));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i32x4 = min_i32x4;
    maximum_values_vec.i32x4 = max_i32x4;
    minimum_iteration_indices_vec.u32x4 = min_iter_u32x4;
    maximum_iteration_indices_vec.u32x4 = max_iter_u32x4;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 4; ++i)
        if (minimum_values_vec.i32s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u32s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 4; ++i)
        if (maximum_values_vec.i32s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u32s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_i32_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i32_neon(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index,                     //
    nk_i32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);
    if (count == 0)
        *min_value = NK_I32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I32_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_partition_count = count / 2;
        nk_i32_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_i32_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                  &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_i32_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                  &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_i32_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u32_neon_contiguous_( //
    nk_u32_t const *data, nk_size_t count,               //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t d = vld1q_u32(data + idx);
        // Widen u32 -> u64 and accumulate sum
        sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(d)));
        sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(d));
        // Sumsq: widening multiply u32*u32 -> u64, saturating add
        uint64x2_t sq_lo = vmull_u32(vget_low_u32(d), vget_low_u32(d));
        uint64x2_t sq_hi = vmull_high_u32(d, d);
        sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, sq_lo);
        sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, sq_hi);
    }
    nk_u64_t s = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t)data[idx];
        s += val;
        nk_u64_t product = val * val;
        nk_u64_sadd_(&sq, &product, &sq);
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_u32_neon_strided_(                 //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x2_t loaded_u32x4x2 = vld2q_u32(data + idx * 2);
            uint32x4_t data_u32x4 = loaded_u32x4x2.val[0];
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
            uint64x2_t squares_lo_u64x2 = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
            uint64x2_t squares_hi_u64x2 = vmull_high_u32(data_u32x4, data_u32x4);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_lo_u64x2);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_hi_u64x2);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x3_t loaded_u32x4x3 = vld3q_u32(data + idx * 3);
            uint32x4_t data_u32x4 = loaded_u32x4x3.val[0];
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
            uint64x2_t squares_lo_u64x2 = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
            uint64x2_t squares_hi_u64x2 = vmull_high_u32(data_u32x4, data_u32x4);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_lo_u64x2);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_hi_u64x2);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x4_t loaded_u32x4x4 = vld4q_u32(data + idx * 4);
            uint32x4_t data_u32x4 = loaded_u32x4x4.val[0];
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_u32(vget_low_u32(data_u32x4)));
            sum_u64x2 = vaddq_u64(sum_u64x2, vmovl_high_u32(data_u32x4));
            uint64x2_t squares_lo_u64x2 = vmull_u32(vget_low_u32(data_u32x4), vget_low_u32(data_u32x4));
            uint64x2_t squares_hi_u64x2 = vmull_high_u32(data_u32x4, data_u32x4);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_lo_u64x2);
            sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, squares_hi_u64x2);
        }
    }
    nk_u64_t s = vgetq_lane_u64(sum_u64x2, 0) + vgetq_lane_u64(sum_u64x2, 1);
    nk_u64_t sq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = (nk_u64_t) * (data + idx * stride_elements);
        s += val;
        nk_u64_t product = val * val;
        nk_u64_sadd_(&sq, &product, &sq);
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u32_neon(                         //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 4) {
        nk_size_t left_partition_count = count / 2;
        nk_u64_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_u32_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_u32_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_sum, &right_sumsq);
        nk_u64_sadd_(&left_sum, &right_sum, sum);
        nk_u64_sadd_(&left_sumsq, &right_sumsq, sumsq);
    }
    else if (stride_elements == 1) nk_reduce_moments_u32_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_u32_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_u32_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u32_neon_contiguous_( //
    nk_u32_t const *data, nk_size_t count,              //
    nk_u32_t *min_value, nk_size_t *min_index,          //
    nk_u32_t *max_value, nk_size_t *max_index) {
    uint32x4_t min_u32x4 = vld1q_u32(data), max_u32x4 = min_u32x4;
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(1), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 4;
    for (; idx + 4 <= count; idx += 4) {
        uint32x4_t data_u32x4 = vld1q_u32(data + idx);
        uint32x4_t less_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
        uint32x4_t greater_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
        min_u32x4 = vbslq_u32(less_u32x4, data_u32x4, min_u32x4);
        max_u32x4 = vbslq_u32(greater_u32x4, data_u32x4, max_u32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
        iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b32x4_serial_(data + idx, &tail_vec, remaining);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u32x4 = vdupq_n_u32(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u32s[i] = 0xFFFFFFFF;
        uint32x4_t data_min_u32x4 = vbslq_u32(valid_mask_vec.u32x4, tail_vec.u32x4, vdupq_n_u32(NK_U32_MAX));
        uint32x4_t data_max_u32x4 = vbslq_u32(valid_mask_vec.u32x4, tail_vec.u32x4, vdupq_n_u32(0));
        uint32x4_t less_u32x4 = vcltq_u32(data_min_u32x4, min_u32x4);
        uint32x4_t greater_u32x4 = vcgtq_u32(data_max_u32x4, max_u32x4);
        min_u32x4 = vbslq_u32(less_u32x4, data_min_u32x4, min_u32x4);
        max_u32x4 = vbslq_u32(greater_u32x4, data_max_u32x4, max_u32x4);
        min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
        max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
    }
    nk_u32_t minimum_scalar = vminvq_u32(min_u32x4), maximum_scalar = vmaxvq_u32(max_u32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_u32(min_u32x4, vdupq_n_u32(minimum_scalar));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_u32(max_u32x4, vdupq_n_u32(maximum_scalar));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u32x4 = min_u32x4;
    maximum_values_vec.u32x4 = max_u32x4;
    minimum_iteration_indices_vec.u32x4 = min_iter_u32x4;
    maximum_iteration_indices_vec.u32x4 = max_iter_u32x4;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 4; ++i)
        if (minimum_values_vec.u32s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u32s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 4; ++i)
        if (maximum_values_vec.u32s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u32s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)i;
            break;
        }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_u32_neon_strided_(                  //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_u32_t *min_value, nk_size_t *min_index,                        //
    nk_u32_t *max_value, nk_size_t *max_index) {
    uint32x4_t min_u32x4 = vdupq_n_u32(NK_U32_MAX), max_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_iter_u32x4 = vdupq_n_u32(0), max_iter_u32x4 = vdupq_n_u32(0);
    uint32x4_t iter_u32x4 = vdupq_n_u32(0), one_u32x4 = vdupq_n_u32(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x2_t loaded_u32x4x2 = vld2q_u32(data + idx * 2);
            uint32x4_t data_u32x4 = loaded_u32x4x2.val[0];
            uint32x4_t less_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
            uint32x4_t greater_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
            min_u32x4 = vbslq_u32(less_u32x4, data_u32x4, min_u32x4);
            max_u32x4 = vbslq_u32(greater_u32x4, data_u32x4, max_u32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x3_t loaded_u32x4x3 = vld3q_u32(data + idx * 3);
            uint32x4_t data_u32x4 = loaded_u32x4x3.val[0];
            uint32x4_t less_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
            uint32x4_t greater_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
            min_u32x4 = vbslq_u32(less_u32x4, data_u32x4, min_u32x4);
            max_u32x4 = vbslq_u32(greater_u32x4, data_u32x4, max_u32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    else {
        for (; idx + 4 <= count; idx += 4) {
            uint32x4x4_t loaded_u32x4x4 = vld4q_u32(data + idx * 4);
            uint32x4_t data_u32x4 = loaded_u32x4x4.val[0];
            uint32x4_t less_u32x4 = vcltq_u32(data_u32x4, min_u32x4);
            uint32x4_t greater_u32x4 = vcgtq_u32(data_u32x4, max_u32x4);
            min_u32x4 = vbslq_u32(less_u32x4, data_u32x4, min_u32x4);
            max_u32x4 = vbslq_u32(greater_u32x4, data_u32x4, max_u32x4);
            min_iter_u32x4 = vbslq_u32(less_u32x4, iter_u32x4, min_iter_u32x4);
            max_iter_u32x4 = vbslq_u32(greater_u32x4, iter_u32x4, max_iter_u32x4);
            iter_u32x4 = vaddq_u32(iter_u32x4, one_u32x4);
        }
    }
    nk_u32_t minimum_scalar = vminvq_u32(min_u32x4), maximum_scalar = vmaxvq_u32(max_u32x4);
    uint32x4_t min_value_match_u32x4 = vceqq_u32(min_u32x4, vdupq_n_u32(minimum_scalar));
    uint32x4_t masked_min_iter_u32x4 = vbslq_u32(min_value_match_u32x4, min_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_min_cycle = vminvq_u32(masked_min_iter_u32x4);
    uint32x4_t max_value_match_u32x4 = vceqq_u32(max_u32x4, vdupq_n_u32(maximum_scalar));
    uint32x4_t masked_max_iter_u32x4 = vbslq_u32(max_value_match_u32x4, max_iter_u32x4, vdupq_n_u32(NK_U32_MAX));
    nk_u32_t earliest_max_cycle = vminvq_u32(masked_max_iter_u32x4);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u32x4 = min_u32x4;
    maximum_values_vec.u32x4 = max_u32x4;
    minimum_iteration_indices_vec.u32x4 = min_iter_u32x4;
    maximum_iteration_indices_vec.u32x4 = max_iter_u32x4;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 4; ++i)
        if (minimum_values_vec.u32s[i] == minimum_scalar &&
            minimum_iteration_indices_vec.u32s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 4; ++i)
        if (maximum_values_vec.u32s[i] == maximum_scalar &&
            maximum_iteration_indices_vec.u32s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 4 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u32_t val = *(data + idx * stride_elements);
        if (val < minimum_scalar) minimum_scalar = val, min_idx = idx;
        if (val > maximum_scalar) maximum_scalar = val, max_idx = idx;
    }
    *min_value = minimum_scalar, *min_index = min_idx;
    *max_value = maximum_scalar, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u32_neon(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index,                     //
    nk_u32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);
    if (count == 0) *min_value = NK_U32_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)NK_U32_MAX * 4) {
        nk_size_t left_partition_count = count / 2;
        nk_u32_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_u32_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                  &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_u32_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                  stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                  &right_maximum_index);
        if (right_minimum_value < left_minimum_value)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (right_maximum_value > left_maximum_value)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_u32_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_i64_neon_contiguous_( //
    nk_i64_t const *data, nk_size_t count,               //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_u64_t sum_lower = 0;
    nk_i64_t sum_upper = 0;
    // NEON can still load/extract i64 vectors for sumsq via scalar nk_i64_smul_
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    int sumsq_overflow = 0;
    int64x2_t sign_bit_i64x2 = vdupq_n_s64((nk_i64_t)0x8000000000000000ULL);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        int64x2_t d = vld1q_s64(data + idx);
        // Sumsq via helper (scalar per-lane multiply)
        uint64x2_t sq = nk_i64_smul_sq_i64x2_neon_(d);
        uint64x2_t sq_before = sumsq_u64x2;
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, sq);
        int64x2_t result_biased = veorq_s64(vreinterpretq_s64_u64(sumsq_u64x2), sign_bit_i64x2);
        int64x2_t before_biased = veorq_s64(vreinterpretq_s64_u64(sq_before), sign_bit_i64x2);
        sumsq_overflow |= (vgetq_lane_s64(vcgtq_s64(before_biased, result_biased), 0) |
                           vgetq_lane_s64(vcgtq_s64(before_biased, result_biased), 1));
        // 128-bit sum via scalar carry detection
        nk_b128_vec_t vec;
        vec.i64x2 = d;
        for (int i = 0; i < 2; i++) {
            nk_i64_t val = vec.i64s[i];
            nk_u64_t before = sum_lower;
            sum_lower += (nk_u64_t)val;
            if (sum_lower < before) sum_upper++;
            sum_upper += (val >> 63);
        }
    }
    nk_u64_t sq;
    if (sumsq_overflow) sq = NK_U64_MAX;
    else sq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx];
        nk_i64_t product;
        nk_i64_smul_(&val, &val, &product);
        nk_u64_t unsigned_product = (nk_u64_t)product;
        nk_u64_sadd_(&sq, &unsigned_product, &sq);
        nk_u64_t before = sum_lower;
        sum_lower += (nk_u64_t)val;
        if (sum_lower < before) sum_upper++;
        sum_upper += (val >> 63);
    }
    nk_i64_t s_lower_signed = (nk_i64_t)sum_lower;
    if (sum_upper == (s_lower_signed >> 63)) *sum = s_lower_signed;
    else if (sum_upper >= 0) *sum = NK_I64_MAX;
    else *sum = NK_I64_MIN;
    *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_i64_neon(                         //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_i64_neon_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_i64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_i64_neon_contiguous_( //
    nk_i64_t const *data, nk_size_t count,              //
    nk_i64_t *min_value, nk_size_t *min_index,          //
    nk_i64_t *max_value, nk_size_t *max_index) {
    int64x2_t min_i64x2 = vld1q_s64(data), max_i64x2 = min_i64x2;
    uint64x2_t min_iter = vdupq_n_u64(0), max_iter = vdupq_n_u64(0);
    uint64x2_t iter = vdupq_n_u64(1), one = vdupq_n_u64(1);
    nk_size_t idx = 2;
    for (; idx + 2 <= count; idx += 2) {
        int64x2_t d = vld1q_s64(data + idx);
        uint64x2_t lt = vcltq_s64(d, min_i64x2), gt = vcgtq_s64(d, max_i64x2);
        min_i64x2 = vbslq_s64(lt, d, min_i64x2);
        max_i64x2 = vbslq_s64(gt, d, max_i64x2);
        min_iter = vbslq_u64(lt, iter, min_iter);
        max_iter = vbslq_u64(gt, iter, max_iter);
        iter = vaddq_u64(iter, one);
    }
    nk_b128_vec_t mv, xv, mi, xi;
    mv.i64x2 = min_i64x2;
    mi.u64x2 = min_iter;
    xv.i64x2 = max_i64x2;
    xi.u64x2 = max_iter;
    nk_i64_t min_s, max_s;
    nk_size_t min_idx, max_idx;
    if (mv.i64s[0] <= mv.i64s[1]) min_s = mv.i64s[0], min_idx = (nk_size_t)mi.u64s[0] * 2;
    else min_s = mv.i64s[1], min_idx = (nk_size_t)mi.u64s[1] * 2 + 1;
    if (xv.i64s[0] >= xv.i64s[1]) max_s = xv.i64s[0], max_idx = (nk_size_t)xi.u64s[0] * 2;
    else max_s = xv.i64s[1], max_idx = (nk_size_t)xi.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_i64_t val = data[idx];
        if (val < min_s) min_s = val, min_idx = idx;
        if (val > max_s) max_s = val, max_idx = idx;
    }
    *min_value = min_s, *min_index = min_idx;
    *max_value = max_s, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_i64_neon(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index,                     //
    nk_i64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);
    if (count == 0)
        *min_value = NK_I64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_I64_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_i64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_u64_neon_contiguous_( //
    nk_u64_t const *data, nk_size_t count,               //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    uint64x2_t sum_u64x2 = vdupq_n_u64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t d = vld1q_u64(data + idx);
        sum_u64x2 = vqaddq_u64(sum_u64x2, d);
        uint64x2_t sq = nk_u64_smul_sq_u64x2_neon_(d);
        sumsq_u64x2 = vqaddq_u64(sumsq_u64x2, sq);
    }
    nk_u64_t s = nk_reduce_sadd_u64x2_neon_(sum_u64x2);
    nk_u64_t sq = nk_reduce_sadd_u64x2_neon_(sumsq_u64x2);
    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx];
        nk_u64_sadd_(&s, &val, &s);
        nk_u64_t product;
        nk_u64_smul_(&val, &val, &product);
        nk_u64_sadd_(&sq, &product, &sq);
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_u64_neon(                         //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum, nk_u64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_u64_neon_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_u64_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_u64_neon_contiguous_( //
    nk_u64_t const *data, nk_size_t count,              //
    nk_u64_t *min_value, nk_size_t *min_index,          //
    nk_u64_t *max_value, nk_size_t *max_index) {
    uint64x2_t min_u64x2 = vld1q_u64(data), max_u64x2 = min_u64x2;
    uint64x2_t min_iter = vdupq_n_u64(0), max_iter = vdupq_n_u64(0);
    uint64x2_t iter = vdupq_n_u64(1), one = vdupq_n_u64(1);
    nk_size_t idx = 2;
    for (; idx + 2 <= count; idx += 2) {
        uint64x2_t d = vld1q_u64(data + idx);
        uint64x2_t lt = vcltq_u64(d, min_u64x2), gt = vcgtq_u64(d, max_u64x2);
        min_u64x2 = vbslq_u64(lt, d, min_u64x2);
        max_u64x2 = vbslq_u64(gt, d, max_u64x2);
        min_iter = vbslq_u64(lt, iter, min_iter);
        max_iter = vbslq_u64(gt, iter, max_iter);
        iter = vaddq_u64(iter, one);
    }
    nk_b128_vec_t mv, xv, mi, xi;
    mv.u64x2 = min_u64x2;
    mi.u64x2 = min_iter;
    xv.u64x2 = max_u64x2;
    xi.u64x2 = max_iter;
    nk_u64_t min_s, max_s;
    nk_size_t min_idx, max_idx;
    if (mv.u64s[0] <= mv.u64s[1]) min_s = mv.u64s[0], min_idx = (nk_size_t)mi.u64s[0] * 2;
    else min_s = mv.u64s[1], min_idx = (nk_size_t)mi.u64s[1] * 2 + 1;
    if (xv.u64s[0] >= xv.u64s[1]) max_s = xv.u64s[0], max_idx = (nk_size_t)xi.u64s[0] * 2;
    else max_s = xv.u64s[1], max_idx = (nk_size_t)xi.u64s[1] * 2 + 1;
    for (; idx < count; ++idx) {
        nk_u64_t val = data[idx];
        if (val < min_s) min_s = val, min_idx = idx;
        if (val > max_s) max_s = val, max_idx = idx;
    }
    *min_value = min_s, *min_index = min_idx;
    *max_value = max_s, *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_u64_neon(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index,                     //
    nk_u64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);
    if (count == 0) *min_value = NK_U64_MAX, *min_index = NK_SIZE_MAX, *max_value = 0, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_u64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

/** @brief Convert 16 raw FP6 (e2m3/e3m2) sign-magnitude bytes to unsigned-comparable bytes.
 *  FP6: sign bit 5, 5-bit magnitude. Positive maps to [0x20..0x3F], negative to [0x00..0x1F]. */
NK_INTERNAL uint8x16_t nk_fp6x16_to_comparable_neon_(uint8x16_t raw_u8x16) {
    uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x20);
    uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, sign_mask_u8x16);
    uint8x16_t positive_u8x16 = vorrq_u8(magnitude_u8x16, sign_mask_u8x16);
    uint8x16_t negative_u8x16 = vsubq_u8(vdupq_n_u8(0x1F), magnitude_u8x16);
    return vbslq_u8(is_negative_u8x16, negative_u8x16, positive_u8x16);
}

/** @brief Convert a single comparable byte back to raw FP6 sign-magnitude byte. */
NK_INTERNAL nk_u8_t nk_comparable_to_fp6_(nk_u8_t comparable) {
    if (comparable >= 0x20) return comparable ^ 0x20; // was positive
    else return (0x1F - comparable) | 0x20;           // was negative
}

/** @brief Convert 16 raw FP8 (e4m3/e5m2) sign-magnitude bytes to unsigned-comparable bytes. */
NK_INTERNAL uint8x16_t nk_fp8x16_to_comparable_neon_(uint8x16_t raw_u8x16) {
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, sign_mask_u8x16);
    uint8x16_t flip_positive_u8x16 = veorq_u8(raw_u8x16, sign_mask_u8x16);
    uint8x16_t flip_negative_u8x16 = vmvnq_u8(raw_u8x16);
    return vbslq_u8(is_negative_u8x16, flip_negative_u8x16, flip_positive_u8x16);
}

/** @brief Convert a single comparable byte back to raw FP8 sign-magnitude byte. */
NK_INTERNAL nk_u8_t nk_comparable_to_fp8_(nk_u8_t comparable) {
    if (comparable >= 0x80) return comparable ^ 0x80; // was positive
    else return ~comparable;                          // was negative
}

NK_INTERNAL void nk_reduce_moments_e2m3_neon_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,               //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    // VTBL LUT: maps 6-bit magnitude (0..31) to value×16 (unsigned), fits in u8
    uint8x16x2_t const lut_e2m3_x16 = {{
        {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
        {32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120},
    }};
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
        uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
        uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
        int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
        int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
        int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
        int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
        sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
        int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
        int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
    }
    nk_i64_t s_i64 = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq_u64 = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e2m3_to_f32(data[idx]);
        s_i64 += (nk_i64_t)(val * 16.0f), sq_u64 += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum = (nk_f32_t)s_i64 / 16.0f, *sumsq = (nk_f32_t)sq_u64 / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e2m3_neon_strided_(                 //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    uint8x16x2_t const lut_e2m3_x16 = {{
        {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
        {32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120},
    }};
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            uint8x16_t raw_u8x16 = loaded_u8x16x2.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            uint8x16_t raw_u8x16 = loaded_u8x16x3.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            uint8x16_t raw_u8x16 = loaded_u8x16x4.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t unsigned_u8x16 = vqtbl2q_u8(lut_e2m3_x16, magnitude_u8x16);
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            int8x16_t positive_i8x16 = vreinterpretq_s8_u8(unsigned_u8x16);
            int8x16_t negative_i8x16 = vnegq_s8(positive_i8x16);
            int8x16_t scaled_i8x16 = vbslq_s8(is_negative_u8x16, negative_i8x16, positive_i8x16);
            int16x8_t pairwise_i16x8 = vpaddlq_s8(scaled_i8x16);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(pairwise_i16x8)));
            int16x8_t squares_lo_i16x8 = vmull_s8(vget_low_s8(scaled_i8x16), vget_low_s8(scaled_i8x16));
            int16x8_t squares_hi_i16x8 = vmull_high_s8(scaled_i8x16, scaled_i8x16);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_lo_i16x8))));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_s16(squares_hi_i16x8))));
        }
    }
    nk_i64_t s_i64 = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq_u64 = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e2m3_to_f32(*(nk_e2m3_t const *)(data + idx * stride_elements));
        s_i64 += (nk_i64_t)(val * 16.0f), sq_u64 += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum = (nk_f32_t)s_i64 / 16.0f, *sumsq = (nk_f32_t)sq_u64 / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e2m3_neon(                         //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e2m3_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e2m3_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                    stride_bytes, &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e2m3_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_e2m3_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e2m3_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_neon_contiguous_( //
    nk_e2m3_t const *data, nk_size_t count,              //
    nk_e2m3_t *min_value, nk_size_t *min_index,          //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    uint8x16_t first_raw_u8x16 = vld1q_u8((nk_u8_t const *)data);
    uint8x16_t first_comparable_u8x16 = nk_fp6x16_to_comparable_neon_(first_raw_u8x16);
    uint8x16_t min_u8x16 = first_comparable_u8x16, max_u8x16 = first_comparable_u8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(tail_vec.u8x16);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u8x16 = vdupq_n_u8(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u8s[i] = 0xFF;
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    *min_value = nk_comparable_to_fp6_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp6_(max_comparable), *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e2m3_neon_strided_(                  //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_e2m3_t *min_value, nk_size_t *min_index,                        //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded_u8x16x2.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded_u8x16x3.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded_u8x16x4.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u8_t raw = *(nk_u8_t const *)(data + idx * stride_elements);
        nk_u8_t magnitude = raw & 0x1F;
        nk_u8_t comparable = (raw & 0x20) ? (0x1F - magnitude) : (magnitude | 0x20);
        if (comparable < min_comparable) min_comparable = comparable, min_idx = idx;
        if (comparable > max_comparable) max_comparable = comparable, max_idx = idx;
    }
    *min_value = nk_comparable_to_fp6_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp6_(max_comparable), *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e2m3_neon(                          //
    nk_e2m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value, nk_size_t *min_index,                     //
    nk_e2m3_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);
    if (count == 0)
        *min_value = NK_E2M3_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E2M3_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e2m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_e2m3_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_e2m3_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                   &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_e2m3_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                   &right_maximum_index);
        if (nk_e2m3_compare_(&right_minimum_value, &left_minimum_value) < 0)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (nk_e2m3_compare_(&right_maximum_value, &left_maximum_value) > 0)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e2m3_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e2m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_e3m2_neon_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,               //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    // VTBL LUT: maps 6-bit magnitude (0..31) to (value×16) low byte; max value×16 = 448 needs i16
    uint8x16x2_t const lut_e3m2_lo = {{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28},
        {32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 0, 64, 128, 192},
    }};
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
        uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
        uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
        uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
        // Interleave low+high bytes into i16 values (two halves of 8 each)
        uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
        uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
        // Sign-extend the per-byte negative mask to per-i16 lanes
        int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
        int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
        uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
        uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
        // Apply sign via conditional negate
        int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
        int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8), positive_lo_i16x8);
        int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
        int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8), positive_hi_i16x8);
        // Sum: i16→i32→i64 widening chain
        sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_lo_i16x8)));
        sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_hi_i16x8)));
        // Sumsq: vmull_s16→i32 (always positive as squares), widen to u64
        int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
        int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
        int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
        int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
        sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
    }
    nk_i64_t s_i64 = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq_u64 = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e3m2_to_f32(data[idx]);
        s_i64 += (nk_i64_t)(val * 16.0f), sq_u64 += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum = (nk_f32_t)s_i64 / 16.0f, *sumsq = (nk_f32_t)sq_u64 / 256.0f;
}

NK_INTERNAL void nk_reduce_moments_e3m2_neon_strided_(                 //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    uint8x16x2_t const lut_e3m2_lo = {{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28},
        {32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 0, 64, 128, 192},
    }};
    int64x2_t sum_i64x2 = vdupq_n_s64(0);
    uint64x2_t sumsq_u64x2 = vdupq_n_u64(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            uint8x16_t raw_u8x16 = loaded_u8x16x2.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
            uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
            uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
            int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
            int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
            uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
            uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
            int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
            int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8),
                                                  positive_lo_i16x8);
            int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
            int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8),
                                                  positive_hi_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_lo_i16x8)));
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_hi_i16x8)));
            int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
            int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
            int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
            int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            uint8x16_t raw_u8x16 = loaded_u8x16x3.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
            uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
            uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
            int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
            int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
            uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
            uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
            int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
            int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8),
                                                  positive_lo_i16x8);
            int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
            int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8),
                                                  positive_hi_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_lo_i16x8)));
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_hi_i16x8)));
            int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
            int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
            int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
            int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            uint8x16_t raw_u8x16 = loaded_u8x16x4.val[0];
            uint8x16_t magnitude_u8x16 = vandq_u8(raw_u8x16, vdupq_n_u8(0x1F));
            uint8x16_t low_byte_u8x16 = vqtbl2q_u8(lut_e3m2_lo, magnitude_u8x16);
            uint8x16_t high_byte_u8x16 = vandq_u8(vcgtq_u8(magnitude_u8x16, vdupq_n_u8(27)), vdupq_n_u8(1));
            uint8x16_t is_negative_u8x16 = vtstq_u8(raw_u8x16, vdupq_n_u8(0x20));
            uint16x8_t unsigned_lo_u16x8 = vreinterpretq_u16_u8(vzip1q_u8(low_byte_u8x16, high_byte_u8x16));
            uint16x8_t unsigned_hi_u16x8 = vreinterpretq_u16_u8(vzip2q_u8(low_byte_u8x16, high_byte_u8x16));
            int8x8_t is_negative_lo_i8x8 = vreinterpret_s8_u8(vget_low_u8(is_negative_u8x16));
            int8x8_t is_negative_hi_i8x8 = vreinterpret_s8_u8(vget_high_u8(is_negative_u8x16));
            uint16x8_t is_negative_lo_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_lo_i8x8));
            uint16x8_t is_negative_hi_u16x8 = vreinterpretq_u16_s16(vmovl_s8(is_negative_hi_i8x8));
            int16x8_t positive_lo_i16x8 = vreinterpretq_s16_u16(unsigned_lo_u16x8);
            int16x8_t scaled_lo_i16x8 = vbslq_s16(is_negative_lo_u16x8, vnegq_s16(positive_lo_i16x8),
                                                  positive_lo_i16x8);
            int16x8_t positive_hi_i16x8 = vreinterpretq_s16_u16(unsigned_hi_u16x8);
            int16x8_t scaled_hi_i16x8 = vbslq_s16(is_negative_hi_u16x8, vnegq_s16(positive_hi_i16x8),
                                                  positive_hi_i16x8);
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_lo_i16x8)));
            sum_i64x2 = vaddq_s64(sum_i64x2, vpaddlq_s32(vpaddlq_s16(scaled_hi_i16x8)));
            int32x4_t squares_lo_a_i32x4 = vmull_s16(vget_low_s16(scaled_lo_i16x8), vget_low_s16(scaled_lo_i16x8));
            int32x4_t squares_lo_b_i32x4 = vmull_high_s16(scaled_lo_i16x8, scaled_lo_i16x8);
            int32x4_t squares_hi_a_i32x4 = vmull_s16(vget_low_s16(scaled_hi_i16x8), vget_low_s16(scaled_hi_i16x8));
            int32x4_t squares_hi_b_i32x4 = vmull_high_s16(scaled_hi_i16x8, scaled_hi_i16x8);
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_lo_b_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_a_i32x4)));
            sumsq_u64x2 = vaddq_u64(sumsq_u64x2, vpaddlq_u32(vreinterpretq_u32_s32(squares_hi_b_i32x4)));
        }
    }
    nk_i64_t s_i64 = vgetq_lane_s64(sum_i64x2, 0) + vgetq_lane_s64(sum_i64x2, 1);
    nk_u64_t sq_u64 = vgetq_lane_u64(sumsq_u64x2, 0) + vgetq_lane_u64(sumsq_u64x2, 1);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e3m2_to_f32(*(nk_e3m2_t const *)(data + idx * stride_elements));
        s_i64 += (nk_i64_t)(val * 16.0f), sq_u64 += (nk_u64_t)(nk_i64_t)(val * val * 256.0f);
    }
    *sum = (nk_f32_t)s_i64 / 16.0f, *sumsq = (nk_f32_t)sq_u64 / 256.0f;
}

NK_PUBLIC void nk_reduce_moments_e3m2_neon(                         //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e3m2_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e3m2_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                    stride_bytes, &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_e3m2_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_neon_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,              //
    nk_e3m2_t *min_value, nk_size_t *min_index,          //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    uint8x16_t first_raw_u8x16 = vld1q_u8((nk_u8_t const *)data);
    uint8x16_t first_comparable_u8x16 = nk_fp6x16_to_comparable_neon_(first_raw_u8x16);
    uint8x16_t min_u8x16 = first_comparable_u8x16, max_u8x16 = first_comparable_u8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(tail_vec.u8x16);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u8x16 = vdupq_n_u8(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u8s[i] = 0xFF;
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    *min_value = nk_comparable_to_fp6_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp6_(max_comparable), *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e3m2_neon_strided_(                  //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_e3m2_t *min_value, nk_size_t *min_index,                        //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded_u8x16x2.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded_u8x16x3.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            uint8x16_t comparable_u8x16 = nk_fp6x16_to_comparable_neon_(loaded_u8x16x4.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u8_t raw = *(nk_u8_t const *)(data + idx * stride_elements);
        nk_u8_t magnitude = raw & 0x1F;
        nk_u8_t comparable = (raw & 0x20) ? (0x1F - magnitude) : (magnitude | 0x20);
        if (comparable < min_comparable) min_comparable = comparable, min_idx = idx;
        if (comparable > max_comparable) max_comparable = comparable, max_idx = idx;
    }
    *min_value = nk_comparable_to_fp6_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp6_(max_comparable), *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e3m2_neon(                          //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value, nk_size_t *min_index,                     //
    nk_e3m2_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0)
        *min_value = NK_E3M2_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E3M2_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e3m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_e3m2_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_e3m2_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                   &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_e3m2_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                   &right_maximum_index);
        if (nk_e3m2_compare_(&right_minimum_value, &left_minimum_value) < 0)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (nk_e3m2_compare_(&right_maximum_value, &left_maximum_value) > 0)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e3m2_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e3m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

// ─── e4m3 moments ────────────────────────────────────────────────────────────

NK_INTERNAL void nk_reduce_moments_e4m3_neon_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,               //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        float16x8_t half_lo_f16x8, half_hi_f16x8;
        nk_e4m3x16_to_f16x8x2_neon_(raw_u8x16, &half_lo_f16x8, &half_hi_f16x8);
        float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
        float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
        float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
        float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
        sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
        sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                        sumsq_f32x4, a_f32x4, a_f32x4),
                                                    b_f32x4, b_f32x4),
                                          c_f32x4, c_f32x4),
                                d_f32x4, d_f32x4);
    }
    nk_f32_t s = vaddvq_f32(sum_f32x4), sq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e4m3_to_f32(data[idx]);
        s += val, sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_e4m3_neon_strided_(                 //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            float16x8_t half_lo_f16x8, half_hi_f16x8;
            nk_e4m3x16_to_f16x8x2_neon_(loaded_u8x16x2.val[0], &half_lo_f16x8, &half_hi_f16x8);
            float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
            float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
            float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
            float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
            sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                            sumsq_f32x4, a_f32x4, a_f32x4),
                                                        b_f32x4, b_f32x4),
                                              c_f32x4, c_f32x4),
                                    d_f32x4, d_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            float16x8_t half_lo_f16x8, half_hi_f16x8;
            nk_e4m3x16_to_f16x8x2_neon_(loaded_u8x16x3.val[0], &half_lo_f16x8, &half_hi_f16x8);
            float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
            float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
            float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
            float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
            sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                            sumsq_f32x4, a_f32x4, a_f32x4),
                                                        b_f32x4, b_f32x4),
                                              c_f32x4, c_f32x4),
                                    d_f32x4, d_f32x4);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            float16x8_t half_lo_f16x8, half_hi_f16x8;
            nk_e4m3x16_to_f16x8x2_neon_(loaded_u8x16x4.val[0], &half_lo_f16x8, &half_hi_f16x8);
            float32x4_t a_f32x4 = vcvt_f32_f16(vget_low_f16(half_lo_f16x8));
            float32x4_t b_f32x4 = vcvt_high_f32_f16(half_lo_f16x8);
            float32x4_t c_f32x4 = vcvt_f32_f16(vget_low_f16(half_hi_f16x8));
            float32x4_t d_f32x4 = vcvt_high_f32_f16(half_hi_f16x8);
            sum_f32x4 = vaddq_f32(vaddq_f32(sum_f32x4, vaddq_f32(a_f32x4, b_f32x4)), vaddq_f32(c_f32x4, d_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(vfmaq_f32(vfmaq_f32( //
                                                            sumsq_f32x4, a_f32x4, a_f32x4),
                                                        b_f32x4, b_f32x4),
                                              c_f32x4, c_f32x4),
                                    d_f32x4, d_f32x4);
        }
    }
    nk_f32_t s = vaddvq_f32(sum_f32x4), sq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e4m3_to_f32(*(nk_e4m3_t const *)(data + idx * stride_elements));
        s += val, sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_e4m3_neon(                         //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e4m3_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e4m3_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                    stride_bytes, &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_e4m3_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
}

// ─── e4m3 minmax ─────────────────────────────────────────────────────────────

NK_INTERNAL void nk_reduce_minmax_e4m3_neon_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_e4m3_t *min_value, nk_size_t *min_index,          //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    uint8x16_t first_raw_u8x16 = vld1q_u8((nk_u8_t const *)data);
    uint8x16_t first_comparable_u8x16 = nk_fp8x16_to_comparable_neon_(first_raw_u8x16);
    uint8x16_t min_u8x16 = first_comparable_u8x16, max_u8x16 = first_comparable_u8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u8x16 = vdupq_n_u8(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u8s[i] = 0xFF;
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    *min_value = nk_comparable_to_fp8_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp8_(max_comparable), *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e4m3_neon_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_e4m3_t *min_value, nk_size_t *min_index,                        //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded_u8x16x2.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded_u8x16x3.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded_u8x16x4.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u8_t raw = *(nk_u8_t const *)(data + idx * stride_elements);
        nk_u8_t comparable = (raw & 0x80) ? (nk_u8_t)(~raw) : (raw ^ 0x80);
        if (comparable < min_comparable) min_comparable = comparable, min_idx = idx;
        if (comparable > max_comparable) max_comparable = comparable, max_idx = idx;
    }
    *min_value = nk_comparable_to_fp8_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp8_(max_comparable), *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e4m3_neon(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value, nk_size_t *min_index,                     //
    nk_e4m3_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0)
        *min_value = NK_E4M3_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E4M3_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e4m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_e4m3_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_e4m3_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                   &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_e4m3_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                   &right_maximum_index);
        if (nk_e4m3_compare_(&right_minimum_value, &left_minimum_value) < 0)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (nk_e4m3_compare_(&right_maximum_value, &left_maximum_value) > 0)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e4m3_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e4m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

// ─── e5m2 moments ────────────────────────────────────────────────────────────

NK_INTERNAL void nk_reduce_moments_e5m2_neon_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,               //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t raw_u8x8 = vld1_u8((nk_u8_t const *)(data + idx));
        float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(raw_u8x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
        float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
        sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
        sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
    }
    nk_f32_t s = vaddvq_f32(sum_f32x4), sq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e5m2_to_f32(data[idx]);
        s += val, sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_INTERNAL void nk_reduce_moments_e5m2_neon_strided_(                 //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0), sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x2_t loaded_u8x8x2 = vld2_u8((nk_u8_t const *)(data + idx * 2));
            float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x2.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
            float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
            sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x3_t loaded_u8x8x3 = vld3_u8((nk_u8_t const *)(data + idx * 3));
            float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x3.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
            float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
            sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
        }
    }
    else {
        for (; idx + 8 <= count; idx += 8) {
            uint8x8x4_t loaded_u8x8x4 = vld4_u8((nk_u8_t const *)(data + idx * 4));
            float16x8_t half_f16x8 = nk_e5m2x8_to_f16x8_neon_(loaded_u8x8x4.val[0]);
            float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(half_f16x8));
            float32x4_t hi_f32x4 = vcvt_high_f32_f16(half_f16x8);
            sum_f32x4 = vaddq_f32(sum_f32x4, vaddq_f32(lo_f32x4, hi_f32x4));
            sumsq_f32x4 = vfmaq_f32(vfmaq_f32(sumsq_f32x4, lo_f32x4, lo_f32x4), hi_f32x4, hi_f32x4);
        }
    }
    nk_f32_t s = vaddvq_f32(sum_f32x4), sq = vaddvq_f32(sumsq_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val = nk_e5m2_to_f32(*(nk_e5m2_t const *)(data + idx * stride_elements));
        s += val, sq += val * val;
    }
    *sum = s, *sumsq = sq;
}

NK_PUBLIC void nk_reduce_moments_e5m2_neon(                         //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum, left_sumsq, right_sum, right_sumsq;
        nk_reduce_moments_e5m2_neon(data, left_partition_count, stride_bytes, &left_sum, &left_sumsq);
        nk_reduce_moments_e5m2_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                    stride_bytes, &right_sum, &right_sumsq);
        *sum = left_sum + right_sum, *sumsq = left_sumsq + right_sumsq;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_neon_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_e5m2_neon_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
}

// ─── e5m2 minmax ─────────────────────────────────────────────────────────────

NK_INTERNAL void nk_reduce_minmax_e5m2_neon_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_e5m2_t *min_value, nk_size_t *min_index,          //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    uint8x16_t first_raw_u8x16 = vld1q_u8((nk_u8_t const *)data);
    uint8x16_t first_comparable_u8x16 = nk_fp8x16_to_comparable_neon_(first_raw_u8x16);
    uint8x16_t min_u8x16 = first_comparable_u8x16, max_u8x16 = first_comparable_u8x16;
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(1), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 16;
    for (; idx + 16 <= count; idx += 16) {
        uint8x16_t raw_u8x16 = vld1q_u8((nk_u8_t const *)(data + idx));
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(raw_u8x16);
        uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
        iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
    }
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b8x16_serial_(data + idx, &tail_vec, remaining);
        uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(tail_vec.u8x16);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u8x16 = vdupq_n_u8(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u8s[i] = 0xFF;
        uint8x16_t data_for_min_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0xFF));
        uint8x16_t data_for_max_u8x16 = vbslq_u8(valid_mask_vec.u8x16, comparable_u8x16, vdupq_n_u8(0));
        uint8x16_t less_u8x16 = vcltq_u8(data_for_min_u8x16, min_u8x16);
        uint8x16_t greater_u8x16 = vcgtq_u8(data_for_max_u8x16, max_u8x16);
        min_u8x16 = vbslq_u8(less_u8x16, data_for_min_u8x16, min_u8x16);
        max_u8x16 = vbslq_u8(greater_u8x16, data_for_max_u8x16, max_u8x16);
        min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
        max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    *min_value = nk_comparable_to_fp8_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp8_(max_comparable), *max_index = max_idx;
}

NK_INTERNAL void nk_reduce_minmax_e5m2_neon_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_e5m2_t *min_value, nk_size_t *min_index,                        //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    uint8x16_t min_u8x16 = vdupq_n_u8(0xFF), max_u8x16 = vdupq_n_u8(0);
    uint8x16_t min_iter_u8x16 = vdupq_n_u8(0), max_iter_u8x16 = vdupq_n_u8(0);
    uint8x16_t iter_u8x16 = vdupq_n_u8(0), one_u8x16 = vdupq_n_u8(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x2_t loaded_u8x16x2 = vld2q_u8((nk_u8_t const *)(data + idx * 2));
            uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded_u8x16x2.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x3_t loaded_u8x16x3 = vld3q_u8((nk_u8_t const *)(data + idx * 3));
            uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded_u8x16x3.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    else {
        for (; idx + 16 <= count; idx += 16) {
            uint8x16x4_t loaded_u8x16x4 = vld4q_u8((nk_u8_t const *)(data + idx * 4));
            uint8x16_t comparable_u8x16 = nk_fp8x16_to_comparable_neon_(loaded_u8x16x4.val[0]);
            uint8x16_t less_u8x16 = vcltq_u8(comparable_u8x16, min_u8x16);
            uint8x16_t greater_u8x16 = vcgtq_u8(comparable_u8x16, max_u8x16);
            min_u8x16 = vbslq_u8(less_u8x16, comparable_u8x16, min_u8x16);
            max_u8x16 = vbslq_u8(greater_u8x16, comparable_u8x16, max_u8x16);
            min_iter_u8x16 = vbslq_u8(less_u8x16, iter_u8x16, min_iter_u8x16);
            max_iter_u8x16 = vbslq_u8(greater_u8x16, iter_u8x16, max_iter_u8x16);
            iter_u8x16 = vaddq_u8(iter_u8x16, one_u8x16);
        }
    }
    nk_u8_t min_comparable = vminvq_u8(min_u8x16), max_comparable = vmaxvq_u8(max_u8x16);
    uint8x16_t min_value_match_u8x16 = vceqq_u8(min_u8x16, vdupq_n_u8(min_comparable));
    uint8x16_t masked_min_iter_u8x16 = vbslq_u8(min_value_match_u8x16, min_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_min_cycle = vminvq_u8(masked_min_iter_u8x16);
    uint8x16_t max_value_match_u8x16 = vceqq_u8(max_u8x16, vdupq_n_u8(max_comparable));
    uint8x16_t masked_max_iter_u8x16 = vbslq_u8(max_value_match_u8x16, max_iter_u8x16, vdupq_n_u8(0xFF));
    nk_u8_t earliest_max_cycle = vminvq_u8(masked_max_iter_u8x16);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.u8x16 = min_u8x16;
    maximum_values_vec.u8x16 = max_u8x16;
    minimum_iteration_indices_vec.u8x16 = min_iter_u8x16;
    maximum_iteration_indices_vec.u8x16 = max_iter_u8x16;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 16; ++i)
        if (minimum_values_vec.u8s[i] == min_comparable && minimum_iteration_indices_vec.u8s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 16; ++i)
        if (maximum_values_vec.u8s[i] == max_comparable && maximum_iteration_indices_vec.u8s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 16 + (nk_size_t)i;
            break;
        }
    for (; idx < count; ++idx) {
        nk_u8_t raw = *(nk_u8_t const *)(data + idx * stride_elements);
        nk_u8_t comparable = (raw & 0x80) ? (nk_u8_t)(~raw) : (raw ^ 0x80);
        if (comparable < min_comparable) min_comparable = comparable, min_idx = idx;
        if (comparable > max_comparable) max_comparable = comparable, max_idx = idx;
    }
    *min_value = nk_comparable_to_fp8_(min_comparable), *min_index = min_idx;
    *max_value = nk_comparable_to_fp8_(max_comparable), *max_index = max_idx;
}

NK_PUBLIC void nk_reduce_minmax_e5m2_neon(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value, nk_size_t *min_index,                     //
    nk_e5m2_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0)
        *min_value = NK_E5M2_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_E5M2_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e5m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)256 * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_e5m2_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_e5m2_neon(data, left_partition_count, stride_bytes, &left_minimum_value, &left_minimum_index,
                                   &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_e5m2_neon(data + left_partition_count * stride_elements, count - left_partition_count,
                                   stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                   &right_maximum_index);
        if (nk_e5m2_compare_(&right_minimum_value, &left_minimum_value) < 0)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (nk_e5m2_compare_(&right_maximum_value, &left_maximum_value) > 0)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_neon_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e5m2_neon_strided_(data, count, stride_elements, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_e5m2_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEON_NEW_H
