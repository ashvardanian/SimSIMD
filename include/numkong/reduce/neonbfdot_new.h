/**
 *  @brief ARMv8.6-BF16 implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonbfdot_new.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONBFDOT_NEW_H
#define NK_REDUCE_NEONBFDOT_NEW_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/cast/neon.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial_new.h"
#include "numkong/reduce/neon.h" // for nk_reduce_add_f32x4_neon_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

NK_INTERNAL void nk_reduce_moments_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                    //
    nk_f32_t *sum, nk_f32_t *sumsq) {

    // bf16 representation of 1.0 is 0x3F80 (same as upper 16 bits of f32 1.0)
    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        bfloat16x8_t data_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(data + idx));
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
    }

    // Handle tail with type-agnostic partial load
    if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, count - idx);
        bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(tail_vec.u16x8);
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
    }

    *sum = vaddvq_f32(sum_f32x4);
    *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_bf16_neonbfdot_strided_(            //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {

    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x2.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x3.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x4.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
            sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
        }
    }

    // Gather tail into contiguous buffer, then dot with ones
    if (idx < count) {
        nk_b128_vec_t tail_vec = {{0}};
        nk_size_t remaining = count - idx;
        for (nk_size_t k = 0; k < remaining; ++k)
            tail_vec.u16s[k] = *(nk_u16_t const *)(data + (idx + k) * stride_elements);
        bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(tail_vec.u16x8);
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        sumsq_f32x4 = vbfdotq_f32(sumsq_f32x4, data_bf16x8, data_bf16x8);
    }

    *sum = vaddvq_f32(sum_f32x4);
    *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_bf16_neonbfdot(                    //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_bf16_neonbfdot(data, left_partition_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_bf16_neonbfdot(data + left_partition_count * stride_elements, count - left_partition_count,
                                         stride_bytes, &right_sum_value, &right_sumsq_value);
        *sum = left_sum_value + right_sum_value, *sumsq = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_bf16_neonbfdot_contiguous_(data, count, sum, sumsq);
    else if (stride_elements <= 4) nk_reduce_moments_bf16_neonbfdot_strided_(data, count, stride_elements, sum, sumsq);
    else nk_reduce_moments_bf16_serial(data, count, stride_bytes, sum, sumsq);
}

// ─── bf16 comparable i16 helpers ─────────────────────────────────────────────

/** @brief Convert 8 raw bf16 sign-magnitude u16 to order-preserving comparable i16. */
NK_INTERNAL int16x8_t nk_bf16x8_to_comparable_i16x8_neon_(uint16x8_t raw_u16x8) {
    uint16x8_t sign_mask_u16x8 = vdupq_n_u16(0x8000);
    uint16x8_t is_negative_u16x8 = vtstq_u16(raw_u16x8, sign_mask_u16x8);
    uint16x8_t flip_positive_u16x8 = veorq_u16(raw_u16x8, sign_mask_u16x8);
    uint16x8_t flip_negative_u16x8 = vmvnq_u16(raw_u16x8);
    return vreinterpretq_s16_u16(vbslq_u16(is_negative_u16x8, flip_negative_u16x8, flip_positive_u16x8));
}

/** @brief Convert a comparable i16 value back to raw bf16 u16 bits. */
NK_INTERNAL nk_u16_t nk_comparable_i16_to_bf16_raw_(nk_i16_t comparable) {
    nk_u16_t unsigned_comparable = (nk_u16_t)comparable;
    if (comparable >= 0) return unsigned_comparable ^ 0x8000;
    else return ~unsigned_comparable;
}

// ─── bf16 minmax contiguous ──────────────────────────────────────────────────

NK_INTERNAL void nk_reduce_minmax_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                   //
    nk_bf16_t *min_value, nk_size_t *min_index,               //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    uint16x8_t first_raw_u16x8 = vld1q_u16((uint16_t const *)data);
    int16x8_t first_comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(first_raw_u16x8);
    int16x8_t min_i16x8 = first_comparable_i16x8, max_i16x8 = first_comparable_i16x8;
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(1), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 8;
    for (; idx + 8 <= count; idx += 8) {
        uint16x8_t raw_u16x8 = vld1q_u16((uint16_t const *)(data + idx));
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(raw_u16x8);
        uint16x8_t less_u16x8 = vcltq_s16(comparable_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(comparable_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, comparable_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, comparable_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
        iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
    }
    // Handle tail with partial load and identity masking
    nk_size_t remaining = count - idx;
    if (remaining > 0) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_serial_(data + idx, &tail_vec, remaining);
        int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(tail_vec.u16x8);
        nk_b128_vec_t valid_mask_vec;
        valid_mask_vec.u16x8 = vdupq_n_u16(0);
        for (nk_size_t i = 0; i < remaining; ++i) valid_mask_vec.u16s[i] = 0xFFFF;
        int16x8_t data_for_min_i16x8 = vbslq_s16(valid_mask_vec.u16x8, comparable_i16x8, vdupq_n_s16(NK_I16_MAX));
        int16x8_t data_for_max_i16x8 = vbslq_s16(valid_mask_vec.u16x8, comparable_i16x8, vdupq_n_s16(NK_I16_MIN));
        uint16x8_t less_u16x8 = vcltq_s16(data_for_min_i16x8, min_i16x8);
        uint16x8_t greater_u16x8 = vcgtq_s16(data_for_max_i16x8, max_i16x8);
        min_i16x8 = vbslq_s16(less_u16x8, data_for_min_i16x8, min_i16x8);
        max_i16x8 = vbslq_s16(greater_u16x8, data_for_max_i16x8, max_i16x8);
        min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
        max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
    }
    // Horizontal reduction
    nk_i16_t min_comparable = vminvq_s16(min_i16x8), max_comparable = vmaxvq_s16(max_i16x8);
    uint16x8_t min_value_match_u16x8 = vceqq_s16(min_i16x8, vdupq_n_s16(min_comparable));
    uint16x8_t masked_min_iter_u16x8 = vbslq_u16(min_value_match_u16x8, min_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_min_cycle = vminvq_u16(masked_min_iter_u16x8);
    uint16x8_t max_value_match_u16x8 = vceqq_s16(max_i16x8, vdupq_n_s16(max_comparable));
    uint16x8_t masked_max_iter_u16x8 = vbslq_u16(max_value_match_u16x8, max_iter_u16x8, vdupq_n_u16(0xFFFF));
    nk_u16_t earliest_max_cycle = vminvq_u16(masked_max_iter_u16x8);
    nk_b128_vec_t minimum_values_vec, maximum_values_vec, minimum_iteration_indices_vec, maximum_iteration_indices_vec;
    minimum_values_vec.i16x8 = min_i16x8;
    maximum_values_vec.i16x8 = max_i16x8;
    minimum_iteration_indices_vec.u16x8 = min_iter_u16x8;
    maximum_iteration_indices_vec.u16x8 = max_iter_u16x8;
    nk_size_t min_idx = 0, max_idx = 0;
    for (int i = 0; i < 8; ++i)
        if (minimum_values_vec.i16s[i] == min_comparable &&
            minimum_iteration_indices_vec.u16s[i] == earliest_min_cycle) {
            min_idx = (nk_size_t)earliest_min_cycle * 8 + (nk_size_t)i;
            break;
        }
    for (int i = 0; i < 8; ++i)
        if (maximum_values_vec.i16s[i] == max_comparable &&
            maximum_iteration_indices_vec.u16s[i] == earliest_max_cycle) {
            max_idx = (nk_size_t)earliest_max_cycle * 8 + (nk_size_t)i;
            break;
        }
    // Convert comparable back to bf16 raw bits
    nk_u16_t min_raw = nk_comparable_i16_to_bf16_raw_(min_comparable);
    nk_u16_t max_raw = nk_comparable_i16_to_bf16_raw_(max_comparable);
    *(nk_u16_t *)min_value = min_raw, *min_index = min_idx;
    *(nk_u16_t *)max_value = max_raw, *max_index = max_idx;
}

// ─── bf16 minmax strided ─────────────────────────────────────────────────────

NK_INTERNAL void nk_reduce_minmax_bf16_neonbfdot_strided_(             //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_bf16_t *min_value, nk_size_t *min_index,                        //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    int16x8_t min_i16x8 = vdupq_n_s16(NK_I16_MAX), max_i16x8 = vdupq_n_s16(NK_I16_MIN);
    uint16x8_t min_iter_u16x8 = vdupq_n_u16(0), max_iter_u16x8 = vdupq_n_u16(0);
    uint16x8_t iter_u16x8 = vdupq_n_u16(0), one_u16x8 = vdupq_n_u16(1);
    nk_size_t idx = 0;
    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(loaded_u16x8x2.val[0]);
            uint16x8_t less_u16x8 = vcltq_s16(comparable_i16x8, min_i16x8);
            uint16x8_t greater_u16x8 = vcgtq_s16(comparable_i16x8, max_i16x8);
            min_i16x8 = vbslq_s16(less_u16x8, comparable_i16x8, min_i16x8);
            max_i16x8 = vbslq_s16(greater_u16x8, comparable_i16x8, max_i16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(loaded_u16x8x3.val[0]);
            uint16x8_t less_u16x8 = vcltq_s16(comparable_i16x8, min_i16x8);
            uint16x8_t greater_u16x8 = vcgtq_s16(comparable_i16x8, max_i16x8);
            min_i16x8 = vbslq_s16(less_u16x8, comparable_i16x8, min_i16x8);
            max_i16x8 = vbslq_s16(greater_u16x8, comparable_i16x8, max_i16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            int16x8_t comparable_i16x8 = nk_bf16x8_to_comparable_i16x8_neon_(loaded_u16x8x4.val[0]);
            uint16x8_t less_u16x8 = vcltq_s16(comparable_i16x8, min_i16x8);
            uint16x8_t greater_u16x8 = vcgtq_s16(comparable_i16x8, max_i16x8);
            min_i16x8 = vbslq_s16(less_u16x8, comparable_i16x8, min_i16x8);
            max_i16x8 = vbslq_s16(greater_u16x8, comparable_i16x8, max_i16x8);
            min_iter_u16x8 = vbslq_u16(less_u16x8, iter_u16x8, min_iter_u16x8);
            max_iter_u16x8 = vbslq_u16(greater_u16x8, iter_u16x8, max_iter_u16x8);
            iter_u16x8 = vaddq_u16(iter_u16x8, one_u16x8);
        }
    }
    // Horizontal reduction from SIMD lanes
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
    // Scalar tail: process remaining elements one by one
    for (; idx < count; ++idx) {
        nk_u16_t raw = *(nk_u16_t const *)(data + idx * stride_elements);
        nk_i16_t comparable = (raw & 0x8000) ? (nk_i16_t)(~raw) : (nk_i16_t)(raw ^ 0x8000);
        if (comparable < minimum_scalar) minimum_scalar = comparable, min_idx = idx;
        if (comparable > maximum_scalar) maximum_scalar = comparable, max_idx = idx;
    }
    // Convert comparable back to bf16 raw bits
    nk_u16_t min_raw = nk_comparable_i16_to_bf16_raw_(minimum_scalar);
    nk_u16_t max_raw = nk_comparable_i16_to_bf16_raw_(maximum_scalar);
    *(nk_u16_t *)min_value = min_raw, *min_index = min_idx;
    *(nk_u16_t *)max_value = max_raw, *max_index = max_idx;
}

// ─── bf16 minmax dispatcher ──────────────────────────────────────────────────

NK_PUBLIC void nk_reduce_minmax_bf16_neonbfdot(                     //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value, nk_size_t *min_index,                     //
    nk_bf16_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (count == 0) {
        *(nk_u16_t *)min_value = NK_BF16_MAX, *min_index = NK_SIZE_MAX;
        *(nk_u16_t *)max_value = NK_BF16_MIN, *max_index = NK_SIZE_MAX;
    }
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_bf16_t left_minimum_value, right_minimum_value, left_maximum_value, right_maximum_value;
        nk_size_t left_minimum_index, right_minimum_index, left_maximum_index, right_maximum_index;
        nk_reduce_minmax_bf16_neonbfdot(data, left_partition_count, stride_bytes, &left_minimum_value,
                                        &left_minimum_index, &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_bf16_neonbfdot(data + left_partition_count * stride_elements, count - left_partition_count,
                                        stride_bytes, &right_minimum_value, &right_minimum_index, &right_maximum_value,
                                        &right_maximum_index);
        if (nk_bf16_compare_(right_minimum_value, left_minimum_value) < 0)
            *min_value = right_minimum_value, *min_index = left_partition_count + right_minimum_index;
        else *min_value = left_minimum_value, *min_index = left_minimum_index;
        if (nk_bf16_compare_(right_maximum_value, left_maximum_value) > 0)
            *max_value = right_maximum_value, *max_index = left_partition_count + right_maximum_index;
        else *max_value = left_maximum_value, *max_index = left_maximum_index;
    }
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_neonbfdot_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_bf16_neonbfdot_strided_(data, count, stride_elements, min_value, min_index, max_value,
                                                 max_index);
    else nk_reduce_minmax_bf16_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONBFDOT_NEW_H
