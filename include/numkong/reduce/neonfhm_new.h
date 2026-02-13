/**
 *  @brief ARMv8.4-FHM implementations for the redesigned reduction API (moments + minmax).
 *  @file include/numkong/reduce/neonfhm_new.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_NEONFHM_NEW_H
#define NK_REDUCE_NEONFHM_NEW_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/cast/serial.h"
#include "numkong/cast/neon.h" // nk_e4m3x8_to_f16x8_neon_, nk_e5m2x8_to_f16x8_neon_, nk_e3m2x8_to_f16x8_neon_
#include "numkong/reduce/serial_new.h"
#include "numkong/reduce/neon.h" // nk_reduce_add_f32x4_neon_

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

NK_INTERNAL void nk_reduce_moments_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((uint8_t const *)(data + idx));
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    // Tail: partial load for remaining elements (< 8)
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(tail_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum = vaddvq_f32(sum_f32x4);
    *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_e4m3_neonfhm_strided_(              //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    if (idx < count) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; idx + i < count; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum = vaddvq_f32(sum_f32x4);
    *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_e4m3_neonfhm(                      //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)5000 * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_e4m3_neonfhm(data, left_partition_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_e4m3_neonfhm(data + left_partition_count * stride_elements, count - left_partition_count,
                                       stride_bytes, &right_sum_value, &right_sumsq_value);
        *sum = left_sum_value + right_sum_value, *sumsq = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_e4m3_neonfhm_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_e4m3_neonfhm_strided_(data, count, stride_elements, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((uint8_t const *)(data + idx));
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    // Tail: partial load for remaining elements (< 8)
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(tail_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum = vaddvq_f32(sum_f32x4);
    *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_e5m2_neonfhm_strided_(              //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {

    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    if (idx < count) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; idx + i < count; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }

    *sum = vaddvq_f32(sum_f32x4);
    *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_e5m2_neonfhm(                      //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)5000 * 8) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_e5m2_neonfhm(data, left_partition_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_e5m2_neonfhm(data + left_partition_count * stride_elements, count - left_partition_count,
                                       stride_bytes, &right_sum_value, &right_sumsq_value);
        *sum = left_sum_value + right_sum_value, *sumsq = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_e5m2_neonfhm_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_e5m2_neonfhm_strided_(data, count, stride_elements, sum, sumsq);
}

NK_INTERNAL void nk_reduce_moments_e3m2_neonfhm_contiguous_( //
    nk_e3m2_t const *data, nk_size_t count,                  //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_u8x8 = vld1_u8((uint8_t const *)(data + idx));
        float16x8_t data_f16x8 = nk_e3m2x8_to_f16x8_neon_(data_u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e3m2x8_to_f16x8_neon_(tail_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }
    *sum = vaddvq_f32(sum_f32x4), *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_INTERNAL void nk_reduce_moments_e3m2_neonfhm_strided_(              //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float32x4_t sumsq_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    nk_e3m2_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e3m2x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }
    if (idx < count) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; idx + i < count; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e3m2x8_to_f16x8_neon_(data_vec.u8x8);
        sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
        sumsq_f32x4 = vfmlalq_low_f16(sumsq_f32x4, data_f16x8, data_f16x8);
        sumsq_f32x4 = vfmlalq_high_f16(sumsq_f32x4, data_f16x8, data_f16x8);
    }
    *sum = vaddvq_f32(sum_f32x4), *sumsq = vaddvq_f32(sumsq_f32x4);
}

NK_PUBLIC void nk_reduce_moments_e3m2_neonfhm(                      //
    nk_e3m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum, nk_f32_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data, count, stride_bytes, sum, sumsq);
    else if (count > (nk_size_t)(NK_U16_MAX + 1) * 16) {
        nk_size_t left_partition_count = count / 2;
        nk_f32_t left_sum_value, left_sumsq_value, right_sum_value, right_sumsq_value;
        nk_reduce_moments_e3m2_neonfhm(data, left_partition_count, stride_bytes, &left_sum_value, &left_sumsq_value);
        nk_reduce_moments_e3m2_neonfhm(data + left_partition_count * stride_elements, count - left_partition_count,
                                       stride_bytes, &right_sum_value, &right_sumsq_value);
        *sum = left_sum_value + right_sum_value, *sumsq = left_sumsq_value + right_sumsq_value;
    }
    else if (stride_elements == 1) nk_reduce_moments_e3m2_neonfhm_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_e3m2_neonfhm_strided_(data, count, stride_elements, sum, sumsq);
}

/** @brief Convert 16 raw FP8 sign-magnitude bytes to unsigned-comparable bytes. */
NK_INTERNAL uint8x16_t nk_fp8x16_to_comparable_neon_(uint8x16_t raw_u8x16) {
    uint8x16_t sign_mask_u8x16 = vdupq_n_u8(0x80);
    // vtstq_u8: for each lane, result = (raw & sign_mask) != 0 ? 0xFF : 0x00
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

NK_INTERNAL void nk_reduce_minmax_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,                 //
    nk_e4m3_t *min_value, nk_size_t *min_index,             //
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

NK_INTERNAL void nk_reduce_minmax_e4m3_neonfhm_strided_(               //
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

NK_PUBLIC void nk_reduce_minmax_e4m3_neonfhm(                       //
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
        nk_reduce_minmax_e4m3_neonfhm(data, left_partition_count, stride_bytes, &left_minimum_value,
                                      &left_minimum_index, &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_e4m3_neonfhm(data + left_partition_count * stride_elements, count - left_partition_count,
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
        nk_reduce_minmax_e4m3_neonfhm_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e4m3_neonfhm_strided_(data, count, stride_elements, min_value, min_index, max_value,
                                               max_index);
    else nk_reduce_minmax_e4m3_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,                 //
    nk_e5m2_t *min_value, nk_size_t *min_index,             //
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

NK_INTERNAL void nk_reduce_minmax_e5m2_neonfhm_strided_(               //
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

NK_PUBLIC void nk_reduce_minmax_e5m2_neonfhm(                       //
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
        nk_reduce_minmax_e5m2_neonfhm(data, left_partition_count, stride_bytes, &left_minimum_value,
                                      &left_minimum_index, &left_maximum_value, &left_maximum_index);
        nk_reduce_minmax_e5m2_neonfhm(data + left_partition_count * stride_elements, count - left_partition_count,
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
        nk_reduce_minmax_e5m2_neonfhm_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_minmax_e5m2_neonfhm_strided_(data, count, stride_elements, min_value, min_index, max_value,
                                               max_index);
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

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_
#endif // NK_REDUCE_NEONFHM_NEW_H
