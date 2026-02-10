/**
 *  @brief SIMD-accelerated Vector Reductions for NEON FHM.
 *  @file include/numkong/reduce/neonfhm.h
 *  @author Ash Vardanian
 *  @date December 29, 2025
 *
 *  @sa include/numkong/reduce.h
 *
 *  @section reduce_neonfhm_instructions ARM NEON FP16 Matrix Instructions (ARMv8.4-FHM)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vld1_u8                     LD1 (V.8B)                      4cy         2/cy        3/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy        4/cy
 *      vget_low_f16                (extract low half, no instr)    0cy         -           -
 *      vget_high_f16               (extract high half, no instr)   0cy         -           -
 *      vaddq_f32                   FADD (V.4S, V.4S, V.4S)         2cy         2/cy        4/cy
 *      vcltq_f32                   FCMLT (V.4S, V.4S, V.4S)        2cy         2/cy        4/cy
 *      vcgtq_f32                   FCMGT (V.4S, V.4S, V.4S)        2cy         2/cy        4/cy
 *      vbslq_f32                   BSL (V.16B, V.16B, V.16B)       2cy         2/cy        4/cy
 *      vbslq_u32                   BSL (V.16B, V.16B, V.16B)       2cy         2/cy        4/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *
 *  This implementation targets E4M3 and E5M2 8-bit floating-point formats used in ML quantization.
 *  Values are first converted to F16 via lookup tables, then widened to F32 for accumulation.
 *  The ARMv8.4-FHM extension provides the underlying F16 infrastructure.
 *
 *  E4M3 (4-bit exponent, 3-bit mantissa) covers range [-448, 448] while E5M2 (5-bit exponent,
 *  2-bit mantissa) covers a wider range similar to FP16 but with lower precision. Both formats
 *  are critical for 8-bit quantized ML inference.
 */
#ifndef NK_REDUCE_NEONFHM_H
#define NK_REDUCE_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/cast/serial.h"
#include "numkong/reduce/serial.h"
#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

NK_INTERNAL void nk_reduce_add_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    uint8x8_t data_e4m3x8;
    nk_size_t idx = 0;

nk_reduce_add_e4m3_neonfhm_contiguous_cycle:
    if (idx + 8 <= count) {
        data_e4m3x8 = vld1_u8((uint8_t const *)data + idx);
        idx += 8;
    }
    else {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        data_e4m3x8 = tail_vec.u8x8;
        idx = count;
    }
    float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
    if (idx < count) goto nk_reduce_add_e4m3_neonfhm_contiguous_cycle;

    *result = vaddvq_f32(sum_f32x4);
}

NK_INTERNAL void nk_reduce_add_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
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
    }

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_reduce_add_e4m3_neonfhm(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (!aligned) nk_reduce_add_e4m3_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_e4m3_neonfhm_contiguous_(data, count, result);
    else nk_reduce_add_e4m3_neonfhm_strided_(data, count, stride_elements, result);
}

NK_INTERNAL void nk_reduce_add_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    float16x8_t ones_f16x8 = vdupq_n_f16(1.0f);
    uint8x8_t data_e5m2x8;
    nk_size_t idx = 0;

nk_reduce_add_e5m2_neonfhm_contiguous_cycle:
    if (idx + 8 <= count) {
        data_e5m2x8 = vld1_u8((uint8_t const *)data + idx);
        idx += 8;
    }
    else {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        data_e5m2x8 = tail_vec.u8x8;
        idx = count;
    }
    float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
    sum_f32x4 = vfmlalq_low_f16(sum_f32x4, data_f16x8, ones_f16x8);
    sum_f32x4 = vfmlalq_high_f16(sum_f32x4, data_f16x8, ones_f16x8);
    if (idx < count) goto nk_reduce_add_e5m2_neonfhm_contiguous_cycle;

    *result = vaddvq_f32(sum_f32x4);
}

NK_INTERNAL void nk_reduce_add_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
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
    }

    *result = vaddvq_f32(sum_f32x4);
}

NK_PUBLIC void nk_reduce_add_e5m2_neonfhm(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (!aligned) nk_reduce_add_e5m2_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_e5m2_neonfhm_contiguous_(data, count, result);
    else nk_reduce_add_e5m2_neonfhm_strided_(data, count, stride_elements, result);
}

NK_INTERNAL void nk_reduce_min_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

    float32x4_t min_low_f32x4 = vdupq_n_f32(NK_F32_MAX);
    float32x4_t min_high_f32x4 = vdupq_n_f32(NK_F32_MAX);
    uint32x4_t min_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8((uint8_t const *)data + idx));
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(tail_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        nk_size_t valid = count - idx;
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, min_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, min_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_INTERNAL void nk_reduce_min_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    float32x4_t min_low_f32x4 = vdupq_n_f32(NK_F32_MAX);
    float32x4_t min_high_f32x4 = vdupq_n_f32(NK_F32_MAX);
    uint32x4_t min_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_size_t valid = count - idx;
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < valid; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, min_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, min_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_PUBLIC void nk_reduce_min_e4m3_neonfhm(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_e4m3_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e4m3_neonfhm_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e4m3_neonfhm_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_min_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

    float32x4_t min_low_f32x4 = vdupq_n_f32(NK_F32_MAX);
    float32x4_t min_high_f32x4 = vdupq_n_f32(NK_F32_MAX);
    uint32x4_t min_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8((uint8_t const *)data + idx));
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_b64_vec_t tail_vec;
        nk_partial_load_b8x8_serial_(data + idx, &tail_vec, count - idx);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(tail_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        nk_size_t valid = count - idx;
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, min_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, min_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_INTERNAL void nk_reduce_min_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {

    float32x4_t min_low_f32x4 = vdupq_n_f32(NK_F32_MAX);
    float32x4_t min_high_f32x4 = vdupq_n_f32(NK_F32_MAX);
    uint32x4_t min_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t min_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_size_t valid = count - idx;
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < valid; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, min_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, min_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcltq_f32(data_low_f32x4, min_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcltq_f32(data_high_f32x4, min_high_f32x4);
        min_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, min_low_f32x4);
        min_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, min_high_f32x4);
        min_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, min_idx_low_u32x4);
        min_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, min_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_PUBLIC void nk_reduce_min_e5m2_neonfhm(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0 || !aligned) nk_reduce_min_e5m2_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e5m2_neonfhm_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e5m2_neonfhm_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

    float32x4_t max_low_f32x4 = vdupq_n_f32(NK_F32_MIN);
    float32x4_t max_high_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t max_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t max_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(data + idx));
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_size_t valid = count - idx;
        nk_b64_vec_t data_vec = {0};
        nk_partial_load_b8x8_serial_(data + idx, &data_vec, valid);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, max_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, max_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_INTERNAL void nk_reduce_max_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    float32x4_t max_low_f32x4 = vdupq_n_f32(NK_F32_MIN);
    float32x4_t max_high_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t max_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t max_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_size_t valid = count - idx;
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < valid; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, max_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, max_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_PUBLIC void nk_reduce_max_e4m3_neonfhm(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_e4m3_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e4m3_neonfhm_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e4m3_neonfhm_strided_(data, count, stride_elements, max_value, max_index);
}

NK_INTERNAL void nk_reduce_max_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

    float32x4_t max_low_f32x4 = vdupq_n_f32(NK_F32_MIN);
    float32x4_t max_high_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t max_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t max_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(data + idx));
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_size_t valid = count - idx;
        nk_b64_vec_t data_vec = {0};
        nk_partial_load_b8x8_serial_(data + idx, &data_vec, valid);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, max_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, max_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_INTERNAL void nk_reduce_max_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {

    float32x4_t max_low_f32x4 = vdupq_n_f32(NK_F32_MIN);
    float32x4_t max_high_f32x4 = vdupq_n_f32(NK_F32_MIN);
    uint32x4_t max_idx_low_u32x4 = vdupq_n_u32(0);
    uint32x4_t max_idx_high_u32x4 = vdupq_n_u32(0);
    uint32x4_t idx_low_u32x4 = {0, 1, 2, 3};
    uint32x4_t idx_high_u32x4 = {4, 5, 6, 7};
    uint32x4_t step_u32x4 = vdupq_n_u32(8);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < 8; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
        idx_low_u32x4 = vaddq_u32(idx_low_u32x4, step_u32x4);
        idx_high_u32x4 = vaddq_u32(idx_high_u32x4, step_u32x4);
    }
    if (idx < count) {
        nk_size_t valid = count - idx;
        nk_b64_vec_t data_vec = {0};
        for (nk_size_t i = 0; i < valid; ++i) {
            data_vec.u8s[i] = *ptr;
            ptr += stride_elements;
        }
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_vec.u8x8);
        float32x4_t data_low_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t data_high_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        uint32x4_t lane_low_u32x4 = {0, 1, 2, 3};
        uint32x4_t lane_high_u32x4 = {4, 5, 6, 7};
        uint32x4_t valid_u32x4 = vdupq_n_u32((uint32_t)valid);
        data_low_f32x4 = vbslq_f32(vcltq_u32(lane_low_u32x4, valid_u32x4), data_low_f32x4, max_low_f32x4);
        data_high_f32x4 = vbslq_f32(vcltq_u32(lane_high_u32x4, valid_u32x4), data_high_f32x4, max_high_f32x4);
        uint32x4_t mask_low_u32x4 = vcgtq_f32(data_low_f32x4, max_low_f32x4);
        uint32x4_t mask_high_u32x4 = vcgtq_f32(data_high_f32x4, max_high_f32x4);
        max_low_f32x4 = vbslq_f32(mask_low_u32x4, data_low_f32x4, max_low_f32x4);
        max_high_f32x4 = vbslq_f32(mask_high_u32x4, data_high_f32x4, max_high_f32x4);
        max_idx_low_u32x4 = vbslq_u32(mask_low_u32x4, idx_low_u32x4, max_idx_low_u32x4);
        max_idx_high_u32x4 = vbslq_u32(mask_high_u32x4, idx_high_u32x4, max_idx_high_u32x4);
    }

    // Horizontal reduction across both halves
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

NK_PUBLIC void nk_reduce_max_e5m2_neonfhm(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (count == 0 || !aligned) nk_reduce_max_e5m2_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e5m2_neonfhm_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e5m2_neonfhm_strided_(data, count, stride_elements, max_value, max_index);
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
#endif // NK_REDUCE_NEONFHM_H
