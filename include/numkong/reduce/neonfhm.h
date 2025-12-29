/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON with FHM (FEAT_FP16FML).
 *  @file include/numkong/reduce/neonfhm.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 29, 2025
 *
 *  This implementation uses the f16 intermediate path for e4m3/e5m2 reductions,
 *  leveraging the FEAT_FP16FML feature for efficient f8 to f16 conversions.
 */
#ifndef NK_REDUCE_NEONFHM_H
#define NK_REDUCE_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Convert 8 E4M3 values to f16x8 via bit manipulation (NEON).
 *  E4M3 format: S EEEE MMM (bias=7). F16: sign<<15, (exp+8)<<10, mant<<7. */
NK_INTERNAL float16x8_t nk_e4m3x8_to_f16x8_neonfhm_(uint8x8_t e4m3_u8x8) {
    // Widen to 16-bit lanes
    uint16x8_t v_u16x8 = vmovl_u8(e4m3_u8x8);

    // Extract sign, exponent, and mantissa
    uint16x8_t sign_u16x8 = vshlq_n_u16(vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x80)), 7), 15);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 3), vdupq_n_u16(0x0F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x07));

    // Build f16 representation: sign | ((exp + 8) << 10) | (mant << 7)
    // F16 has bias=15, E4M3 has bias=7, so we add (15-7)=8 to convert exponent
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(vaddq_u16(exp_u16x8, vdupq_n_u16(8)), 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 7);
    uint16x8_t f16_bits_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));

    // Zero out denormals (when exp == 0)
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    f16_bits_u16x8 = vbicq_u16(f16_bits_u16x8, zero_mask_u16x8);

    return vreinterpretq_f16_u16(f16_bits_u16x8);
}

/** @brief Convert 8 E5M2 values to f16x8 via bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). F16: sign<<15, exp<<10, mant<<8. */
NK_INTERNAL float16x8_t nk_e5m2x8_to_f16x8_neonfhm_(uint8x8_t e5m2_u8x8) {
    // Widen to 16-bit lanes
    uint16x8_t v_u16x8 = vmovl_u8(e5m2_u8x8);

    // Extract sign, exponent, and mantissa
    uint16x8_t sign_u16x8 = vshlq_n_u16(vshrq_n_u16(vandq_u16(v_u16x8, vdupq_n_u16(0x80)), 7), 15);
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(v_u16x8, 2), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vandq_u16(v_u16x8, vdupq_n_u16(0x03));

    // Build f16 representation: sign | (exp << 10) | (mant << 8)
    // F16 has bias=15, E5M2 has bias=15, so exponent stays the same
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(exp_u16x8, 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 8);
    uint16x8_t f16_bits_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));

    // Zero out denormals (when exp == 0)
    uint16x8_t zero_mask_u16x8 = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    f16_bits_u16x8 = vbicq_u16(f16_bits_u16x8, zero_mask_u16x8);

    return vreinterpretq_f16_u16(f16_bits_u16x8);
}

NK_INTERNAL void nk_reduce_add_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    // Main loop: process 8 elements at a time
    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e4m3x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neonfhm_(data_e4m3x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
    }

    nk_f32_t sum = vaddvq_f32(sum_f32x4);

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e4m3_to_f32(&data[idx], &val);
        sum += val;
    }

    *result = sum;
}

NK_INTERNAL void nk_reduce_add_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;

    // Gather 8 elements at a time into a buffer
    for (; idx + 8 <= count; idx += 8) {
        nk_u8_t buf[8];
        for (nk_size_t i = 0; i < 8; ++i) {
            buf[i] = *ptr;
            ptr += stride_elements;
        }
        uint8x8_t data_e4m3x8 = vld1_u8(buf);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neonfhm_(data_e4m3x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
    }

    nk_f32_t sum = vaddvq_f32(sum_f32x4);

    // Scalar tail
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32(ptr, &val);
        sum += val;
    }

    *result = sum;
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
    nk_size_t idx = 0;

    // Main loop: process 8 elements at a time
    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e5m2x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neonfhm_(data_e5m2x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
    }

    nk_f32_t sum = vaddvq_f32(sum_f32x4);

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e5m2_to_f32(&data[idx], &val);
        sum += val;
    }

    *result = sum;
}

NK_INTERNAL void nk_reduce_add_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;

    // Gather 8 elements at a time into a buffer
    for (; idx + 8 <= count; idx += 8) {
        nk_u8_t buf[8];
        for (nk_size_t i = 0; i < 8; ++i) {
            buf[i] = *ptr;
            ptr += stride_elements;
        }
        uint8x8_t data_e5m2x8 = vld1_u8(buf);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neonfhm_(data_e5m2x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));
        sum_f32x4 = vaddq_f32(sum_f32x4, lo_f32x4);
        sum_f32x4 = vaddq_f32(sum_f32x4, hi_f32x4);
    }

    nk_f32_t sum = vaddvq_f32(sum_f32x4);

    // Scalar tail
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32(ptr, &val);
        sum += val;
    }

    *result = sum;
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
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    // Track min values in f32, indices in 2x int32x4 (for 8 lanes)
    float32x4_t min_lo_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    float32x4_t min_hi_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t min_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e4m3x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neonfhm_(data_e4m3x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        // Compare and update min values and indices
        uint32x4_t lt_lo_u32x4 = vcltq_f32(lo_f32x4, min_lo_f32x4);
        uint32x4_t lt_hi_u32x4 = vcltq_f32(hi_f32x4, min_hi_f32x4);
        min_lo_f32x4 = vbslq_f32(lt_lo_u32x4, lo_f32x4, min_lo_f32x4);
        min_hi_f32x4 = vbslq_f32(lt_hi_u32x4, hi_f32x4, min_hi_f32x4);
        min_idx_lo_i32x4 = vbslq_s32(lt_lo_u32x4, idx_lo_i32x4, min_idx_lo_i32x4);
        min_idx_hi_i32x4 = vbslq_s32(lt_hi_u32x4, idx_hi_i32x4, min_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    // Horizontal reduction: find best lane across both vectors
    nk_b128_vec_t min_lo_vec, min_hi_vec, idx_lo_vec, idx_hi_vec;
    min_lo_vec.f32x4 = min_lo_f32x4;
    min_hi_vec.f32x4 = min_hi_f32x4;
    idx_lo_vec.i32x4 = min_idx_lo_i32x4;
    idx_hi_vec.i32x4 = min_idx_hi_i32x4;

    nk_f32_t best_val = min_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_lo_vec.f32s[i] < best_val) {
            best_val = min_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (min_hi_vec.f32s[i] < best_val) {
            best_val = min_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e4m3_to_f32(&data[idx], &val);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    float32x4_t min_lo_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    float32x4_t min_hi_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t min_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        nk_u8_t buf[8];
        for (nk_size_t i = 0; i < 8; ++i) {
            buf[i] = *ptr;
            ptr += stride_elements;
        }
        uint8x8_t data_e4m3x8 = vld1_u8(buf);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neonfhm_(data_e4m3x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        uint32x4_t lt_lo_u32x4 = vcltq_f32(lo_f32x4, min_lo_f32x4);
        uint32x4_t lt_hi_u32x4 = vcltq_f32(hi_f32x4, min_hi_f32x4);
        min_lo_f32x4 = vbslq_f32(lt_lo_u32x4, lo_f32x4, min_lo_f32x4);
        min_hi_f32x4 = vbslq_f32(lt_hi_u32x4, hi_f32x4, min_hi_f32x4);
        min_idx_lo_i32x4 = vbslq_s32(lt_lo_u32x4, idx_lo_i32x4, min_idx_lo_i32x4);
        min_idx_hi_i32x4 = vbslq_s32(lt_hi_u32x4, idx_hi_i32x4, min_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    // Horizontal reduction
    nk_b128_vec_t min_lo_vec, min_hi_vec, idx_lo_vec, idx_hi_vec;
    min_lo_vec.f32x4 = min_lo_f32x4;
    min_hi_vec.f32x4 = min_hi_f32x4;
    idx_lo_vec.i32x4 = min_idx_lo_i32x4;
    idx_hi_vec.i32x4 = min_idx_hi_i32x4;

    nk_f32_t best_val = min_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_lo_vec.f32s[i] < best_val) {
            best_val = min_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (min_hi_vec.f32s[i] < best_val) {
            best_val = min_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32(ptr, &val);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_e4m3_neonfhm(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (!aligned) nk_reduce_min_e4m3_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e4m3_neonfhm_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e4m3_neonfhm_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_min_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    float32x4_t min_lo_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    float32x4_t min_hi_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t min_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e5m2x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neonfhm_(data_e5m2x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        uint32x4_t lt_lo_u32x4 = vcltq_f32(lo_f32x4, min_lo_f32x4);
        uint32x4_t lt_hi_u32x4 = vcltq_f32(hi_f32x4, min_hi_f32x4);
        min_lo_f32x4 = vbslq_f32(lt_lo_u32x4, lo_f32x4, min_lo_f32x4);
        min_hi_f32x4 = vbslq_f32(lt_hi_u32x4, hi_f32x4, min_hi_f32x4);
        min_idx_lo_i32x4 = vbslq_s32(lt_lo_u32x4, idx_lo_i32x4, min_idx_lo_i32x4);
        min_idx_hi_i32x4 = vbslq_s32(lt_hi_u32x4, idx_hi_i32x4, min_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    nk_b128_vec_t min_lo_vec, min_hi_vec, idx_lo_vec, idx_hi_vec;
    min_lo_vec.f32x4 = min_lo_f32x4;
    min_hi_vec.f32x4 = min_hi_f32x4;
    idx_lo_vec.i32x4 = min_idx_lo_i32x4;
    idx_hi_vec.i32x4 = min_idx_hi_i32x4;

    nk_f32_t best_val = min_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_lo_vec.f32s[i] < best_val) {
            best_val = min_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (min_hi_vec.f32s[i] < best_val) {
            best_val = min_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e5m2_to_f32(&data[idx], &val);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    float32x4_t min_lo_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    float32x4_t min_hi_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t min_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        nk_u8_t buf[8];
        for (nk_size_t i = 0; i < 8; ++i) {
            buf[i] = *ptr;
            ptr += stride_elements;
        }
        uint8x8_t data_e5m2x8 = vld1_u8(buf);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neonfhm_(data_e5m2x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        uint32x4_t lt_lo_u32x4 = vcltq_f32(lo_f32x4, min_lo_f32x4);
        uint32x4_t lt_hi_u32x4 = vcltq_f32(hi_f32x4, min_hi_f32x4);
        min_lo_f32x4 = vbslq_f32(lt_lo_u32x4, lo_f32x4, min_lo_f32x4);
        min_hi_f32x4 = vbslq_f32(lt_hi_u32x4, hi_f32x4, min_hi_f32x4);
        min_idx_lo_i32x4 = vbslq_s32(lt_lo_u32x4, idx_lo_i32x4, min_idx_lo_i32x4);
        min_idx_hi_i32x4 = vbslq_s32(lt_hi_u32x4, idx_hi_i32x4, min_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    nk_b128_vec_t min_lo_vec, min_hi_vec, idx_lo_vec, idx_hi_vec;
    min_lo_vec.f32x4 = min_lo_f32x4;
    min_hi_vec.f32x4 = min_hi_f32x4;
    idx_lo_vec.i32x4 = min_idx_lo_i32x4;
    idx_hi_vec.i32x4 = min_idx_hi_i32x4;

    nk_f32_t best_val = min_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_lo_vec.f32s[i] < best_val) {
            best_val = min_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (min_hi_vec.f32s[i] < best_val) {
            best_val = min_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32(ptr, &val);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_e5m2_neonfhm(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (!aligned) nk_reduce_min_e5m2_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e5m2_neonfhm_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e5m2_neonfhm_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    // Track max values in f32, indices in 2x int32x4 (for 8 lanes)
    float32x4_t max_lo_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    float32x4_t max_hi_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t max_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e4m3x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neonfhm_(data_e4m3x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        // Compare and update max values and indices
        uint32x4_t gt_lo_u32x4 = vcgtq_f32(lo_f32x4, max_lo_f32x4);
        uint32x4_t gt_hi_u32x4 = vcgtq_f32(hi_f32x4, max_hi_f32x4);
        max_lo_f32x4 = vbslq_f32(gt_lo_u32x4, lo_f32x4, max_lo_f32x4);
        max_hi_f32x4 = vbslq_f32(gt_hi_u32x4, hi_f32x4, max_hi_f32x4);
        max_idx_lo_i32x4 = vbslq_s32(gt_lo_u32x4, idx_lo_i32x4, max_idx_lo_i32x4);
        max_idx_hi_i32x4 = vbslq_s32(gt_hi_u32x4, idx_hi_i32x4, max_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    // Horizontal reduction: find best lane across both vectors
    nk_b128_vec_t max_lo_vec, max_hi_vec, idx_lo_vec, idx_hi_vec;
    max_lo_vec.f32x4 = max_lo_f32x4;
    max_hi_vec.f32x4 = max_hi_f32x4;
    idx_lo_vec.i32x4 = max_idx_lo_i32x4;
    idx_hi_vec.i32x4 = max_idx_hi_i32x4;

    nk_f32_t best_val = max_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_lo_vec.f32s[i] > best_val) {
            best_val = max_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (max_hi_vec.f32s[i] > best_val) {
            best_val = max_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e4m3_to_f32(&data[idx], &val);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_e4m3_neonfhm_strided_(                  //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    float32x4_t max_lo_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    float32x4_t max_hi_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t max_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_e4m3_t const *ptr = data;
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        nk_u8_t buf[8];
        for (nk_size_t i = 0; i < 8; ++i) {
            buf[i] = *ptr;
            ptr += stride_elements;
        }
        uint8x8_t data_e4m3x8 = vld1_u8(buf);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neonfhm_(data_e4m3x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        uint32x4_t gt_lo_u32x4 = vcgtq_f32(lo_f32x4, max_lo_f32x4);
        uint32x4_t gt_hi_u32x4 = vcgtq_f32(hi_f32x4, max_hi_f32x4);
        max_lo_f32x4 = vbslq_f32(gt_lo_u32x4, lo_f32x4, max_lo_f32x4);
        max_hi_f32x4 = vbslq_f32(gt_hi_u32x4, hi_f32x4, max_hi_f32x4);
        max_idx_lo_i32x4 = vbslq_s32(gt_lo_u32x4, idx_lo_i32x4, max_idx_lo_i32x4);
        max_idx_hi_i32x4 = vbslq_s32(gt_hi_u32x4, idx_hi_i32x4, max_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    // Horizontal reduction
    nk_b128_vec_t max_lo_vec, max_hi_vec, idx_lo_vec, idx_hi_vec;
    max_lo_vec.f32x4 = max_lo_f32x4;
    max_hi_vec.f32x4 = max_hi_f32x4;
    idx_lo_vec.i32x4 = max_idx_lo_i32x4;
    idx_hi_vec.i32x4 = max_idx_hi_i32x4;

    nk_f32_t best_val = max_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_lo_vec.f32s[i] > best_val) {
            best_val = max_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (max_hi_vec.f32s[i] > best_val) {
            best_val = max_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f32_t val;
        nk_e4m3_to_f32(ptr, &val);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_e4m3_neonfhm(                          //
    nk_e4m3_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);
    if (!aligned) nk_reduce_max_e4m3_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e4m3_neonfhm_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e4m3_neonfhm_strided_(data, count, stride_elements, max_value, max_index);
}

NK_INTERNAL void nk_reduce_max_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    float32x4_t max_lo_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    float32x4_t max_hi_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t max_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e5m2x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neonfhm_(data_e5m2x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        uint32x4_t gt_lo_u32x4 = vcgtq_f32(lo_f32x4, max_lo_f32x4);
        uint32x4_t gt_hi_u32x4 = vcgtq_f32(hi_f32x4, max_hi_f32x4);
        max_lo_f32x4 = vbslq_f32(gt_lo_u32x4, lo_f32x4, max_lo_f32x4);
        max_hi_f32x4 = vbslq_f32(gt_hi_u32x4, hi_f32x4, max_hi_f32x4);
        max_idx_lo_i32x4 = vbslq_s32(gt_lo_u32x4, idx_lo_i32x4, max_idx_lo_i32x4);
        max_idx_hi_i32x4 = vbslq_s32(gt_hi_u32x4, idx_hi_i32x4, max_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    nk_b128_vec_t max_lo_vec, max_hi_vec, idx_lo_vec, idx_hi_vec;
    max_lo_vec.f32x4 = max_lo_f32x4;
    max_hi_vec.f32x4 = max_hi_f32x4;
    idx_lo_vec.i32x4 = max_idx_lo_i32x4;
    idx_hi_vec.i32x4 = max_idx_hi_i32x4;

    nk_f32_t best_val = max_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_lo_vec.f32s[i] > best_val) {
            best_val = max_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (max_hi_vec.f32s[i] > best_val) {
            best_val = max_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_e5m2_to_f32(&data[idx], &val);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_e5m2_neonfhm_strided_(                  //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    float32x4_t max_lo_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    float32x4_t max_hi_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_lo_i32x4 = vdupq_n_s32(0);
    int32x4_t max_idx_hi_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_lo_i32x4 = {0, 1, 2, 3};
    int32x4_t idx_hi_i32x4 = {4, 5, 6, 7};
    int32x4_t step_i32x4 = vdupq_n_s32(8);
    nk_e5m2_t const *ptr = data;
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        nk_u8_t buf[8];
        for (nk_size_t i = 0; i < 8; ++i) {
            buf[i] = *ptr;
            ptr += stride_elements;
        }
        uint8x8_t data_e5m2x8 = vld1_u8(buf);
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neonfhm_(data_e5m2x8);
        float32x4_t lo_f32x4 = vcvt_f32_f16(vget_low_f16(data_f16x8));
        float32x4_t hi_f32x4 = vcvt_f32_f16(vget_high_f16(data_f16x8));

        uint32x4_t gt_lo_u32x4 = vcgtq_f32(lo_f32x4, max_lo_f32x4);
        uint32x4_t gt_hi_u32x4 = vcgtq_f32(hi_f32x4, max_hi_f32x4);
        max_lo_f32x4 = vbslq_f32(gt_lo_u32x4, lo_f32x4, max_lo_f32x4);
        max_hi_f32x4 = vbslq_f32(gt_hi_u32x4, hi_f32x4, max_hi_f32x4);
        max_idx_lo_i32x4 = vbslq_s32(gt_lo_u32x4, idx_lo_i32x4, max_idx_lo_i32x4);
        max_idx_hi_i32x4 = vbslq_s32(gt_hi_u32x4, idx_hi_i32x4, max_idx_hi_i32x4);
        idx_lo_i32x4 = vaddq_s32(idx_lo_i32x4, step_i32x4);
        idx_hi_i32x4 = vaddq_s32(idx_hi_i32x4, step_i32x4);
    }

    nk_b128_vec_t max_lo_vec, max_hi_vec, idx_lo_vec, idx_hi_vec;
    max_lo_vec.f32x4 = max_lo_f32x4;
    max_hi_vec.f32x4 = max_hi_f32x4;
    idx_lo_vec.i32x4 = max_idx_lo_i32x4;
    idx_hi_vec.i32x4 = max_idx_hi_i32x4;

    nk_f32_t best_val = max_lo_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_lo_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_lo_vec.f32s[i] > best_val) {
            best_val = max_lo_vec.f32s[i];
            best_idx = (nk_size_t)idx_lo_vec.i32s[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (max_hi_vec.f32s[i] > best_val) {
            best_val = max_hi_vec.f32s[i];
            best_idx = (nk_size_t)idx_hi_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx, ptr += stride_elements) {
        nk_f32_t val;
        nk_e5m2_to_f32(ptr, &val);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_e5m2_neonfhm(                          //
    nk_e5m2_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);
    if (!aligned) nk_reduce_max_e5m2_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e5m2_neonfhm_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e5m2_neonfhm_strided_(data, count, stride_elements, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEONFHM_H
