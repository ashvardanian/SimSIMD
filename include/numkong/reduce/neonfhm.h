/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON with FHM (FEAT_FP16FML).
 *  @file include/numkong/reduce/neonfhm.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 29, 2025
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
 *      vbslq_s32                   BSL (V.16B, V.16B, V.16B)       2cy         2/cy        4/cy
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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

#include "numkong/types.h"
#include "numkong/reduce/serial.h"
#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL void nk_reduce_add_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count, nk_f32_t *result) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    // Main loop: process 8 elements at a time
    for (; idx + 8 <= count; idx += 8) {
        uint8x8_t data_e4m3x8 = vld1_u8((uint8_t const *)data + idx);
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
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
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
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
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
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
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
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
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
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
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
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
    if (count == 0 || !aligned) nk_reduce_min_e4m3_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e4m3_neonfhm_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e4m3_neonfhm_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_min_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {

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
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
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
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
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
    if (count == 0 || !aligned) nk_reduce_min_e5m2_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_e5m2_neonfhm_contiguous_(data, count, min_value, min_index);
    else nk_reduce_min_e5m2_neonfhm_strided_(data, count, stride_elements, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_e4m3_neonfhm_contiguous_( //
    nk_e4m3_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

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
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
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
        float16x8_t data_f16x8 = nk_e4m3x8_to_f16x8_neon_(data_e4m3x8);
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
    if (count == 0 || !aligned) nk_reduce_max_e4m3_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e4m3_neonfhm_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e4m3_neonfhm_strided_(data, count, stride_elements, max_value, max_index);
}

NK_INTERNAL void nk_reduce_max_e5m2_neonfhm_contiguous_( //
    nk_e5m2_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {

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
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
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
        float16x8_t data_f16x8 = nk_e5m2x8_to_f16x8_neon_(data_e5m2x8);
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
    if (count == 0 || !aligned) nk_reduce_max_e5m2_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_e5m2_neonfhm_contiguous_(data, count, max_value, max_index);
    else nk_reduce_max_e5m2_neonfhm_strided_(data, count, stride_elements, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEONFHM_H
