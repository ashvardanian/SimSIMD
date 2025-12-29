/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm NEON-capable CPUs.
 *  @file include/numkong/reduce/neonbfdot.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_NEONBFDOT_H
#define NK_REDUCE_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#include "numkong/reduce/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL void nk_reduce_add_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data, nk_size_t count, nk_f32_t *result) {
    // Use vbfdotq_f32 with ones vector: dot(data, ones) = sum(data)
    // bf16 representation of 1.0 is 0x3F80 (same as upper 16 bits of f32 1.0)
    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    for (; idx + 8 <= count; idx += 8) {
        bfloat16x8_t data_bf16x8 = vld1q_bf16((nk_bf16_for_arm_simd_t const *)(data + idx));
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
    }

    // Handle tail with type-agnostic partial load
    if (idx < count) {
        nk_b128_vec_t tail_vec;
        nk_partial_load_b16x8_neon_(data + idx, count - idx, &tail_vec);
        bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(tail_vec.u16x8);
        sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
    }

    *result = vaddvq_f32(sum_f32x4);
}

NK_INTERNAL void nk_reduce_add_bf16_neonbfdot_strided_(                //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *result) {
    // For strided bf16, use vld2/vld3/vld4 to de-interleave, then dot with ones
    bfloat16x8_t ones_bf16x8 = vreinterpretq_bf16_u16(vdupq_n_u16(0x3F80));
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 8 <= count; idx += 8) {
            // vld2 loads 16 bf16 values and de-interleaves into two bfloat16x8_t
            // We need to load as u16 and reinterpret since vld2q_bf16 may not exist
            uint16x8x2_t loaded_u16x8x2 = vld2q_u16((uint16_t const *)(data + idx * 2));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x2.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x3_t loaded_u16x8x3 = vld3q_u16((uint16_t const *)(data + idx * 3));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x3.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 8 <= count; idx += 8) {
            uint16x8x4_t loaded_u16x8x4 = vld4q_u16((uint16_t const *)(data + idx * 4));
            bfloat16x8_t data_bf16x8 = vreinterpretq_bf16_u16(loaded_u16x8x4.val[0]);
            sum_f32x4 = vbfdotq_f32(sum_f32x4, data_bf16x8, ones_bf16x8);
        }
    }

    // Scalar tail
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_bf16_to_f32_(&data[idx * stride_elements], &val);
        sum += val;
    }

    *result = sum;
}

NK_PUBLIC void nk_reduce_add_bf16_neonbfdot(                        //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (!aligned) nk_reduce_add_bf16_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) nk_reduce_add_bf16_neonbfdot_contiguous_(data, count, result);
    else if (stride_elements <= 4) nk_reduce_add_bf16_neonbfdot_strided_(data, count, stride_elements, result);
    else nk_reduce_add_bf16_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void nk_reduce_min_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    // Track min values in f32 (converted from bf16), process 4 at a time
    float32x4_t min_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        // Load 4 bf16 values and convert to f32 via bit shift
        uint16x4_t data_u16x4 = vld1_u16((uint16_t const *)(data + idx));
        uint32x4_t data_u32x4 = vshll_n_u16(data_u16x4, 16);
        float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
        uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
        min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
        min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    // Horizontal reduction
    nk_b128_vec_t min_vec, idx_vec;
    min_vec.f32x4 = min_f32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_f32_t best_val = min_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.f32s[i] < best_val) {
            best_val = min_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_bf16_to_f32_(&data[idx], &val);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_INTERNAL void nk_reduce_min_bf16_neonbfdot_strided_(                //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }

    float32x4_t min_f32x4 = vdupq_n_f32(__builtin_huge_valf());
    int32x4_t min_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint16x4x2_t loaded_u16x4x2 = vld2_u16((uint16_t const *)(data + idx * 2));
            uint32x4_t data_u32x4 = vshll_n_u16(loaded_u16x4x2.val[0], 16);
            float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
            uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint16x4x3_t loaded_u16x4x3 = vld3_u16((uint16_t const *)(data + idx * 3));
            uint32x4_t data_u32x4 = vshll_n_u16(loaded_u16x4x3.val[0], 16);
            float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
            uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            uint16x4x4_t loaded_u16x4x4 = vld4_u16((uint16_t const *)(data + idx * 4));
            uint32x4_t data_u32x4 = vshll_n_u16(loaded_u16x4x4.val[0], 16);
            float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
            uint32x4_t lt_u32x4 = vcltq_f32(data_f32x4, min_f32x4);
            min_f32x4 = vbslq_f32(lt_u32x4, data_f32x4, min_f32x4);
            min_idx_i32x4 = vbslq_s32(lt_u32x4, idx_i32x4, min_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    // Horizontal reduction
    nk_b128_vec_t min_vec, idx_vec;
    min_vec.f32x4 = min_f32x4;
    idx_vec.i32x4 = min_idx_i32x4;

    nk_f32_t best_val = min_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (min_vec.f32s[i] < best_val) {
            best_val = min_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_bf16_to_f32_(&data[idx * stride_elements], &val);
        if (val < best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *min_value = best_val;
    *min_index = best_idx;
}

NK_PUBLIC void nk_reduce_min_bf16_neonbfdot(                        //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (!aligned) nk_reduce_min_bf16_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1) nk_reduce_min_bf16_neonbfdot_contiguous_(data, count, min_value, min_index);
    else if (stride_elements <= 4)
        nk_reduce_min_bf16_neonbfdot_strided_(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_bf16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void nk_reduce_max_bf16_neonbfdot_contiguous_( //
    nk_bf16_t const *data, nk_size_t count,                //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    // Track max values in f32 (converted from bf16), process 4 at a time
    float32x4_t max_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    for (; idx + 4 <= count; idx += 4) {
        // Load 4 bf16 values and convert to f32 via bit shift
        uint16x4_t data_u16x4 = vld1_u16((uint16_t const *)(data + idx));
        uint32x4_t data_u32x4 = vshll_n_u16(data_u16x4, 16);
        float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
        uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
        max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
        max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
        idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
    }

    // Horizontal reduction
    nk_b128_vec_t max_vec, idx_vec;
    max_vec.f32x4 = max_f32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_f32_t best_val = max_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.f32s[i] > best_val) {
            best_val = max_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_bf16_to_f32_(&data[idx], &val);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_INTERNAL void nk_reduce_max_bf16_neonbfdot_strided_(                //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }

    float32x4_t max_f32x4 = vdupq_n_f32(-__builtin_huge_valf());
    int32x4_t max_idx_i32x4 = vdupq_n_s32(0);
    int32x4_t idx_i32x4 = {0, 1, 2, 3};
    int32x4_t step_i32x4 = vdupq_n_s32(4);
    nk_size_t idx = 0;

    if (stride_elements == 2) {
        for (; idx + 4 <= count; idx += 4) {
            uint16x4x2_t loaded_u16x4x2 = vld2_u16((uint16_t const *)(data + idx * 2));
            uint32x4_t data_u32x4 = vshll_n_u16(loaded_u16x4x2.val[0], 16);
            float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
            uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 3) {
        for (; idx + 4 <= count; idx += 4) {
            uint16x4x3_t loaded_u16x4x3 = vld3_u16((uint16_t const *)(data + idx * 3));
            uint32x4_t data_u32x4 = vshll_n_u16(loaded_u16x4x3.val[0], 16);
            float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
            uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }
    else if (stride_elements == 4) {
        for (; idx + 4 <= count; idx += 4) {
            uint16x4x4_t loaded_u16x4x4 = vld4_u16((uint16_t const *)(data + idx * 4));
            uint32x4_t data_u32x4 = vshll_n_u16(loaded_u16x4x4.val[0], 16);
            float32x4_t data_f32x4 = vreinterpretq_f32_u32(data_u32x4);
            uint32x4_t gt_u32x4 = vcgtq_f32(data_f32x4, max_f32x4);
            max_f32x4 = vbslq_f32(gt_u32x4, data_f32x4, max_f32x4);
            max_idx_i32x4 = vbslq_s32(gt_u32x4, idx_i32x4, max_idx_i32x4);
            idx_i32x4 = vaddq_s32(idx_i32x4, step_i32x4);
        }
    }

    // Horizontal reduction
    nk_b128_vec_t max_vec, idx_vec;
    max_vec.f32x4 = max_f32x4;
    idx_vec.i32x4 = max_idx_i32x4;

    nk_f32_t best_val = max_vec.f32s[0];
    nk_size_t best_idx = (nk_size_t)idx_vec.i32s[0];
    for (int i = 1; i < 4; ++i) {
        if (max_vec.f32s[i] > best_val) {
            best_val = max_vec.f32s[i];
            best_idx = (nk_size_t)idx_vec.i32s[i];
        }
    }

    // Scalar tail
    for (; idx < count; ++idx) {
        nk_f32_t val;
        nk_bf16_to_f32_(&data[idx * stride_elements], &val);
        if (val > best_val) {
            best_val = val;
            best_idx = idx;
        }
    }

    *max_value = best_val;
    *max_index = best_idx;
}

NK_PUBLIC void nk_reduce_max_bf16_neonbfdot(                        //
    nk_bf16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);
    if (!aligned) nk_reduce_max_bf16_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1) nk_reduce_max_bf16_neonbfdot_contiguous_(data, count, max_value, max_index);
    else if (stride_elements <= 4)
        nk_reduce_max_bf16_neonbfdot_strided_(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_bf16_serial(data, count, stride_bytes, max_value, max_index);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_NEONBFDOT_H
