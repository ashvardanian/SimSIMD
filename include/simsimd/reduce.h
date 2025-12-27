/**
 *  @brief SIMD-accelerated Horizontal Reduction Operations.
 *  @file include/simsimd/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2024
 *
 *  Provides horizontal reduction operations (sum, min, max) over vectors with:
 *  - Internal helpers for single-register reductions (native precision)
 *  - Public interfaces with stride support and precision widening for sums
 *  - Dynamic dispatch for runtime ISA selection
 *
 *  @section reduction_strategy Reduction Strategy
 *
 *  The key insight is that `_mm512_reduce_add_ps()` and similar intrinsics are
 *  actually SERIAL operations - they don't parallelize the reduction across lanes.
 *  The correct approach is:
 *
 *  1. Accumulate VERTICALLY in SIMD registers throughout the entire loop
 *  2. Perform ONE horizontal reduction at the very end
 *
 *  @code{.c}
 *  __m512 sum_f32x16 = _mm512_setzero_ps();
 *  for (...) {
 *      __m512 data_f32x16 = _mm512_loadu_ps(ptr);
 *      sum_f32x16 = _mm512_add_ps(sum_f32x16, data_f32x16);
 *  }
 *  // Single horizontal reduce at the END only
 *  simsimd_f32_t result = _simsimd_reduce_add_f32x16_skylake(sum_f32x16);
 *  @endcode
 *
 *  @section stride_handling Stride Handling Strategies
 *
 *  - stride == sizeof(scalar): Contiguous SIMD loads with masked tail
 *  - Large stride with gather support: Use gather instructions (32/64-bit types)
 *  - Otherwise: Serial fallback
 *
 *  @section argminmax Argmin/Argmax Strategy
 *
 *  Single-pass algorithm tracking both value and index in SIMD registers:
 *  @code{.c}
 *  __m512 min_f32x16 = _mm512_set1_ps(FLT_MAX);
 *  __m512i min_idx_i32x16 = _mm512_setzero_si512();
 *  __m512i current_idx_i32x16 = _mm512_setr_epi32(0,1,2,3,...,15);
 *  __m512i step_i32x16 = _mm512_set1_epi32(16);
 *  for (...) {
 *      __m512 data_f32x16 = _mm512_loadu_ps(ptr);
 *      __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
 *      min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
 *      min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
 *      current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
 *  }
 *  @endcode
 */
#ifndef SIMSIMD_REDUCE_H
#define SIMSIMD_REDUCE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region SIMSIMD_DYNAMIC Function Declarations

/**
 *  @brief  Horizontal sum reduction over a strided array of f32 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f32) for contiguous).
 *  @param[out] result      Output sum, widened to f64 for precision.
 */
SIMSIMD_DYNAMIC void simsimd_reduce_add_f32(simsimd_f32_t const *data, simsimd_size_t count,
                                            simsimd_size_t stride_bytes, simsimd_f64_t *result);

/**
 *  @brief  Horizontal sum reduction over a strided array of f64 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f64) for contiguous).
 *  @param[out] result      Output sum.
 */
SIMSIMD_DYNAMIC void simsimd_reduce_add_f64(simsimd_f64_t const *data, simsimd_size_t count,
                                            simsimd_size_t stride_bytes, simsimd_f64_t *result);

/**
 *  @brief  Horizontal minimum reduction with argmin over a strided array of f32 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f32) for contiguous).
 *  @param[out] min_value   Output minimum value.
 *  @param[out] min_index   Output index of the minimum value.
 */
SIMSIMD_DYNAMIC void simsimd_reduce_min_f32(simsimd_f32_t const *data, simsimd_size_t count,
                                            simsimd_size_t stride_bytes, simsimd_f32_t *min_value,
                                            simsimd_size_t *min_index);

/**
 *  @brief  Horizontal maximum reduction with argmax over a strided array of f32 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f32) for contiguous).
 *  @param[out] max_value   Output maximum value.
 *  @param[out] max_index   Output index of the maximum value.
 */
SIMSIMD_DYNAMIC void simsimd_reduce_max_f32(simsimd_f32_t const *data, simsimd_size_t count,
                                            simsimd_size_t stride_bytes, simsimd_f32_t *max_value,
                                            simsimd_size_t *max_index);

/**
 *  @brief  Horizontal minimum reduction with argmin over a strided array of f64 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f64) for contiguous).
 *  @param[out] min_value   Output minimum value.
 *  @param[out] min_index   Output index of the minimum value.
 */
SIMSIMD_DYNAMIC void simsimd_reduce_min_f64(simsimd_f64_t const *data, simsimd_size_t count,
                                            simsimd_size_t stride_bytes, simsimd_f64_t *min_value,
                                            simsimd_size_t *min_index);

/**
 *  @brief  Horizontal maximum reduction with argmax over a strided array of f64 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f64) for contiguous).
 *  @param[out] max_value   Output maximum value.
 *  @param[out] max_index   Output index of the maximum value.
 */
SIMSIMD_DYNAMIC void simsimd_reduce_max_f64(simsimd_f64_t const *data, simsimd_size_t count,
                                            simsimd_size_t stride_bytes, simsimd_f64_t *max_value,
                                            simsimd_size_t *max_index);

#pragma endregion

#pragma region Forward Declarations - Serial

/** @copydoc simsimd_reduce_add_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f32_serial(simsimd_f32_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_add_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f64_serial(simsimd_f64_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_add_f32 but for i32 input with i64 output */
SIMSIMD_PUBLIC void simsimd_reduce_add_i32_serial(simsimd_i32_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_i64_t *result);
/** @copydoc simsimd_reduce_min_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f32_serial(simsimd_f32_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_f32_t *min_value,
                                                  simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f32_serial(simsimd_f32_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_f32_t *max_value,
                                                  simsimd_size_t *max_index);
/** @copydoc simsimd_reduce_min_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f64_serial(simsimd_f64_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_f64_t *min_value,
                                                  simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f64_serial(simsimd_f64_t const *data, simsimd_size_t count,
                                                  simsimd_size_t stride_bytes, simsimd_f64_t *max_value,
                                                  simsimd_size_t *max_index);

#pragma endregion

#pragma region Forward Declarations - ARM NEON

#if SIMSIMD_TARGET_NEON
/** @copydoc simsimd_reduce_add_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f32_neon(simsimd_f32_t const *data, simsimd_size_t count,
                                                simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_add_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f64_neon(simsimd_f64_t const *data, simsimd_size_t count,
                                                simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_min_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f32_neon(simsimd_f32_t const *data, simsimd_size_t count,
                                                simsimd_size_t stride_bytes, simsimd_f32_t *min_value,
                                                simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f32_neon(simsimd_f32_t const *data, simsimd_size_t count,
                                                simsimd_size_t stride_bytes, simsimd_f32_t *max_value,
                                                simsimd_size_t *max_index);
#endif // SIMSIMD_TARGET_NEON

#pragma endregion

#pragma region Forward Declarations - x86 Haswell

#if SIMSIMD_TARGET_HASWELL
/** @copydoc simsimd_reduce_add_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f32_haswell(simsimd_f32_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_add_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f64_haswell(simsimd_f64_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_min_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f32_haswell(simsimd_f32_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f32_t *min_value,
                                                   simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f32_haswell(simsimd_f32_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f32_t *max_value,
                                                   simsimd_size_t *max_index);
/** @copydoc simsimd_reduce_min_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f64_haswell(simsimd_f64_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *min_value,
                                                   simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f64_haswell(simsimd_f64_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *max_value,
                                                   simsimd_size_t *max_index);
#endif // SIMSIMD_TARGET_HASWELL

#pragma endregion

#pragma region Forward Declarations - x86 Skylake

#if SIMSIMD_TARGET_SKYLAKE
/** @copydoc simsimd_reduce_add_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f32_skylake(simsimd_f32_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_add_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_add_f64_skylake(simsimd_f64_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *result);
/** @copydoc simsimd_reduce_min_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f32_skylake(simsimd_f32_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f32_t *min_value,
                                                   simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f32 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f32_skylake(simsimd_f32_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f32_t *max_value,
                                                   simsimd_size_t *max_index);
/** @copydoc simsimd_reduce_min_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_min_f64_skylake(simsimd_f64_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *min_value,
                                                   simsimd_size_t *min_index);
/** @copydoc simsimd_reduce_max_f64 */
SIMSIMD_PUBLIC void simsimd_reduce_max_f64_skylake(simsimd_f64_t const *data, simsimd_size_t count,
                                                   simsimd_size_t stride_bytes, simsimd_f64_t *max_value,
                                                   simsimd_size_t *max_index);
#endif // SIMSIMD_TARGET_SKYLAKE

#pragma endregion

#pragma region Serial Implementations

SIMSIMD_PUBLIC void simsimd_reduce_add_f32_serial(                                //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    simsimd_f64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (simsimd_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(simsimd_f32_t const *)ptr;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f64_serial(                                //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    simsimd_f64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (simsimd_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(simsimd_f64_t const *)ptr;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_i32_serial(                                //
    simsimd_i32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_i64_t *result) {
    simsimd_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (simsimd_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(simsimd_i32_t const *)ptr;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f32_serial(                                //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    simsimd_f32_t best_value = *(simsimd_f32_t const *)ptr;
    simsimd_size_t best_index = 0;
    ptr += stride_bytes;
    for (simsimd_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        simsimd_f32_t val = *(simsimd_f32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f32_serial(                                //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    simsimd_f32_t best_value = *(simsimd_f32_t const *)ptr;
    simsimd_size_t best_index = 0;
    ptr += stride_bytes;
    for (simsimd_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        simsimd_f32_t val = *(simsimd_f32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f64_serial(                                //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *min_value, simsimd_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    simsimd_f64_t best_value = *(simsimd_f64_t const *)ptr;
    simsimd_size_t best_index = 0;
    ptr += stride_bytes;
    for (simsimd_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        simsimd_f64_t val = *(simsimd_f64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f64_serial(                                //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *max_value, simsimd_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    simsimd_f64_t best_value = *(simsimd_f64_t const *)ptr;
    simsimd_size_t best_index = 0;
    ptr += stride_bytes;
    for (simsimd_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        simsimd_f64_t val = *(simsimd_f64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

#pragma endregion

#pragma region ARM NEON Implementations

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

#pragma region ARM NEON Internal Helpers

/** @brief Horizontal sum of 4 floats in a NEON register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_add_f32x4_neon(float32x4_t sum_f32x4) { return vaddvq_f32(sum_f32x4); }

/** @brief Horizontal sum of 2 doubles in a NEON register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_add_f64x2_neon(float64x2_t sum_f64x2) { return vaddvq_f64(sum_f64x2); }

/** @brief Horizontal min of 4 floats in a NEON register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_min_f32x4_neon(float32x4_t min_f32x4) { return vminvq_f32(min_f32x4); }

/** @brief Horizontal max of 4 floats in a NEON register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_max_f32x4_neon(float32x4_t max_f32x4) { return vmaxvq_f32(max_f32x4); }

/** @brief Horizontal sum of 4 i32s in a NEON register. */
SIMSIMD_INTERNAL simsimd_i32_t _simsimd_reduce_add_i32x4_neon(int32x4_t sum_i32x4) { return vaddvq_s32(sum_i32x4); }

/** @brief Horizontal min of 4 i32s in a NEON register. */
SIMSIMD_INTERNAL simsimd_i32_t _simsimd_reduce_min_i32x4_neon(int32x4_t min_i32x4) { return vminvq_s32(min_i32x4); }

/** @brief Horizontal max of 4 i32s in a NEON register. */
SIMSIMD_INTERNAL simsimd_i32_t _simsimd_reduce_max_i32x4_neon(int32x4_t max_i32x4) { return vmaxvq_s32(max_i32x4); }

/** @brief Horizontal sum of 16 u8s in a NEON register, returning u32. */
SIMSIMD_INTERNAL simsimd_u32_t _simsimd_reduce_add_u8x16_neon(uint8x16_t sum_u8x16) {
    uint16x8_t low_u16x8 = vmovl_u8(vget_low_u8(sum_u8x16));
    uint16x8_t high_u16x8 = vmovl_u8(vget_high_u8(sum_u8x16));
    uint16x8_t sum_u16x8 = vaddq_u16(low_u16x8, high_u16x8);
    uint32x4_t sum_u32x4 = vpaddlq_u16(sum_u16x8);
    uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
    return (simsimd_u32_t)vaddvq_u64(sum_u64x2);
}

#pragma endregion // ARM NEON Internal Helpers

#pragma region ARM NEON Public Implementations

SIMSIMD_INTERNAL void _simsimd_reduce_add_f32_neon_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_f64_t *result) {
    // Accumulate in f64 for precision
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        float64x2_t lo_f64x2 = vcvt_f64_f32(vget_low_f32(data_f32x4));
        float64x2_t hi_f64x2 = vcvt_f64_f32(vget_high_f32(data_f32x4));
        sum_f64x2 = vaddq_f64(sum_f64x2, lo_f64x2);
        sum_f64x2 = vaddq_f64(sum_f64x2, hi_f64x2);
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x2_neon(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f32_neon(                                  //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    if (stride_bytes == sizeof(simsimd_f32_t)) return _simsimd_reduce_add_f32_neon_contiguous(data, count, result);
    simsimd_reduce_add_f32_serial(data, count, stride_bytes, result);
}

SIMSIMD_INTERNAL void _simsimd_reduce_add_f64_neon_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count; idx_scalars += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data + idx_scalars);
        sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2);
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x2_neon(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f64_neon(                                  //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    if (stride_bytes == sizeof(simsimd_f64_t)) return _simsimd_reduce_add_f64_neon_contiguous(data, count, result);
    simsimd_reduce_add_f64_serial(data, count, stride_bytes, result);
}

SIMSIMD_INTERNAL void _simsimd_reduce_min_f32_neon_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count,           //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    // First pass: find minimum value using SIMD
    float32x4_t min_f32x4 = vld1q_f32(data);
    simsimd_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        min_f32x4 = vminq_f32(min_f32x4, data_f32x4);
    }
    simsimd_f32_t min_val = _simsimd_reduce_min_f32x4_neon(min_f32x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] < min_val) min_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != min_val) continue;
        *min_value = min_val;
        *min_index = idx_scalars;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f32_neon(                                  //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    if (stride_bytes == sizeof(simsimd_f32_t) && count >= 4)
        return _simsimd_reduce_min_f32_neon_contiguous(data, count, min_value, min_index);
    simsimd_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_max_f32_neon_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count,           //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    float32x4_t max_f32x4 = vld1q_f32(data);
    simsimd_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        max_f32x4 = vmaxq_f32(max_f32x4, data_f32x4);
    }
    simsimd_f32_t max_val = _simsimd_reduce_max_f32x4_neon(max_f32x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] > max_val) max_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != max_val) continue;
        *max_value = max_val;
        *max_index = idx_scalars;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f32_neon(                                  //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    if (stride_bytes == sizeof(simsimd_f32_t) && count >= 4)
        return _simsimd_reduce_max_f32_neon_contiguous(data, count, max_value, max_index);
    simsimd_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion // ARM NEON Public Implementations

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

/** @brief Horizontal sum of 8 f16s in a NEON register, returning f32. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_add_f16x8_neon(float16x8_t sum_f16x8) {
    float16x4_t low_f16x4 = vget_low_f16(sum_f16x8);
    float16x4_t high_f16x4 = vget_high_f16(sum_f16x8);
    float16x4_t sum_f16x4 = vadd_f16(low_f16x4, high_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    return vgetq_lane_f32(vcvt_f32_f16(sum_f16x4), 0);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16
#endif // _SIMSIMD_TARGET_ARM

#pragma endregion // ARM NEON Implementations

#pragma region x86 Haswell Implementations

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)

#pragma region x86 Haswell Internal Helpers

/** @brief Horizontal sum of 4 doubles in a YMM register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_add_f64x4_haswell(__m256d sum_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(sum_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(sum_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(lo_f64x2, hi_f64x2);
    sum_f64x2 = _mm_hadd_pd(sum_f64x2, sum_f64x2);
    return _mm_cvtsd_f64(sum_f64x2);
}

/** @brief Horizontal sum of 8 floats in a YMM register (native f32 precision). */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_add_f32x8_haswell(__m256 sum_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(sum_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(sum_f32x8, 1);
    __m128 sum_f32x4 = _mm_add_ps(lo_f32x4, hi_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    return _mm_cvtss_f32(sum_f32x4);
}

/** @brief Horizontal sum of 8 i32s in a YMM register. */
SIMSIMD_INTERNAL simsimd_i32_t _simsimd_reduce_add_i32x8_haswell(__m256i sum_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(sum_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(sum_i32x8, 1);
    __m128i sum_i32x4 = _mm_add_epi32(lo_i32x4, hi_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    return _mm_cvtsi128_si32(sum_i32x4);
}

/** @brief Horizontal min of 8 floats in a YMM register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_min_f32x8_haswell(__m256 min_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(min_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(min_f32x8, 1);
    __m128 min_f32x4 = _mm_min_ps(lo_f32x4, hi_f32x4);
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(min_f32x4);
}

/** @brief Horizontal max of 8 floats in a YMM register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_max_f32x8_haswell(__m256 max_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(max_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(max_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(lo_f32x4, hi_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max_f32x4);
}

/** @brief Horizontal min of 4 doubles in a YMM register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_min_f64x4_haswell(__m256d min_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(min_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(min_f64x4, 1);
    __m128d min_f64x2 = _mm_min_pd(lo_f64x2, hi_f64x2);
    min_f64x2 = _mm_min_pd(min_f64x2, _mm_shuffle_pd(min_f64x2, min_f64x2, 1));
    return _mm_cvtsd_f64(min_f64x2);
}

/** @brief Horizontal max of 4 doubles in a YMM register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_max_f64x4_haswell(__m256d max_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(max_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(max_f64x4, 1);
    __m128d max_f64x2 = _mm_max_pd(lo_f64x2, hi_f64x2);
    max_f64x2 = _mm_max_pd(max_f64x2, _mm_shuffle_pd(max_f64x2, max_f64x2, 1));
    return _mm_cvtsd_f64(max_f64x2);
}

#pragma endregion // x86 Haswell Internal Helpers

#pragma region x86 Haswell Public Implementations

SIMSIMD_INTERNAL void _simsimd_reduce_add_f32_haswell_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_f64_t *result) {
    // Accumulate in f64 for precision
    __m256d sum_f64x4 = _mm256_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        __m128 lo_f32x4 = _mm256_castps256_ps128(data_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(data_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x4_haswell(sum_f64x4);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

SIMSIMD_INTERNAL void _simsimd_reduce_add_f32_haswell_gather(                     //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    simsimd_i32_t stride_elements = (simsimd_i32_t)(stride_bytes / sizeof(simsimd_f32_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m256d sum_f64x4 = _mm256_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 gathered_f32x8 = _mm256_i32gather_ps(data + idx_scalars * stride_elements, indices_i32x8,
                                                    sizeof(simsimd_f32_t));
        __m128 lo_f32x4 = _mm256_castps256_ps128(gathered_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(gathered_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x4_haswell(sum_f64x4);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(simsimd_f32_t const *)ptr;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f32_haswell(                               //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    if (stride_bytes == sizeof(simsimd_f32_t)) return _simsimd_reduce_add_f32_haswell_contiguous(data, count, result);
    if (stride_bytes % sizeof(simsimd_f32_t) == 0 && stride_bytes <= 8 * sizeof(simsimd_f32_t))
        return _simsimd_reduce_add_f32_haswell_gather(data, count, stride_bytes, result);
    simsimd_reduce_add_f32_serial(data, count, stride_bytes, result);
}

SIMSIMD_INTERNAL void _simsimd_reduce_add_f64_haswell_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_f64_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, data_f64x4);
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x4_haswell(sum_f64x4);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f64_haswell(                               //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    if (stride_bytes == sizeof(simsimd_f64_t)) return _simsimd_reduce_add_f64_haswell_contiguous(data, count, result);
    simsimd_reduce_add_f64_serial(data, count, stride_bytes, result);
}

SIMSIMD_INTERNAL void _simsimd_reduce_min_f32_haswell_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count,              //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    // First pass: find minimum value
    __m256 min_f32x8 = _mm256_loadu_ps(data);
    simsimd_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    simsimd_f32_t min_val = _simsimd_reduce_min_f32x8_haswell(min_f32x8);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] < min_val) min_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != min_val) continue;
        *min_value = min_val;
        *min_index = idx_scalars;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f32_haswell(                               //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    if (stride_bytes == sizeof(simsimd_f32_t) && count >= 8)
        return _simsimd_reduce_min_f32_haswell_contiguous(data, count, min_value, min_index);
    simsimd_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_max_f32_haswell_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count,              //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    __m256 max_f32x8 = _mm256_loadu_ps(data);
    simsimd_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    simsimd_f32_t max_val = _simsimd_reduce_max_f32x8_haswell(max_f32x8);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] > max_val) max_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != max_val) continue;
        *max_value = max_val;
        *max_index = idx_scalars;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f32_haswell(                               //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    if (stride_bytes == sizeof(simsimd_f32_t) && count >= 8)
        return _simsimd_reduce_max_f32_haswell_contiguous(data, count, max_value, max_index);
    simsimd_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_min_f64_haswell_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count,              //
    simsimd_f64_t *min_value, simsimd_size_t *min_index) {
    __m256d min_f64x4 = _mm256_loadu_pd(data);
    simsimd_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        min_f64x4 = _mm256_min_pd(min_f64x4, data_f64x4);
    }
    simsimd_f64_t min_val = _simsimd_reduce_min_f64x4_haswell(min_f64x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] < min_val) min_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != min_val) continue;
        *min_value = min_val;
        *min_index = idx_scalars;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f64_haswell(                               //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *min_value, simsimd_size_t *min_index) {
    if (stride_bytes == sizeof(simsimd_f64_t) && count >= 4)
        return _simsimd_reduce_min_f64_haswell_contiguous(data, count, min_value, min_index);
    simsimd_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_max_f64_haswell_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count,              //
    simsimd_f64_t *max_value, simsimd_size_t *max_index) {
    __m256d max_f64x4 = _mm256_loadu_pd(data);
    simsimd_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        max_f64x4 = _mm256_max_pd(max_f64x4, data_f64x4);
    }
    simsimd_f64_t max_val = _simsimd_reduce_max_f64x4_haswell(max_f64x4);
    for (; idx_scalars < count; ++idx_scalars)
        if (data[idx_scalars] > max_val) max_val = data[idx_scalars];
    // Second pass: find first index
    for (idx_scalars = 0; idx_scalars < count; ++idx_scalars) {
        if (data[idx_scalars] != max_val) continue;
        *max_value = max_val;
        *max_index = idx_scalars;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f64_haswell(                               //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *max_value, simsimd_size_t *max_index) {
    if (stride_bytes == sizeof(simsimd_f64_t) && count >= 4)
        return _simsimd_reduce_max_f64_haswell_contiguous(data, count, max_value, max_index);
    simsimd_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion // x86 Haswell Public Implementations

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#pragma endregion // x86 Haswell Implementations

#pragma region x86 Skylake Implementations

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#pragma region x86 Skylake Internal Helpers

/** @brief Horizontal sum of 16 floats in a ZMM register (native f32 precision). */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_add_f32x16_skylake(__m512 sum_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(sum_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(sum_f32x16, 1);
    __m256 sum_f32x8 = _mm256_add_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(sum_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(sum_f32x8, 1);
    __m128 sum_f32x4 = _mm_add_ps(lo_f32x4, hi_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    return _mm_cvtss_f32(sum_f32x4);
}

/** @brief Horizontal sum of 8 doubles in a ZMM register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_add_f64x8_skylake(__m512d sum_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(sum_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(sum_f64x8, 1);
    __m256d sum_f64x4 = _mm256_add_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(sum_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(sum_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(lo_f64x2, hi_f64x2);
    sum_f64x2 = _mm_hadd_pd(sum_f64x2, sum_f64x2);
    return _mm_cvtsd_f64(sum_f64x2);
}

/** @brief Horizontal min of 16 floats in a ZMM register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_min_f32x16_skylake(__m512 min_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(min_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(min_f32x16, 1);
    __m256 min_f32x8 = _mm256_min_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(min_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(min_f32x8, 1);
    __m128 min_f32x4 = _mm_min_ps(lo_f32x4, hi_f32x4);
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(min_f32x4);
}

/** @brief Horizontal max of 16 floats in a ZMM register. */
SIMSIMD_INTERNAL simsimd_f32_t _simsimd_reduce_max_f32x16_skylake(__m512 max_f32x16) {
    __m256 lo_f32x8 = _mm512_castps512_ps256(max_f32x16);
    __m256 hi_f32x8 = _mm512_extractf32x8_ps(max_f32x16, 1);
    __m256 max_f32x8 = _mm256_max_ps(lo_f32x8, hi_f32x8);
    __m128 lo_f32x4 = _mm256_castps256_ps128(max_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(max_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(lo_f32x4, hi_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max_f32x4);
}

/** @brief Horizontal min of 8 doubles in a ZMM register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_min_f64x8_skylake(__m512d min_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(min_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(min_f64x8, 1);
    __m256d min_f64x4 = _mm256_min_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(min_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(min_f64x4, 1);
    __m128d min_f64x2 = _mm_min_pd(lo_f64x2, hi_f64x2);
    min_f64x2 = _mm_min_pd(min_f64x2, _mm_shuffle_pd(min_f64x2, min_f64x2, 1));
    return _mm_cvtsd_f64(min_f64x2);
}

/** @brief Horizontal max of 8 doubles in a ZMM register. */
SIMSIMD_INTERNAL simsimd_f64_t _simsimd_reduce_max_f64x8_skylake(__m512d max_f64x8) {
    __m256d lo_f64x4 = _mm512_castpd512_pd256(max_f64x8);
    __m256d hi_f64x4 = _mm512_extractf64x4_pd(max_f64x8, 1);
    __m256d max_f64x4 = _mm256_max_pd(lo_f64x4, hi_f64x4);
    __m128d lo_f64x2 = _mm256_castpd256_pd128(max_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(max_f64x4, 1);
    __m128d max_f64x2 = _mm_max_pd(lo_f64x2, hi_f64x2);
    max_f64x2 = _mm_max_pd(max_f64x2, _mm_shuffle_pd(max_f64x2, max_f64x2, 1));
    return _mm_cvtsd_f64(max_f64x2);
}

/** @brief Horizontal sum of 16 i32s in a ZMM register. */
SIMSIMD_INTERNAL simsimd_i32_t _simsimd_reduce_add_i32x16_skylake(__m512i sum_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(sum_i32x16);
    __m256i hi_i32x8 = _mm512_extracti32x8_epi32(sum_i32x16, 1);
    __m256i sum_i32x8 = _mm256_add_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(sum_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(sum_i32x8, 1);
    __m128i sum_i32x4 = _mm_add_epi32(lo_i32x4, hi_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    return _mm_cvtsi128_si32(sum_i32x4);
}

/** @brief Horizontal sum of 8 i64s in a ZMM register. */
SIMSIMD_INTERNAL simsimd_i64_t _simsimd_reduce_add_i64x8_skylake(__m512i sum_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(sum_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(sum_i64x8, 1);
    __m256i sum_i64x4 = _mm256_add_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    sum_i64x2 = _mm_add_epi64(sum_i64x2, _mm_shuffle_epi32(sum_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si64(sum_i64x2);
}

#pragma endregion // x86 Skylake Internal Helpers

#pragma region x86 Skylake Public Implementations

SIMSIMD_INTERNAL void _simsimd_reduce_add_f32_skylake_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(data_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(data_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    // Handle tail with masked load
    simsimd_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(tail_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(tail_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        if (remaining > 8) sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    *result = _simsimd_reduce_add_f64x8_skylake(sum_f64x8);
}

SIMSIMD_INTERNAL void _simsimd_reduce_add_f32_skylake_gather(                     //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    simsimd_i32_t stride_elements = (simsimd_i32_t)(stride_bytes / sizeof(simsimd_f32_t));
    __m512i indices_i32x16 = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                _mm512_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 gathered_f32x16 = _mm512_i32gather_ps(indices_i32x16, data + idx_scalars * stride_elements,
                                                     sizeof(simsimd_f32_t));
        __m256 lo_f32x8 = _mm512_castps512_ps256(gathered_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(gathered_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x8_skylake(sum_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(simsimd_f32_t const *)ptr;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f32_skylake(                               //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    if (stride_bytes == sizeof(simsimd_f32_t)) return _simsimd_reduce_add_f32_skylake_contiguous(data, count, result);
    if (stride_bytes % sizeof(simsimd_f32_t) == 0)
        return _simsimd_reduce_add_f32_skylake_gather(data, count, stride_bytes, result);
    simsimd_reduce_add_f32_serial(data, count, stride_bytes, result);
}

SIMSIMD_INTERNAL void _simsimd_reduce_add_f64_skylake_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, data_f64x8);
    }
    // Handle tail with masked load
    simsimd_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, tail_f64x8);
    }
    *result = _simsimd_reduce_add_f64x8_skylake(sum_f64x8);
}

SIMSIMD_INTERNAL void _simsimd_reduce_add_f64_skylake_gather(                     //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    simsimd_i32_t stride_elements = (simsimd_i32_t)(stride_bytes / sizeof(simsimd_f64_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    simsimd_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d gathered_f64x8 = _mm512_i32gather_pd(indices_i32x8, data + idx_scalars * stride_elements,
                                                     sizeof(simsimd_f64_t));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, gathered_f64x8);
    }
    simsimd_f64_t sum = _simsimd_reduce_add_f64x8_skylake(sum_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(simsimd_f64_t const *)ptr;
    *result = sum;
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f64_skylake(                               //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *result) {
    if (stride_bytes == sizeof(simsimd_f64_t)) return _simsimd_reduce_add_f64_skylake_contiguous(data, count, result);
    if (stride_bytes % sizeof(simsimd_f64_t) == 0)
        return _simsimd_reduce_add_f64_skylake_gather(data, count, stride_bytes, result);
    simsimd_reduce_add_f64_serial(data, count, stride_bytes, result);
}

SIMSIMD_INTERNAL void _simsimd_reduce_min_f32_skylake_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count,              //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512 min_f32x16 = _mm512_loadu_ps(data);
    __m512i min_idx_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_idx_i32x16 = _mm512_set1_epi32(16);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    simsimd_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
        current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    simsimd_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        // Set masked-out lanes to +inf so they don't affect the min
        __m512 inf_f32x16 = _mm512_set1_ps(__builtin_huge_valf());
        tail_f32x16 = _mm512_mask_mov_ps(inf_f32x16, tail_mask, tail_f32x16);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(tail_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, tail_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
    }

    // Horizontal reduction to find lane with minimum
    simsimd_f32_t min_val = _simsimd_reduce_min_f32x16_skylake(min_f32x16);

    // Find the first lane that matches the minimum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    simsimd_i32_t indices[16];
    _mm512_storeu_si512(indices, min_idx_i32x16);

    *min_value = min_val;
    *min_index = (simsimd_size_t)indices[first_lane];
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f32_skylake(                               //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *min_value, simsimd_size_t *min_index) {
    if (stride_bytes == sizeof(simsimd_f32_t) && count >= 16)
        return _simsimd_reduce_min_f32_skylake_contiguous(data, count, min_value, min_index);
    simsimd_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_max_f32_skylake_contiguous( //
    simsimd_f32_t const *data, simsimd_size_t count,              //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512 max_f32x16 = _mm512_loadu_ps(data);
    __m512i max_idx_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_idx_i32x16 = _mm512_set1_epi32(16);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    simsimd_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask, current_idx_i32x16);
        current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    simsimd_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        // Set masked-out lanes to -inf so they don't affect the max
        __m512 neg_inf_f32x16 = _mm512_set1_ps(-__builtin_huge_valf());
        tail_f32x16 = _mm512_mask_mov_ps(neg_inf_f32x16, tail_mask, tail_f32x16);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(tail_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, tail_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask, current_idx_i32x16);
    }

    // Horizontal reduction to find lane with maximum
    simsimd_f32_t max_val = _simsimd_reduce_max_f32x16_skylake(max_f32x16);

    // Find the first lane that matches the maximum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    simsimd_i32_t indices[16];
    _mm512_storeu_si512(indices, max_idx_i32x16);

    *max_value = max_val;
    *max_index = (simsimd_size_t)indices[first_lane];
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f32_skylake(                               //
    simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f32_t *max_value, simsimd_size_t *max_index) {
    if (stride_bytes == sizeof(simsimd_f32_t) && count >= 16)
        return _simsimd_reduce_max_f32_skylake_contiguous(data, count, max_value, max_index);
    simsimd_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_min_f64_skylake_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count,              //
    simsimd_f64_t *min_value, simsimd_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512d min_f64x8 = _mm512_loadu_pd(data);
    __m512i min_idx_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_idx_i64x8 = _mm512_set1_epi64(8);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    simsimd_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask, current_idx_i64x8);
        current_idx_i64x8 = _mm512_add_epi64(current_idx_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    simsimd_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        __m512d inf_f64x8 = _mm512_set1_pd(__builtin_huge_val());
        tail_f64x8 = _mm512_mask_mov_pd(inf_f64x8, tail_mask, tail_f64x8);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(tail_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, tail_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask, current_idx_i64x8);
    }

    // Horizontal reduction
    simsimd_f64_t min_val = _simsimd_reduce_min_f64x8_skylake(min_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    simsimd_i64_t indices[8];
    _mm512_storeu_si512(indices, min_idx_i64x8);

    *min_value = min_val;
    *min_index = (simsimd_size_t)indices[first_lane];
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f64_skylake(                               //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *min_value, simsimd_size_t *min_index) {
    if (stride_bytes == sizeof(simsimd_f64_t) && count >= 8)
        return _simsimd_reduce_min_f64_skylake_contiguous(data, count, min_value, min_index);
    simsimd_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

SIMSIMD_INTERNAL void _simsimd_reduce_max_f64_skylake_contiguous( //
    simsimd_f64_t const *data, simsimd_size_t count,              //
    simsimd_f64_t *max_value, simsimd_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512d max_f64x8 = _mm512_loadu_pd(data);
    __m512i max_idx_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_idx_i64x8 = _mm512_set1_epi64(8);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    simsimd_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask, current_idx_i64x8);
        current_idx_i64x8 = _mm512_add_epi64(current_idx_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    simsimd_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        __m512d neg_inf_f64x8 = _mm512_set1_pd(-__builtin_huge_val());
        tail_f64x8 = _mm512_mask_mov_pd(neg_inf_f64x8, tail_mask, tail_f64x8);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(tail_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, tail_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask, current_idx_i64x8);
    }

    // Horizontal reduction
    simsimd_f64_t max_val = _simsimd_reduce_max_f64x8_skylake(max_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    simsimd_i64_t indices[8];
    _mm512_storeu_si512(indices, max_idx_i64x8);

    *max_value = max_val;
    *max_index = (simsimd_size_t)indices[first_lane];
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f64_skylake(                               //
    simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes, //
    simsimd_f64_t *max_value, simsimd_size_t *max_index) {
    if (stride_bytes == sizeof(simsimd_f64_t) && count >= 8)
        return _simsimd_reduce_max_f64_skylake_contiguous(data, count, max_value, max_index);
    simsimd_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion // x86 Skylake Public Implementations

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE
#endif // _SIMSIMD_TARGET_X86

#pragma endregion // x86 Skylake Implementations

#pragma region Compile-Time Dispatch

#if !SIMSIMD_DYNAMIC_DISPATCH

SIMSIMD_PUBLIC void simsimd_reduce_add_f32(simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes,
                                           simsimd_f64_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_reduce_add_f32_skylake(data, count, stride_bytes, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_reduce_add_f32_haswell(data, count, stride_bytes, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_reduce_add_f32_neon(data, count, stride_bytes, result);
#else
    simsimd_reduce_add_f32_serial(data, count, stride_bytes, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_reduce_add_f64(simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes,
                                           simsimd_f64_t *result) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_reduce_add_f64_skylake(data, count, stride_bytes, result);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_reduce_add_f64_haswell(data, count, stride_bytes, result);
#elif SIMSIMD_TARGET_NEON
    simsimd_reduce_add_f64_neon(data, count, stride_bytes, result);
#else
    simsimd_reduce_add_f64_serial(data, count, stride_bytes, result);
#endif
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f32(simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes,
                                           simsimd_f32_t *min_value, simsimd_size_t *min_index) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_reduce_min_f32_skylake(data, count, stride_bytes, min_value, min_index);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_reduce_min_f32_haswell(data, count, stride_bytes, min_value, min_index);
#elif SIMSIMD_TARGET_NEON
    simsimd_reduce_min_f32_neon(data, count, stride_bytes, min_value, min_index);
#else
    simsimd_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
#endif
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f32(simsimd_f32_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes,
                                           simsimd_f32_t *max_value, simsimd_size_t *max_index) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_reduce_max_f32_skylake(data, count, stride_bytes, max_value, max_index);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_reduce_max_f32_haswell(data, count, stride_bytes, max_value, max_index);
#elif SIMSIMD_TARGET_NEON
    simsimd_reduce_max_f32_neon(data, count, stride_bytes, max_value, max_index);
#else
    simsimd_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
#endif
}

SIMSIMD_PUBLIC void simsimd_reduce_min_f64(simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes,
                                           simsimd_f64_t *min_value, simsimd_size_t *min_index) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_reduce_min_f64_skylake(data, count, stride_bytes, min_value, min_index);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_reduce_min_f64_haswell(data, count, stride_bytes, min_value, min_index);
#else
    simsimd_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
#endif
}

SIMSIMD_PUBLIC void simsimd_reduce_max_f64(simsimd_f64_t const *data, simsimd_size_t count, simsimd_size_t stride_bytes,
                                           simsimd_f64_t *max_value, simsimd_size_t *max_index) {
#if SIMSIMD_TARGET_SKYLAKE
    simsimd_reduce_max_f64_skylake(data, count, stride_bytes, max_value, max_index);
#elif SIMSIMD_TARGET_HASWELL
    simsimd_reduce_max_f64_haswell(data, count, stride_bytes, max_value, max_index);
#else
    simsimd_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
#endif
}

#endif // !SIMSIMD_DYNAMIC_DISPATCH

#pragma endregion // Compile-Time Dispatch

#ifdef __cplusplus
}
#endif

#endif // SIMSIMD_REDUCE_H
