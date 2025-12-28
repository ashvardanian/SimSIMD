/**
 *  @brief SIMD-accelerated horizontal reduction operations.
 *  @file include/numkong/reduce.h
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
 *  nk_f32_t result = nk_reduce_add_f32x16_skylake_(sum_f32x16);
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
#ifndef NK_REDUCE_H
#define NK_REDUCE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Horizontal sum reduction over a strided array of f32 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f32) for contiguous).
 *  @param[out] result      Output sum, widened to f64 for precision.
 */
NK_DYNAMIC void nk_reduce_add_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *result);

/**
 *  @brief  Horizontal sum reduction over a strided array of f64 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f64) for contiguous).
 *  @param[out] result      Output sum.
 */
NK_DYNAMIC void nk_reduce_add_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *result);

/**
 *  @brief  Horizontal minimum reduction with argmin over a strided array of f32 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f32) for contiguous).
 *  @param[out] min_value   Output minimum value.
 *  @param[out] min_index   Output index of the minimum value.
 */
NK_DYNAMIC void nk_reduce_min_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *min_value,
                                  nk_size_t *min_index);

/**
 *  @brief  Horizontal maximum reduction with argmax over a strided array of f32 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f32) for contiguous).
 *  @param[out] max_value   Output maximum value.
 *  @param[out] max_index   Output index of the maximum value.
 */
NK_DYNAMIC void nk_reduce_max_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *max_value,
                                  nk_size_t *max_index);

/**
 *  @brief  Horizontal minimum reduction with argmin over a strided array of f64 values.
 *  @param[in] data         Pointer to the input data.
 *  @param[in] count        Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f64) for contiguous).
 *  @param[out] min_value   Output minimum value.
 *  @param[out] min_index   Output index of the minimum value.
 */
NK_DYNAMIC void nk_reduce_min_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *min_value,
                                  nk_size_t *min_index);

/**
 *  @brief Horizontal maximum reduction with argmax over a strided array of f64 values.
 *
 *  @param[in] data Pointer to the input data.
 *  @param[in] count Number of elements to reduce.
 *  @param[in] stride_bytes Stride between elements in bytes (sizeof(f64) for contiguous).
 *  @param[out] max_value Output maximum value.
 *  @param[out] max_index Output index of the maximum value.
 */
NK_DYNAMIC void nk_reduce_max_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *max_value,
                                  nk_size_t *max_index);

/** @copydoc nk_reduce_add_f32 */
NK_PUBLIC void nk_reduce_add_f32_serial(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_f64_t *result);
/** @copydoc nk_reduce_add_f64 */
NK_PUBLIC void nk_reduce_add_f64_serial(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_f64_t *result);
/** @copydoc nk_reduce_add_f32 but for i8 input with i64 output */
NK_PUBLIC void nk_reduce_add_i8_serial(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *result);
/** @copydoc nk_reduce_add_f32 but for u8 input with u64 output */
NK_PUBLIC void nk_reduce_add_u8_serial(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *result);
/** @copydoc nk_reduce_add_f32 but for i16 input with i64 output */
NK_PUBLIC void nk_reduce_add_i16_serial(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i64_t *result);
/** @copydoc nk_reduce_add_f32 but for u16 input with u64 output */
NK_PUBLIC void nk_reduce_add_u16_serial(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u64_t *result);
/** @copydoc nk_reduce_add_f32 but for i32 input with i64 output */
NK_PUBLIC void nk_reduce_add_i32_serial(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i64_t *result);
/** @copydoc nk_reduce_add_f32 but for u32 input with u64 output */
NK_PUBLIC void nk_reduce_add_u32_serial(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u64_t *result);
/** @copydoc nk_reduce_add_f64 but for i64 input */
NK_PUBLIC void nk_reduce_add_i64_serial(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i64_t *result);
/** @copydoc nk_reduce_add_f64 but for u64 input */
NK_PUBLIC void nk_reduce_add_u64_serial(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u64_t *result);
/** @copydoc nk_reduce_min_f32 but for i8 */
NK_PUBLIC void nk_reduce_min_i8_serial(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i8_t *min_value,
                                       nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 but for i8 */
NK_PUBLIC void nk_reduce_max_i8_serial(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i8_t *max_value,
                                       nk_size_t *max_index);
/** @copydoc nk_reduce_min_f32 but for u8 */
NK_PUBLIC void nk_reduce_min_u8_serial(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u8_t *min_value,
                                       nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 but for u8 */
NK_PUBLIC void nk_reduce_max_u8_serial(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u8_t *max_value,
                                       nk_size_t *max_index);
/** @copydoc nk_reduce_min_f32 but for i16 */
NK_PUBLIC void nk_reduce_min_i16_serial(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i16_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 but for i16 */
NK_PUBLIC void nk_reduce_max_i16_serial(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i16_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f32 but for u16 */
NK_PUBLIC void nk_reduce_min_u16_serial(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u16_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 but for u16 */
NK_PUBLIC void nk_reduce_max_u16_serial(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u16_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f32 but for i32 */
NK_PUBLIC void nk_reduce_min_i32_serial(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i32_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 but for i32 */
NK_PUBLIC void nk_reduce_max_i32_serial(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i32_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f32 but for u32 */
NK_PUBLIC void nk_reduce_min_u32_serial(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u32_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 but for u32 */
NK_PUBLIC void nk_reduce_max_u32_serial(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u32_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f64 but for i64 */
NK_PUBLIC void nk_reduce_min_i64_serial(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i64_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f64 but for i64 */
NK_PUBLIC void nk_reduce_max_i64_serial(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i64_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f64 but for u64 */
NK_PUBLIC void nk_reduce_min_u64_serial(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u64_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f64 but for u64 */
NK_PUBLIC void nk_reduce_max_u64_serial(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u64_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f32 */
NK_PUBLIC void nk_reduce_min_f32_serial(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_f32_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 */
NK_PUBLIC void nk_reduce_max_f32_serial(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_f32_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f64 */
NK_PUBLIC void nk_reduce_min_f64_serial(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_f64_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f64 */
NK_PUBLIC void nk_reduce_max_f64_serial(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_f64_t *max_value, nk_size_t *max_index);

#if NK_TARGET_NEON
/** @copydoc nk_reduce_add_f32 */
NK_PUBLIC void nk_reduce_add_f32_neon(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *result);
/** @copydoc nk_reduce_add_f64 */
NK_PUBLIC void nk_reduce_add_f64_neon(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *result);
/** @copydoc nk_reduce_min_f32 */
NK_PUBLIC void nk_reduce_min_f32_neon(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 */
NK_PUBLIC void nk_reduce_max_f32_neon(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                      nk_f32_t *max_value, nk_size_t *max_index);
#endif // NK_TARGET_NEON

#if NK_TARGET_HASWELL
/** @copydoc nk_reduce_add_f32 */
NK_PUBLIC void nk_reduce_add_f32_haswell(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *result);
/** @copydoc nk_reduce_add_f64 */
NK_PUBLIC void nk_reduce_add_f64_haswell(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *result);
/** @copydoc nk_reduce_min_f32 */
NK_PUBLIC void nk_reduce_min_f32_haswell(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f32_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 */
NK_PUBLIC void nk_reduce_max_f32_haswell(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f32_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f64 */
NK_PUBLIC void nk_reduce_min_f64_haswell(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f64 */
NK_PUBLIC void nk_reduce_max_f64_haswell(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *max_value, nk_size_t *max_index);
/** @brief i8 sum reduction with i64 accumulator */
NK_PUBLIC void nk_reduce_add_i8_haswell(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *result);
/** @brief u8 sum reduction with u64 accumulator */
NK_PUBLIC void nk_reduce_add_u8_haswell(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *result);
/** @brief i16 sum reduction with i64 accumulator */
NK_PUBLIC void nk_reduce_add_i16_haswell(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *result);
/** @brief u16 sum reduction with u64 accumulator */
NK_PUBLIC void nk_reduce_add_u16_haswell(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *result);
/** @brief i32 sum reduction with i64 accumulator */
NK_PUBLIC void nk_reduce_add_i32_haswell(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *result);
/** @brief u32 sum reduction with u64 accumulator */
NK_PUBLIC void nk_reduce_add_u32_haswell(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *result);
/** @brief i64 sum reduction */
NK_PUBLIC void nk_reduce_add_i64_haswell(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *result);
/** @brief u64 sum reduction */
NK_PUBLIC void nk_reduce_add_u64_haswell(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *result);
/** @brief i8 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i8_haswell(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i8_t *min_value, nk_size_t *min_index);
/** @brief i8 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i8_haswell(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i8_t *max_value, nk_size_t *max_index);
/** @brief u8 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u8_haswell(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u8_t *min_value, nk_size_t *min_index);
/** @brief u8 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u8_haswell(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u8_t *max_value, nk_size_t *max_index);
/** @brief i16 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i16_haswell(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i16_t *min_value, nk_size_t *min_index);
/** @brief i16 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i16_haswell(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i16_t *max_value, nk_size_t *max_index);
/** @brief u16 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u16_haswell(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u16_t *min_value, nk_size_t *min_index);
/** @brief u16 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u16_haswell(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u16_t *max_value, nk_size_t *max_index);
/** @brief i32 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i32_haswell(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i32_t *min_value, nk_size_t *min_index);
/** @brief i32 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i32_haswell(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i32_t *max_value, nk_size_t *max_index);
/** @brief u32 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u32_haswell(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u32_t *min_value, nk_size_t *min_index);
/** @brief u32 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u32_haswell(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u32_t *max_value, nk_size_t *max_index);
/** @brief i64 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i64_haswell(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *min_value, nk_size_t *min_index);
/** @brief i64 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i64_haswell(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *max_value, nk_size_t *max_index);
/** @brief u64 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u64_haswell(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *min_value, nk_size_t *min_index);
/** @brief u64 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u64_haswell(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *max_value, nk_size_t *max_index);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_reduce_add_f32 */
NK_PUBLIC void nk_reduce_add_f32_skylake(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *result);
/** @copydoc nk_reduce_add_f64 */
NK_PUBLIC void nk_reduce_add_f64_skylake(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *result);
/** @copydoc nk_reduce_min_f32 */
NK_PUBLIC void nk_reduce_min_f32_skylake(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f32_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f32 */
NK_PUBLIC void nk_reduce_max_f32_skylake(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f32_t *max_value, nk_size_t *max_index);
/** @copydoc nk_reduce_min_f64 */
NK_PUBLIC void nk_reduce_min_f64_skylake(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *min_value, nk_size_t *min_index);
/** @copydoc nk_reduce_max_f64 */
NK_PUBLIC void nk_reduce_max_f64_skylake(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_f64_t *max_value, nk_size_t *max_index);
/** @brief i8 sum reduction with i64 accumulator */
NK_PUBLIC void nk_reduce_add_i8_skylake(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_i64_t *result);
/** @brief u8 sum reduction with u64 accumulator */
NK_PUBLIC void nk_reduce_add_u8_skylake(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_u64_t *result);
/** @brief i16 sum reduction with i64 accumulator */
NK_PUBLIC void nk_reduce_add_i16_skylake(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *result);
/** @brief u16 sum reduction with u64 accumulator */
NK_PUBLIC void nk_reduce_add_u16_skylake(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *result);
/** @brief i32 sum reduction with i64 accumulator */
NK_PUBLIC void nk_reduce_add_i32_skylake(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *result);
/** @brief u32 sum reduction with u64 accumulator */
NK_PUBLIC void nk_reduce_add_u32_skylake(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *result);
/** @brief i64 sum reduction */
NK_PUBLIC void nk_reduce_add_i64_skylake(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *result);
/** @brief u64 sum reduction */
NK_PUBLIC void nk_reduce_add_u64_skylake(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *result);
/** @brief i8 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i8_skylake(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i8_t *min_value, nk_size_t *min_index);
/** @brief i8 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i8_skylake(nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_i8_t *max_value, nk_size_t *max_index);
/** @brief u8 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u8_skylake(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u8_t *min_value, nk_size_t *min_index);
/** @brief u8 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u8_skylake(nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                        nk_u8_t *max_value, nk_size_t *max_index);
/** @brief i16 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i16_skylake(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i16_t *min_value, nk_size_t *min_index);
/** @brief i16 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i16_skylake(nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i16_t *max_value, nk_size_t *max_index);
/** @brief u16 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u16_skylake(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u16_t *min_value, nk_size_t *min_index);
/** @brief u16 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u16_skylake(nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u16_t *max_value, nk_size_t *max_index);
/** @brief i32 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i32_skylake(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i32_t *min_value, nk_size_t *min_index);
/** @brief i32 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i32_skylake(nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i32_t *max_value, nk_size_t *max_index);
/** @brief u32 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u32_skylake(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u32_t *min_value, nk_size_t *min_index);
/** @brief u32 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u32_skylake(nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u32_t *max_value, nk_size_t *max_index);
/** @brief i64 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_i64_skylake(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *min_value, nk_size_t *min_index);
/** @brief i64 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_i64_skylake(nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_i64_t *max_value, nk_size_t *max_index);
/** @brief u64 min reduction with argmin */
NK_PUBLIC void nk_reduce_min_u64_skylake(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *min_value, nk_size_t *min_index);
/** @brief u64 max reduction with argmax */
NK_PUBLIC void nk_reduce_max_u64_skylake(nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes,
                                         nk_u64_t *max_value, nk_size_t *max_index);
#endif // NK_TARGET_SKYLAKE

#include "numkong/reduce/neon.h"
#include "numkong/reduce/neon_f16.h"
#include "numkong/reduce/haswell.h"
#include "numkong/reduce/skylake.h"

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_reduce_add_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_reduce_add_f32_skylake(data, count, stride_bytes, result);
#elif NK_TARGET_HASWELL
    nk_reduce_add_f32_haswell(data, count, stride_bytes, result);
#elif NK_TARGET_NEON
    nk_reduce_add_f32_neon(data, count, stride_bytes, result);
#else
    nk_reduce_add_f32_serial(data, count, stride_bytes, result);
#endif
}

NK_PUBLIC void nk_reduce_add_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_reduce_add_f64_skylake(data, count, stride_bytes, result);
#elif NK_TARGET_HASWELL
    nk_reduce_add_f64_haswell(data, count, stride_bytes, result);
#elif NK_TARGET_NEON
    nk_reduce_add_f64_neon(data, count, stride_bytes, result);
#else
    nk_reduce_add_f64_serial(data, count, stride_bytes, result);
#endif
}

NK_PUBLIC void nk_reduce_min_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *min_value,
                                 nk_size_t *min_index) {
#if NK_TARGET_SKYLAKE
    nk_reduce_min_f32_skylake(data, count, stride_bytes, min_value, min_index);
#elif NK_TARGET_HASWELL
    nk_reduce_min_f32_haswell(data, count, stride_bytes, min_value, min_index);
#elif NK_TARGET_NEON
    nk_reduce_min_f32_neon(data, count, stride_bytes, min_value, min_index);
#else
    nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
#endif
}

NK_PUBLIC void nk_reduce_max_f32(nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f32_t *max_value,
                                 nk_size_t *max_index) {
#if NK_TARGET_SKYLAKE
    nk_reduce_max_f32_skylake(data, count, stride_bytes, max_value, max_index);
#elif NK_TARGET_HASWELL
    nk_reduce_max_f32_haswell(data, count, stride_bytes, max_value, max_index);
#elif NK_TARGET_NEON
    nk_reduce_max_f32_neon(data, count, stride_bytes, max_value, max_index);
#else
    nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
#endif
}

NK_PUBLIC void nk_reduce_min_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *min_value,
                                 nk_size_t *min_index) {
#if NK_TARGET_SKYLAKE
    nk_reduce_min_f64_skylake(data, count, stride_bytes, min_value, min_index);
#elif NK_TARGET_HASWELL
    nk_reduce_min_f64_haswell(data, count, stride_bytes, min_value, min_index);
#else
    nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
#endif
}

NK_PUBLIC void nk_reduce_max_f64(nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, nk_f64_t *max_value,
                                 nk_size_t *max_index) {
#if NK_TARGET_SKYLAKE
    nk_reduce_max_f64_skylake(data, count, stride_bytes, max_value, max_index);
#elif NK_TARGET_HASWELL
    nk_reduce_max_f64_haswell(data, count, stride_bytes, max_value, max_index);
#else
    nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#ifdef __cplusplus
}
#endif

#endif // NK_REDUCE_H
