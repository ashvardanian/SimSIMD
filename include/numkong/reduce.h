/**
 *  @brief SIMD-accelerated Horizontal Reduction Operations.
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
 *  nk_f32_t result = _nk_reduce_add_f32x16_skylake(sum_f32x16);
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

#pragma region NK_DYNAMIC Function Declarations

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

#pragma endregion

#pragma region Forward Declarations - Serial

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

#pragma endregion

#pragma region Forward Declarations - ARM NEON

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

#pragma endregion

#pragma region Forward Declarations - x86 Haswell

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

#pragma endregion

#pragma region Forward Declarations - x86 Skylake

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

#pragma endregion

#pragma region Serial Implementations

NK_PUBLIC void nk_reduce_add_f32_serial(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_f64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_f32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_serial(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_f64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_f64_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i8_serial(                           //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i8_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u8_serial(                           //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u8_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i16_serial(                           //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i16_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u16_serial(                           //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u16_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i32_serial(                           //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u32_serial(                           //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_i64_serial(                           //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    nk_i64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_i64_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_u64_serial(                           //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    nk_u64_t sum = 0;
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_bytes) sum += *(nk_u64_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_min_i8_serial(                           //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_value = *(nk_i8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i8_t val = *(nk_i8_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i8_serial(                           //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i8_t best_value = *(nk_i8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i8_t val = *(nk_i8_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u8_serial(                           //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_value = *(nk_u8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u8_t val = *(nk_u8_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u8_serial(                           //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u8_t best_value = *(nk_u8_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u8_t val = *(nk_u8_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_i16_serial(                           //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_value = *(nk_i16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i16_t val = *(nk_i16_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i16_serial(                           //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i16_t best_value = *(nk_i16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i16_t val = *(nk_i16_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u16_serial(                           //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_value = *(nk_u16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u16_t val = *(nk_u16_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u16_serial(                           //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u16_t best_value = *(nk_u16_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u16_t val = *(nk_u16_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_i32_serial(                           //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_value = *(nk_i32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i32_t val = *(nk_i32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i32_serial(                           //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i32_t best_value = *(nk_i32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i32_t val = *(nk_i32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u32_serial(                           //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_value = *(nk_u32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u32_t val = *(nk_u32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u32_serial(                           //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u32_t best_value = *(nk_u32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u32_t val = *(nk_u32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_i64_serial(                           //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_value = *(nk_i64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_i64_serial(                           //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_i64_t best_value = *(nk_i64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_i64_t val = *(nk_i64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_u64_serial(                           //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_value = *(nk_u64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_u64_serial(                           //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_u64_t best_value = *(nk_u64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_u64_t val = *(nk_u64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_f32_serial(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = *(nk_f32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val = *(nk_f32_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_f32_serial(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f32_t best_value = *(nk_f32_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f32_t val = *(nk_f32_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

NK_PUBLIC void nk_reduce_min_f64_serial(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    if (count == 0) {
        *min_value = 0;
        *min_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_value = *(nk_f64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        if (val >= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *min_value = best_value;
    *min_index = best_index;
}

NK_PUBLIC void nk_reduce_max_f64_serial(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    if (count == 0) {
        *max_value = 0;
        *max_index = 0;
        return;
    }
    unsigned char const *ptr = (unsigned char const *)data;
    nk_f64_t best_value = *(nk_f64_t const *)ptr;
    nk_size_t best_index = 0;
    ptr += stride_bytes;
    for (nk_size_t i = 1; i < count; ++i, ptr += stride_bytes) {
        nk_f64_t val = *(nk_f64_t const *)ptr;
        if (val <= best_value) continue;
        best_value = val;
        best_index = i;
    }
    *max_value = best_value;
    *max_index = best_index;
}

#pragma endregion

#pragma region ARM NEON Implementations

#if _NK_TARGET_ARM
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

#pragma region ARM NEON Internal Helpers

/** @brief Horizontal sum of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t _nk_reduce_add_f32x4_neon(float32x4_t sum_f32x4) { return vaddvq_f32(sum_f32x4); }

/** @brief Horizontal sum of 2 doubles in a NEON register. */
NK_INTERNAL nk_f64_t _nk_reduce_add_f64x2_neon(float64x2_t sum_f64x2) { return vaddvq_f64(sum_f64x2); }

/** @brief Horizontal min of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t _nk_reduce_min_f32x4_neon(float32x4_t min_f32x4) { return vminvq_f32(min_f32x4); }

/** @brief Horizontal max of 4 floats in a NEON register. */
NK_INTERNAL nk_f32_t _nk_reduce_max_f32x4_neon(float32x4_t max_f32x4) { return vmaxvq_f32(max_f32x4); }

/** @brief Horizontal sum of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t _nk_reduce_add_i32x4_neon(int32x4_t sum_i32x4) { return vaddvq_s32(sum_i32x4); }

/** @brief Horizontal min of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t _nk_reduce_min_i32x4_neon(int32x4_t min_i32x4) { return vminvq_s32(min_i32x4); }

/** @brief Horizontal max of 4 i32s in a NEON register. */
NK_INTERNAL nk_i32_t _nk_reduce_max_i32x4_neon(int32x4_t max_i32x4) { return vmaxvq_s32(max_i32x4); }

/** @brief Horizontal sum of 16 u8s in a NEON register, returning u32. */
NK_INTERNAL nk_u32_t _nk_reduce_add_u8x16_neon(uint8x16_t sum_u8x16) {
    uint16x8_t low_u16x8 = vmovl_u8(vget_low_u8(sum_u8x16));
    uint16x8_t high_u16x8 = vmovl_u8(vget_high_u8(sum_u8x16));
    uint16x8_t sum_u16x8 = vaddq_u16(low_u16x8, high_u16x8);
    uint32x4_t sum_u32x4 = vpaddlq_u16(sum_u16x8);
    uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
    return (nk_u32_t)vaddvq_u64(sum_u64x2);
}

#pragma endregion // ARM NEON Internal Helpers

#pragma region ARM NEON Public Implementations

NK_INTERNAL void _nk_reduce_add_f32_neon_contiguous( //
    nk_f32_t const *data, nk_size_t count, nk_f64_t *result) {
    // Accumulate in f64 for precision
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        float64x2_t lo_f64x2 = vcvt_f64_f32(vget_low_f32(data_f32x4));
        float64x2_t hi_f64x2 = vcvt_f64_f32(vget_high_f32(data_f32x4));
        sum_f64x2 = vaddq_f64(sum_f64x2, lo_f64x2);
        sum_f64x2 = vaddq_f64(sum_f64x2, hi_f64x2);
    }
    nk_f64_t sum = _nk_reduce_add_f64x2_neon(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    if (stride_bytes == sizeof(nk_f32_t)) return _nk_reduce_add_f32_neon_contiguous(data, count, result);
    nk_reduce_add_f32_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_add_f64_neon_contiguous( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 2 <= count; idx_scalars += 2) {
        float64x2_t data_f64x2 = vld1q_f64(data + idx_scalars);
        sum_f64x2 = vaddq_f64(sum_f64x2, data_f64x2);
    }
    nk_f64_t sum = _nk_reduce_add_f64x2_neon(sum_f64x2);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_neon(                             //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    if (stride_bytes == sizeof(nk_f64_t)) return _nk_reduce_add_f64_neon_contiguous(data, count, result);
    nk_reduce_add_f64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_min_f32_neon_contiguous( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // First pass: find minimum value using SIMD
    float32x4_t min_f32x4 = vld1q_f32(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        min_f32x4 = vminq_f32(min_f32x4, data_f32x4);
    }
    nk_f32_t min_val = _nk_reduce_min_f32x4_neon(min_f32x4);
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

NK_PUBLIC void nk_reduce_min_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_f32_t) && count >= 4)
        return _nk_reduce_min_f32_neon_contiguous(data, count, min_value, min_index);
    nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void _nk_reduce_max_f32_neon_contiguous( //
    nk_f32_t const *data, nk_size_t count,           //
    nk_f32_t *max_value, nk_size_t *max_index) {
    float32x4_t max_f32x4 = vld1q_f32(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        float32x4_t data_f32x4 = vld1q_f32(data + idx_scalars);
        max_f32x4 = vmaxq_f32(max_f32x4, data_f32x4);
    }
    nk_f32_t max_val = _nk_reduce_max_f32x4_neon(max_f32x4);
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

NK_PUBLIC void nk_reduce_max_f32_neon(                             //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_f32_t) && count >= 4)
        return _nk_reduce_max_f32_neon_contiguous(data, count, max_value, max_index);
    nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion // ARM NEON Public Implementations

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON

#if NK_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

/** @brief Horizontal sum of 8 f16s in a NEON register, returning f32. */
NK_INTERNAL nk_f32_t _nk_reduce_add_f16x8_neon(float16x8_t sum_f16x8) {
    float16x4_t low_f16x4 = vget_low_f16(sum_f16x8);
    float16x4_t high_f16x4 = vget_high_f16(sum_f16x8);
    float16x4_t sum_f16x4 = vadd_f16(low_f16x4, high_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    sum_f16x4 = vpadd_f16(sum_f16x4, sum_f16x4);
    return vgetq_lane_f32(vcvt_f32_f16(sum_f16x4), 0);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_F16
#endif // _NK_TARGET_ARM

#pragma endregion // ARM NEON Implementations

#pragma region x86 Haswell Implementations

#if _NK_TARGET_X86
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)

#pragma region x86 Haswell Internal Helpers

/** @brief Horizontal sum of 4 doubles in a YMM register. */
NK_INTERNAL nk_f64_t _nk_reduce_add_f64x4_haswell(__m256d sum_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(sum_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(sum_f64x4, 1);
    __m128d sum_f64x2 = _mm_add_pd(lo_f64x2, hi_f64x2);
    sum_f64x2 = _mm_hadd_pd(sum_f64x2, sum_f64x2);
    return _mm_cvtsd_f64(sum_f64x2);
}

/** @brief Horizontal sum of 8 floats in a YMM register (native f32 precision). */
NK_INTERNAL nk_f32_t _nk_reduce_add_f32x8_haswell(__m256 sum_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(sum_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(sum_f32x8, 1);
    __m128 sum_f32x4 = _mm_add_ps(lo_f32x4, hi_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    sum_f32x4 = _mm_hadd_ps(sum_f32x4, sum_f32x4);
    return _mm_cvtss_f32(sum_f32x4);
}

/** @brief Horizontal sum of 8 i32s in a YMM register. */
NK_INTERNAL nk_i32_t _nk_reduce_add_i32x8_haswell(__m256i sum_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(sum_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(sum_i32x8, 1);
    __m128i sum_i32x4 = _mm_add_epi32(lo_i32x4, hi_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    sum_i32x4 = _mm_hadd_epi32(sum_i32x4, sum_i32x4);
    return _mm_cvtsi128_si32(sum_i32x4);
}

/** @brief Horizontal sum of 4 i64s in a YMM register. */
NK_INTERNAL nk_i64_t _nk_reduce_add_i64x4_haswell(__m256i sum_i64x4) {
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(sum_i64x2, sum_i64x2);
    __m128i final_i64 = _mm_add_epi64(sum_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal min of 8 signed i8s in a YMM register. */
NK_INTERNAL nk_i8_t _nk_reduce_min_i8x32_haswell(__m256i min_i8x32) {
    __m128i lo_i8x16 = _mm256_castsi256_si128(min_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(min_i8x32, 1);
    __m128i min_i8x16 = _mm_min_epi8(lo_i8x16, hi_i8x16);
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shufflelo_epi16(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_epi16(min_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(min_i8x16);
}

/** @brief Horizontal max of 8 signed i8s in a YMM register. */
NK_INTERNAL nk_i8_t _nk_reduce_max_i8x32_haswell(__m256i max_i8x32) {
    __m128i lo_i8x16 = _mm256_castsi256_si128(max_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(max_i8x32, 1);
    __m128i max_i8x16 = _mm_max_epi8(lo_i8x16, hi_i8x16);
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shufflelo_epi16(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_epi16(max_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(max_i8x16);
}

/** @brief Horizontal min of 8 unsigned u8s in a YMM register. */
NK_INTERNAL nk_u8_t _nk_reduce_min_u8x32_haswell(__m256i min_u8x32) {
    __m128i lo_u8x16 = _mm256_castsi256_si128(min_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(min_u8x32, 1);
    __m128i min_u8x16 = _mm_min_epu8(lo_u8x16, hi_u8x16);
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shufflelo_epi16(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_epi16(min_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(min_u8x16);
}

/** @brief Horizontal max of 8 unsigned u8s in a YMM register. */
NK_INTERNAL nk_u8_t _nk_reduce_max_u8x32_haswell(__m256i max_u8x32) {
    __m128i lo_u8x16 = _mm256_castsi256_si128(max_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(max_u8x32, 1);
    __m128i max_u8x16 = _mm_max_epu8(lo_u8x16, hi_u8x16);
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shufflelo_epi16(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_epi16(max_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(max_u8x16);
}

/** @brief Horizontal min of 16 signed i16s in a YMM register. */
NK_INTERNAL nk_i16_t _nk_reduce_min_i16x16_haswell(__m256i min_i16x16) {
    __m128i lo_i16x8 = _mm256_castsi256_si128(min_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(min_i16x16, 1);
    __m128i min_i16x8 = _mm_min_epi16(lo_i16x8, hi_i16x8);
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shufflelo_epi16(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(min_i16x8);
}

/** @brief Horizontal max of 16 signed i16s in a YMM register. */
NK_INTERNAL nk_i16_t _nk_reduce_max_i16x16_haswell(__m256i max_i16x16) {
    __m128i lo_i16x8 = _mm256_castsi256_si128(max_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(max_i16x16, 1);
    __m128i max_i16x8 = _mm_max_epi16(lo_i16x8, hi_i16x8);
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shufflelo_epi16(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(max_i16x8);
}

/** @brief Horizontal min of 16 unsigned u16s in a YMM register. */
NK_INTERNAL nk_u16_t _nk_reduce_min_u16x16_haswell(__m256i min_u16x16) {
    __m128i lo_u16x8 = _mm256_castsi256_si128(min_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(min_u16x16, 1);
    __m128i min_u16x8 = _mm_min_epu16(lo_u16x8, hi_u16x8);
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shufflelo_epi16(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(min_u16x8);
}

/** @brief Horizontal max of 16 unsigned u16s in a YMM register. */
NK_INTERNAL nk_u16_t _nk_reduce_max_u16x16_haswell(__m256i max_u16x16) {
    __m128i lo_u16x8 = _mm256_castsi256_si128(max_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(max_u16x16, 1);
    __m128i max_u16x8 = _mm_max_epu16(lo_u16x8, hi_u16x8);
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shufflelo_epi16(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(max_u16x8);
}

/** @brief Horizontal min of 8 signed i32s in a YMM register. */
NK_INTERNAL nk_i32_t _nk_reduce_min_i32x8_haswell(__m256i min_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(min_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(min_i32x8, 1);
    __m128i min_i32x4 = _mm_min_epi32(lo_i32x4, hi_i32x4);
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(min_i32x4);
}

/** @brief Horizontal max of 8 signed i32s in a YMM register. */
NK_INTERNAL nk_i32_t _nk_reduce_max_i32x8_haswell(__m256i max_i32x8) {
    __m128i lo_i32x4 = _mm256_castsi256_si128(max_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(max_i32x8, 1);
    __m128i max_i32x4 = _mm_max_epi32(lo_i32x4, hi_i32x4);
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(max_i32x4);
}

/** @brief Horizontal min of 8 unsigned u32s in a YMM register. */
NK_INTERNAL nk_u32_t _nk_reduce_min_u32x8_haswell(__m256i min_u32x8) {
    __m128i lo_u32x4 = _mm256_castsi256_si128(min_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(min_u32x8, 1);
    __m128i min_u32x4 = _mm_min_epu32(lo_u32x4, hi_u32x4);
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(min_u32x4);
}

/** @brief Horizontal max of 8 unsigned u32s in a YMM register. */
NK_INTERNAL nk_u32_t _nk_reduce_max_u32x8_haswell(__m256i max_u32x8) {
    __m128i lo_u32x4 = _mm256_castsi256_si128(max_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(max_u32x8, 1);
    __m128i max_u32x4 = _mm_max_epu32(lo_u32x4, hi_u32x4);
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(max_u32x4);
}

/** @brief Horizontal min of 4 signed i64s in a YMM register using comparison+blend. */
NK_INTERNAL nk_i64_t _nk_reduce_min_i64x4_haswell(__m256i min_i64x4) {
    __m128i lo_i64x2 = _mm256_castsi256_si128(min_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(lo_i64x2, hi_i64x2);
    __m128i min_i64x2 = _mm_blendv_epi8(lo_i64x2, hi_i64x2, cmp_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(min_i64x2, min_i64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(min_i64x2, hi_lane_i64);
    __m128i result_i64 = _mm_blendv_epi8(min_i64x2, hi_lane_i64, cmp_final);
    return _mm_cvtsi128_si64(result_i64);
}

/** @brief Horizontal max of 4 signed i64s in a YMM register using comparison+blend. */
NK_INTERNAL nk_i64_t _nk_reduce_max_i64x4_haswell(__m256i max_i64x4) {
    __m128i lo_i64x2 = _mm256_castsi256_si128(max_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(lo_i64x2, hi_i64x2);
    __m128i max_i64x2 = _mm_blendv_epi8(hi_i64x2, lo_i64x2, cmp_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(max_i64x2, max_i64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(max_i64x2, hi_lane_i64);
    __m128i result_i64 = _mm_blendv_epi8(hi_lane_i64, max_i64x2, cmp_final);
    return _mm_cvtsi128_si64(result_i64);
}

/** @brief Horizontal min of 4 unsigned u64s in a YMM register using XOR trick for unsigned comparison. */
NK_INTERNAL nk_u64_t _nk_reduce_min_u64x4_haswell(__m256i min_u64x4) {
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    __m128i lo_u64x2 = _mm256_castsi256_si128(min_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(_mm_xor_si128(lo_u64x2, sign_bit_i64), _mm_xor_si128(hi_u64x2, sign_bit_i64));
    __m128i min_u64x2 = _mm_blendv_epi8(lo_u64x2, hi_u64x2, cmp_i64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(min_u64x2, min_u64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(_mm_xor_si128(min_u64x2, sign_bit_i64),
                                        _mm_xor_si128(hi_lane_u64, sign_bit_i64));
    __m128i result_u64 = _mm_blendv_epi8(min_u64x2, hi_lane_u64, cmp_final);
    return (nk_u64_t)_mm_cvtsi128_si64(result_u64);
}

/** @brief Horizontal max of 4 unsigned u64s in a YMM register using XOR trick for unsigned comparison. */
NK_INTERNAL nk_u64_t _nk_reduce_max_u64x4_haswell(__m256i max_u64x4) {
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    __m128i lo_u64x2 = _mm256_castsi256_si128(max_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
    __m128i cmp_i64x2 = _mm_cmpgt_epi64(_mm_xor_si128(lo_u64x2, sign_bit_i64), _mm_xor_si128(hi_u64x2, sign_bit_i64));
    __m128i max_u64x2 = _mm_blendv_epi8(hi_u64x2, lo_u64x2, cmp_i64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(max_u64x2, max_u64x2);
    __m128i cmp_final = _mm_cmpgt_epi64(_mm_xor_si128(max_u64x2, sign_bit_i64),
                                        _mm_xor_si128(hi_lane_u64, sign_bit_i64));
    __m128i result_u64 = _mm_blendv_epi8(hi_lane_u64, max_u64x2, cmp_final);
    return (nk_u64_t)_mm_cvtsi128_si64(result_u64);
}

/** @brief Horizontal min of 8 floats in a YMM register. */
NK_INTERNAL nk_f32_t _nk_reduce_min_f32x8_haswell(__m256 min_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(min_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(min_f32x8, 1);
    __m128 min_f32x4 = _mm_min_ps(lo_f32x4, hi_f32x4);
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_f32x4 = _mm_min_ps(min_f32x4, _mm_shuffle_ps(min_f32x4, min_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(min_f32x4);
}

/** @brief Horizontal max of 8 floats in a YMM register. */
NK_INTERNAL nk_f32_t _nk_reduce_max_f32x8_haswell(__m256 max_f32x8) {
    __m128 lo_f32x4 = _mm256_castps256_ps128(max_f32x8);
    __m128 hi_f32x4 = _mm256_extractf128_ps(max_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(lo_f32x4, hi_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(max_f32x4);
}

/** @brief Horizontal min of 4 doubles in a YMM register. */
NK_INTERNAL nk_f64_t _nk_reduce_min_f64x4_haswell(__m256d min_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(min_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(min_f64x4, 1);
    __m128d min_f64x2 = _mm_min_pd(lo_f64x2, hi_f64x2);
    min_f64x2 = _mm_min_pd(min_f64x2, _mm_shuffle_pd(min_f64x2, min_f64x2, 1));
    return _mm_cvtsd_f64(min_f64x2);
}

/** @brief Horizontal max of 4 doubles in a YMM register. */
NK_INTERNAL nk_f64_t _nk_reduce_max_f64x4_haswell(__m256d max_f64x4) {
    __m128d lo_f64x2 = _mm256_castpd256_pd128(max_f64x4);
    __m128d hi_f64x2 = _mm256_extractf128_pd(max_f64x4, 1);
    __m128d max_f64x2 = _mm_max_pd(lo_f64x2, hi_f64x2);
    max_f64x2 = _mm_max_pd(max_f64x2, _mm_shuffle_pd(max_f64x2, max_f64x2, 1));
    return _mm_cvtsd_f64(max_f64x2);
}

/**
 *  @brief Returns AVX2 blend mask for strided access of 32-bit elements (8-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Returns -1 (all bits set) for positions to keep, 0 for positions to blend away.
 *  Use with _mm256_blendv_ps(identity, data, mask) where identity is 0/+inf/-inf.
 */
NK_INTERNAL __m256i _nk_stride_blend_b32x8(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0); // 4 elems
    case 3: return _mm256_setr_epi32(-1, 0, 0, -1, 0, 0, -1, 0);  // 3 elems
    case 4: return _mm256_setr_epi32(-1, 0, 0, 0, -1, 0, 0, 0);   // 2 elems
    case 5: return _mm256_setr_epi32(-1, 0, 0, 0, 0, -1, 0, 0);   // 2 elems
    case 6: return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, -1, 0);   // 2 elems
    case 7: return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, -1);   // 2 elems
    case 8: return _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0);    // 1 elem
    default: return _mm256_setzero_si256();
    }
}

/**
 *  @brief Returns AVX2 blend mask for strided access of 64-bit elements (4-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Returns -1 (all bits set) for positions to keep, 0 for positions to blend away.
 *  Use with _mm256_blendv_pd(identity, data, mask) where identity is 0/+inf/-inf.
 */
NK_INTERNAL __m256i _nk_stride_blend_b64x4(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm256_setr_epi64x(-1, 0, -1, 0); // 2 elems
    case 3: return _mm256_setr_epi64x(-1, 0, 0, -1); // 2 elems (wraps)
    case 4: return _mm256_setr_epi64x(-1, 0, 0, 0);  // 1 elem
    default: return _mm256_setr_epi64x(-1, 0, 0, 0); // 1 elem for stride 5+
    }
}

#pragma endregion // x86 Haswell Internal Helpers

#pragma region x86 Haswell Public Implementations

NK_INTERNAL void _nk_reduce_add_f32_haswell_contiguous( //
    nk_f32_t const *data, nk_size_t count, nk_f64_t *result) {
    // Accumulate in f64 for precision
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        __m128 lo_f32x4 = _mm256_castps256_ps128(data_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(data_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }
    nk_f64_t sum = _nk_reduce_add_f64x4_haswell(sum_f64x4);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_f32_haswell_strided(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Blend-based strided access: load full register, blend with zero, sum all
    __m256i blend_mask_i32x8 = _nk_stride_blend_b32x8(stride_elements);
    __m256 zero_f32x8 = _mm256_setzero_ps();
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        // Blend: keep stride elements, replace others with zero
        __m256 masked_f32x8 = _mm256_blendv_ps(zero_f32x8, data_f32x8, _mm256_castsi256_ps(blend_mask_i32x8));
        // Sum all - zeros don't contribute
        __m128 lo_f32x4 = _mm256_castps256_ps128(masked_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(masked_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }

    // Scalar tail
    nk_f64_t sum = _nk_reduce_add_f64x4_haswell(sum_f64x4);
    nk_f32_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_f32_haswell_gather(                //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 gathered_f32x8 = _mm256_i32gather_ps(data + idx_scalars * stride_elements, indices_i32x8,
                                                    sizeof(nk_f32_t));
        __m128 lo_f32x4 = _mm256_castps256_ps128(gathered_f32x8);
        __m128 hi_f32x4 = _mm256_extractf128_ps(gathered_f32x8, 1);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(lo_f32x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_cvtps_pd(hi_f32x4));
    }
    nk_f64_t sum = _nk_reduce_add_f64x4_haswell(sum_f64x4);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(nk_f32_t const *)ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f32_haswell(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_add_f32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) _nk_reduce_add_f32_haswell_contiguous(data, count, result);
    else if (stride_elements <= 8) _nk_reduce_add_f32_haswell_strided(data, count, stride_elements, result);
    else _nk_reduce_add_f32_haswell_gather(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_add_f64_haswell_contiguous( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        sum_f64x4 = _mm256_add_pd(sum_f64x4, data_f64x4);
    }
    nk_f64_t sum = _nk_reduce_add_f64x4_haswell(sum_f64x4);
    for (; idx_scalars < count; ++idx_scalars) sum += data[idx_scalars];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_f64_haswell_strided(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Blend-based strided access: load full register, blend with zero, sum all
    __m256i blend_mask_i64x4 = _nk_stride_blend_b64x4(stride_elements);
    __m256d zero_f64x4 = _mm256_setzero_pd();
    __m256d sum_f64x4 = _mm256_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with zero
        __m256d masked_f64x4 = _mm256_blendv_pd(zero_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        sum_f64x4 = _mm256_add_pd(sum_f64x4, masked_f64x4);
    }

    // Scalar tail
    nk_f64_t sum = _nk_reduce_add_f64x4_haswell(sum_f64x4);
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) sum += *ptr;
    *result = sum;
}

NK_PUBLIC void nk_reduce_add_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_add_f64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) _nk_reduce_add_f64_haswell_contiguous(data, count, result);
    else if (stride_elements <= 4) _nk_reduce_add_f64_haswell_strided(data, count, stride_elements, result);
    else nk_reduce_add_f64_serial(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_min_f32_haswell_contiguous( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // First pass: find minimum value
    __m256 min_f32x8 = _mm256_loadu_ps(data);
    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        min_f32x8 = _mm256_min_ps(min_f32x8, data_f32x8);
    }
    nk_f32_t min_val = _nk_reduce_min_f32x8_haswell(min_f32x8);
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

NK_INTERNAL void _nk_reduce_min_f32_haswell_strided(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Blend-based strided access: load full register, blend with +inf, find min
    __m256i blend_mask_i32x8 = _nk_stride_blend_b32x8(stride_elements);
    __m256 pos_inf_f32x8 = _mm256_set1_ps(__builtin_huge_valf());
    __m256 min_f32x8 = pos_inf_f32x8;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        // Blend: keep stride elements, replace others with +inf
        __m256 masked_f32x8 = _mm256_blendv_ps(pos_inf_f32x8, data_f32x8, _mm256_castsi256_ps(blend_mask_i32x8));
        min_f32x8 = _mm256_min_ps(min_f32x8, masked_f32x8);
    }

    // Scalar tail + horizontal reduce
    nk_f32_t min_val = _nk_reduce_min_f32x8_haswell(min_f32x8);
    nk_f32_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr < min_val) min_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != min_val) continue;
        *min_value = min_val;
        *min_index = i;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_f32_haswell(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8)
        _nk_reduce_min_f32_haswell_contiguous(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        _nk_reduce_min_f32_haswell_strided(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void _nk_reduce_max_f32_haswell_contiguous( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    __m256 max_f32x8 = _mm256_loadu_ps(data);
    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        max_f32x8 = _mm256_max_ps(max_f32x8, data_f32x8);
    }
    nk_f32_t max_val = _nk_reduce_max_f32x8_haswell(max_f32x8);
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

NK_INTERNAL void _nk_reduce_max_f32_haswell_strided(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Blend-based strided access: load full register, blend with -inf, find max
    __m256i blend_mask_i32x8 = _nk_stride_blend_b32x8(stride_elements);
    __m256 neg_inf_f32x8 = _mm256_set1_ps(-__builtin_huge_valf());
    __m256 max_f32x8 = neg_inf_f32x8;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m256 data_f32x8 = _mm256_loadu_ps(data + idx_scalars);
        // Blend: keep stride elements, replace others with -inf
        __m256 masked_f32x8 = _mm256_blendv_ps(neg_inf_f32x8, data_f32x8, _mm256_castsi256_ps(blend_mask_i32x8));
        max_f32x8 = _mm256_max_ps(max_f32x8, masked_f32x8);
    }

    // Scalar tail + horizontal reduce
    nk_f32_t max_val = _nk_reduce_max_f32x8_haswell(max_f32x8);
    nk_f32_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr > max_val) max_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != max_val) continue;
        *max_value = max_val;
        *max_index = i;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_f32_haswell(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8)
        _nk_reduce_max_f32_haswell_contiguous(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        _nk_reduce_max_f32_haswell_strided(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void _nk_reduce_min_f64_haswell_contiguous( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *min_value, nk_size_t *min_index) {
    __m256d min_f64x4 = _mm256_loadu_pd(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        min_f64x4 = _mm256_min_pd(min_f64x4, data_f64x4);
    }
    nk_f64_t min_val = _nk_reduce_min_f64x4_haswell(min_f64x4);
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

NK_INTERNAL void _nk_reduce_min_f64_haswell_strided(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Blend-based strided access: load full register, blend with +inf, find min
    __m256i blend_mask_i64x4 = _nk_stride_blend_b64x4(stride_elements);
    __m256d pos_inf_f64x4 = _mm256_set1_pd(__builtin_huge_val());
    __m256d min_f64x4 = pos_inf_f64x4;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with +inf
        __m256d masked_f64x4 = _mm256_blendv_pd(pos_inf_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        min_f64x4 = _mm256_min_pd(min_f64x4, masked_f64x4);
    }

    // Scalar tail + horizontal reduce
    nk_f64_t min_val = _nk_reduce_min_f64x4_haswell(min_f64x4);
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr < min_val) min_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != min_val) continue;
        *min_value = min_val;
        *min_index = i;
        return;
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_PUBLIC void nk_reduce_min_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 4)
        _nk_reduce_min_f64_haswell_contiguous(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        _nk_reduce_min_f64_haswell_strided(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void _nk_reduce_max_f64_haswell_contiguous( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *max_value, nk_size_t *max_index) {
    __m256d max_f64x4 = _mm256_loadu_pd(data);
    nk_size_t idx_scalars = 4;
    for (; idx_scalars + 4 <= count; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        max_f64x4 = _mm256_max_pd(max_f64x4, data_f64x4);
    }
    nk_f64_t max_val = _nk_reduce_max_f64x4_haswell(max_f64x4);
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

NK_INTERNAL void _nk_reduce_max_f64_haswell_strided(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Blend-based strided access: load full register, blend with -inf, find max
    __m256i blend_mask_i64x4 = _nk_stride_blend_b64x4(stride_elements);
    __m256d neg_inf_f64x4 = _mm256_set1_pd(-__builtin_huge_val());
    __m256d max_f64x4 = neg_inf_f64x4;
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    for (; idx_scalars + 4 <= total_scalars; idx_scalars += 4) {
        __m256d data_f64x4 = _mm256_loadu_pd(data + idx_scalars);
        // Blend: keep stride elements, replace others with -inf
        __m256d masked_f64x4 = _mm256_blendv_pd(neg_inf_f64x4, data_f64x4, _mm256_castsi256_pd(blend_mask_i64x4));
        max_f64x4 = _mm256_max_pd(max_f64x4, masked_f64x4);
    }

    // Scalar tail + horizontal reduce
    nk_f64_t max_val = _nk_reduce_max_f64x4_haswell(max_f64x4);
    nk_f64_t const *ptr = data + idx_scalars;
    nk_size_t remaining = count - idx_scalars / stride_elements;
    for (nk_size_t i = 0; i < remaining; ++i, ptr += stride_elements) {
        if (*ptr > max_val) max_val = *ptr;
    }

    // Second pass: find first index (logical index in column)
    ptr = data;
    for (nk_size_t i = 0; i < count; ++i, ptr += stride_elements) {
        if (*ptr != max_val) continue;
        *max_value = max_val;
        *max_index = i;
        return;
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_PUBLIC void nk_reduce_max_f64_haswell(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 4)
        _nk_reduce_max_f64_haswell_contiguous(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 4)
        _nk_reduce_max_f64_haswell_strided(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

// Integer add reductions with widening accumulation

NK_INTERNAL void _nk_reduce_add_i8_haswell_contiguous( //
    nk_i8_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // Widen lower 16 bytes: i8 -> i16 -> i32 -> i64
        __m128i lo_i8x16 = _mm256_castsi256_si128(data_i8x32);
        __m256i lo_i16x16 = _mm256_cvtepi8_epi16(lo_i8x16);
        __m128i lo_lo_i16x8 = _mm256_castsi256_si128(lo_i16x16);
        __m128i lo_hi_i16x8 = _mm256_extracti128_si256(lo_i16x16, 1);
        __m256i lo_lo_i32x8 = _mm256_cvtepi16_epi32(lo_lo_i16x8);
        __m256i lo_hi_i32x8 = _mm256_cvtepi16_epi32(lo_hi_i16x8);
        __m128i a_i32x4 = _mm256_castsi256_si128(lo_lo_i32x8);
        __m128i b_i32x4 = _mm256_extracti128_si256(lo_lo_i32x8, 1);
        __m128i c_i32x4 = _mm256_castsi256_si128(lo_hi_i32x8);
        __m128i d_i32x4 = _mm256_extracti128_si256(lo_hi_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(a_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(b_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(c_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(d_i32x4));
        // Widen upper 16 bytes
        __m128i hi_i8x16 = _mm256_extracti128_si256(data_i8x32, 1);
        __m256i hi_i16x16 = _mm256_cvtepi8_epi16(hi_i8x16);
        __m128i hi_lo_i16x8 = _mm256_castsi256_si128(hi_i16x16);
        __m128i hi_hi_i16x8 = _mm256_extracti128_si256(hi_i16x16, 1);
        __m256i hi_lo_i32x8 = _mm256_cvtepi16_epi32(hi_lo_i16x8);
        __m256i hi_hi_i32x8 = _mm256_cvtepi16_epi32(hi_hi_i16x8);
        __m128i e_i32x4 = _mm256_castsi256_si128(hi_lo_i32x8);
        __m128i f_i32x4 = _mm256_extracti128_si256(hi_lo_i32x8, 1);
        __m128i g_i32x4 = _mm256_castsi256_si128(hi_hi_i32x8);
        __m128i h_i32x4 = _mm256_extracti128_si256(hi_hi_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(e_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(f_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(g_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(h_i32x4));
    }
    nk_i64_t sum = _nk_reduce_add_i64x4_haswell(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_u8_haswell_contiguous( //
    nk_u8_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_u8x16 = _mm256_castsi256_si128(data_u8x32);
        __m256i lo_u16x16 = _mm256_cvtepu8_epi16(lo_u8x16);
        __m128i lo_lo_u16x8 = _mm256_castsi256_si128(lo_u16x16);
        __m128i lo_hi_u16x8 = _mm256_extracti128_si256(lo_u16x16, 1);
        __m256i lo_lo_u32x8 = _mm256_cvtepu16_epi32(lo_lo_u16x8);
        __m256i lo_hi_u32x8 = _mm256_cvtepu16_epi32(lo_hi_u16x8);
        __m128i a_u32x4 = _mm256_castsi256_si128(lo_lo_u32x8);
        __m128i b_u32x4 = _mm256_extracti128_si256(lo_lo_u32x8, 1);
        __m128i c_u32x4 = _mm256_castsi256_si128(lo_hi_u32x8);
        __m128i d_u32x4 = _mm256_extracti128_si256(lo_hi_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(a_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(b_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(c_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(d_u32x4));
        __m128i hi_u8x16 = _mm256_extracti128_si256(data_u8x32, 1);
        __m256i hi_u16x16 = _mm256_cvtepu8_epi16(hi_u8x16);
        __m128i hi_lo_u16x8 = _mm256_castsi256_si128(hi_u16x16);
        __m128i hi_hi_u16x8 = _mm256_extracti128_si256(hi_u16x16, 1);
        __m256i hi_lo_u32x8 = _mm256_cvtepu16_epi32(hi_lo_u16x8);
        __m256i hi_hi_u32x8 = _mm256_cvtepu16_epi32(hi_hi_u16x8);
        __m128i e_u32x4 = _mm256_castsi256_si128(hi_lo_u32x8);
        __m128i f_u32x4 = _mm256_extracti128_si256(hi_lo_u32x8, 1);
        __m128i g_u32x4 = _mm256_castsi256_si128(hi_hi_u32x8);
        __m128i h_u32x4 = _mm256_extracti128_si256(hi_hi_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(e_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(f_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(g_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(h_u32x4));
    }
    nk_u64_t sum = (nk_u64_t)_nk_reduce_add_i64x4_haswell(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_i16_haswell_contiguous( //
    nk_i16_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_i16x8 = _mm256_castsi256_si128(data_i16x16);
        __m128i hi_i16x8 = _mm256_extracti128_si256(data_i16x16, 1);
        __m256i lo_i32x8 = _mm256_cvtepi16_epi32(lo_i16x8);
        __m256i hi_i32x8 = _mm256_cvtepi16_epi32(hi_i16x8);
        __m128i a_i32x4 = _mm256_castsi256_si128(lo_i32x8);
        __m128i b_i32x4 = _mm256_extracti128_si256(lo_i32x8, 1);
        __m128i c_i32x4 = _mm256_castsi256_si128(hi_i32x8);
        __m128i d_i32x4 = _mm256_extracti128_si256(hi_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(a_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(b_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(c_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(d_i32x4));
    }
    nk_i64_t sum = _nk_reduce_add_i64x4_haswell(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_u16_haswell_contiguous( //
    nk_u16_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_u16x8 = _mm256_castsi256_si128(data_u16x16);
        __m128i hi_u16x8 = _mm256_extracti128_si256(data_u16x16, 1);
        __m256i lo_u32x8 = _mm256_cvtepu16_epi32(lo_u16x8);
        __m256i hi_u32x8 = _mm256_cvtepu16_epi32(hi_u16x8);
        __m128i a_u32x4 = _mm256_castsi256_si128(lo_u32x8);
        __m128i b_u32x4 = _mm256_extracti128_si256(lo_u32x8, 1);
        __m128i c_u32x4 = _mm256_castsi256_si128(hi_u32x8);
        __m128i d_u32x4 = _mm256_extracti128_si256(hi_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(a_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(b_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(c_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(d_u32x4));
    }
    nk_u64_t sum = (nk_u64_t)_nk_reduce_add_i64x4_haswell(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_i32_haswell_contiguous( //
    nk_i32_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_i32x4 = _mm256_castsi256_si128(data_i32x8);
        __m128i hi_i32x4 = _mm256_extracti128_si256(data_i32x8, 1);
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(lo_i32x4));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, _mm256_cvtepi32_epi64(hi_i32x4));
    }
    nk_i64_t sum = _nk_reduce_add_i64x4_haswell(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_u32_haswell_contiguous( //
    nk_u32_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_u32x4 = _mm256_castsi256_si128(data_u32x8);
        __m128i hi_u32x4 = _mm256_extracti128_si256(data_u32x8, 1);
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(lo_u32x4));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, _mm256_cvtepu32_epi64(hi_u32x4));
    }
    nk_u64_t sum = (nk_u64_t)_nk_reduce_add_i64x4_haswell(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_i64_haswell_contiguous( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *result) {
    __m256i sum_i64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_i64x4 = _mm256_add_epi64(sum_i64x4, data_i64x4);
    }
    nk_i64_t sum = _nk_reduce_add_i64x4_haswell(sum_i64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_u64_haswell_contiguous( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *result) {
    __m256i sum_u64x4 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        sum_u64x4 = _mm256_add_epi64(sum_u64x4, data_u64x4);
    }
    nk_u64_t sum = (nk_u64_t)_nk_reduce_add_i64x4_haswell(sum_u64x4);
    for (; idx < count; ++idx) sum += data[idx];
    *result = sum;
}

// Integer min/max contiguous implementations

NK_INTERNAL void _nk_reduce_min_i8_haswell_contiguous( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *min_value, nk_size_t *min_index) {
    __m256i min_i8x32 = _mm256_set1_epi8(127);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_i8x32 = _mm256_min_epi8(min_i8x32, data_i8x32);
    }
    nk_i8_t min_val = _nk_reduce_min_i8x32_haswell(min_i8x32);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    // Second pass for index
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_i8_haswell_contiguous( //
    nk_i8_t const *data, nk_size_t count, nk_i8_t *max_value, nk_size_t *max_index) {
    __m256i max_i8x32 = _mm256_set1_epi8(-128);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_i8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_i8x32 = _mm256_max_epi8(max_i8x32, data_i8x32);
    }
    nk_i8_t max_val = _nk_reduce_max_i8x32_haswell(max_i8x32);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_u8_haswell_contiguous( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *min_value, nk_size_t *min_index) {
    __m256i min_u8x32 = _mm256_set1_epi8((char)255);
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_u8x32 = _mm256_min_epu8(min_u8x32, data_u8x32);
    }
    nk_u8_t min_val = _nk_reduce_min_u8x32_haswell(min_u8x32);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_u8_haswell_contiguous( //
    nk_u8_t const *data, nk_size_t count, nk_u8_t *max_value, nk_size_t *max_index) {
    __m256i max_u8x32 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 32 <= count; idx += 32) {
        __m256i data_u8x32 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_u8x32 = _mm256_max_epu8(max_u8x32, data_u8x32);
    }
    nk_u8_t max_val = _nk_reduce_max_u8x32_haswell(max_u8x32);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_i16_haswell_contiguous( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *min_value, nk_size_t *min_index) {
    __m256i min_i16x16 = _mm256_set1_epi16(32767);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_i16x16 = _mm256_min_epi16(min_i16x16, data_i16x16);
    }
    nk_i16_t min_val = _nk_reduce_min_i16x16_haswell(min_i16x16);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_i16_haswell_contiguous( //
    nk_i16_t const *data, nk_size_t count, nk_i16_t *max_value, nk_size_t *max_index) {
    __m256i max_i16x16 = _mm256_set1_epi16(-32768);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_i16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_i16x16 = _mm256_max_epi16(max_i16x16, data_i16x16);
    }
    nk_i16_t max_val = _nk_reduce_max_i16x16_haswell(max_i16x16);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_u16_haswell_contiguous( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *min_value, nk_size_t *min_index) {
    __m256i min_u16x16 = _mm256_set1_epi16((short)65535);
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_u16x16 = _mm256_min_epu16(min_u16x16, data_u16x16);
    }
    nk_u16_t min_val = _nk_reduce_min_u16x16_haswell(min_u16x16);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_u16_haswell_contiguous( //
    nk_u16_t const *data, nk_size_t count, nk_u16_t *max_value, nk_size_t *max_index) {
    __m256i max_u16x16 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 16 <= count; idx += 16) {
        __m256i data_u16x16 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_u16x16 = _mm256_max_epu16(max_u16x16, data_u16x16);
    }
    nk_u16_t max_val = _nk_reduce_max_u16x16_haswell(max_u16x16);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_i32_haswell_contiguous( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *min_value, nk_size_t *min_index) {
    __m256i min_i32x8 = _mm256_set1_epi32(0x7FFFFFFF);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_i32x8 = _mm256_min_epi32(min_i32x8, data_i32x8);
    }
    nk_i32_t min_val = _nk_reduce_min_i32x8_haswell(min_i32x8);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_i32_haswell_contiguous( //
    nk_i32_t const *data, nk_size_t count, nk_i32_t *max_value, nk_size_t *max_index) {
    __m256i max_i32x8 = _mm256_set1_epi32((nk_i32_t)0x80000000);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_i32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_i32x8 = _mm256_max_epi32(max_i32x8, data_i32x8);
    }
    nk_i32_t max_val = _nk_reduce_max_i32x8_haswell(max_i32x8);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_u32_haswell_contiguous( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *min_value, nk_size_t *min_index) {
    __m256i min_u32x8 = _mm256_set1_epi32((nk_i32_t)0xFFFFFFFF);
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        min_u32x8 = _mm256_min_epu32(min_u32x8, data_u32x8);
    }
    nk_u32_t min_val = _nk_reduce_min_u32x8_haswell(min_u32x8);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_u32_haswell_contiguous( //
    nk_u32_t const *data, nk_size_t count, nk_u32_t *max_value, nk_size_t *max_index) {
    __m256i max_u32x8 = _mm256_setzero_si256();
    nk_size_t idx = 0;
    for (; idx + 8 <= count; idx += 8) {
        __m256i data_u32x8 = _mm256_loadu_si256((__m256i const *)(data + idx));
        max_u32x8 = _mm256_max_epu32(max_u32x8, data_u32x8);
    }
    nk_u32_t max_val = _nk_reduce_max_u32x8_haswell(max_u32x8);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_i64_haswell_contiguous( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *min_value, nk_size_t *min_index) {
    __m256i min_i64x4 = _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL);
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // Manual 64-bit signed min using comparison
        __m128i lo_data_i64x2 = _mm256_castsi256_si128(data_i64x4);
        __m128i hi_data_i64x2 = _mm256_extracti128_si256(data_i64x4, 1);
        __m128i lo_min_i64x2 = _mm256_castsi256_si128(min_i64x4);
        __m128i hi_min_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(lo_min_i64x2, lo_data_i64x2);
        __m128i hi_cmp = _mm_cmpgt_epi64(hi_min_i64x2, hi_data_i64x2);
        lo_min_i64x2 = _mm_blendv_epi8(lo_min_i64x2, lo_data_i64x2, lo_cmp);
        hi_min_i64x2 = _mm_blendv_epi8(hi_min_i64x2, hi_data_i64x2, hi_cmp);
        min_i64x4 = _mm256_setr_m128i(lo_min_i64x2, hi_min_i64x2);
    }
    nk_i64_t min_val = _nk_reduce_min_i64x4_haswell(min_i64x4);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_i64_haswell_contiguous( //
    nk_i64_t const *data, nk_size_t count, nk_i64_t *max_value, nk_size_t *max_index) {
    __m256i max_i64x4 = _mm256_set1_epi64x((nk_i64_t)0x8000000000000000LL);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_i64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_data_i64x2 = _mm256_castsi256_si128(data_i64x4);
        __m128i hi_data_i64x2 = _mm256_extracti128_si256(data_i64x4, 1);
        __m128i lo_max_i64x2 = _mm256_castsi256_si128(max_i64x4);
        __m128i hi_max_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(lo_data_i64x2, lo_max_i64x2);
        __m128i hi_cmp = _mm_cmpgt_epi64(hi_data_i64x2, hi_max_i64x2);
        lo_max_i64x2 = _mm_blendv_epi8(lo_max_i64x2, lo_data_i64x2, lo_cmp);
        hi_max_i64x2 = _mm_blendv_epi8(hi_max_i64x2, hi_data_i64x2, hi_cmp);
        max_i64x4 = _mm256_setr_m128i(lo_max_i64x2, hi_max_i64x2);
    }
    nk_i64_t max_val = _nk_reduce_max_i64x4_haswell(max_i64x4);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

NK_INTERNAL void _nk_reduce_min_u64_haswell_contiguous( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *min_value, nk_size_t *min_index) {
    __m256i min_u64x4 = _mm256_set1_epi64x((nk_i64_t)0xFFFFFFFFFFFFFFFFULL);
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        // Unsigned comparison via XOR with sign bit
        __m128i lo_data_u64x2 = _mm256_castsi256_si128(data_u64x4);
        __m128i hi_data_u64x2 = _mm256_extracti128_si256(data_u64x4, 1);
        __m128i lo_min_u64x2 = _mm256_castsi256_si128(min_u64x4);
        __m128i hi_min_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(_mm_xor_si128(lo_min_u64x2, sign_bit_i64),
                                         _mm_xor_si128(lo_data_u64x2, sign_bit_i64));
        __m128i hi_cmp = _mm_cmpgt_epi64(_mm_xor_si128(hi_min_u64x2, sign_bit_i64),
                                         _mm_xor_si128(hi_data_u64x2, sign_bit_i64));
        lo_min_u64x2 = _mm_blendv_epi8(lo_min_u64x2, lo_data_u64x2, lo_cmp);
        hi_min_u64x2 = _mm_blendv_epi8(hi_min_u64x2, hi_data_u64x2, hi_cmp);
        min_u64x4 = _mm256_setr_m128i(lo_min_u64x2, hi_min_u64x2);
    }
    nk_u64_t min_val = _nk_reduce_min_u64x4_haswell(min_u64x4);
    for (; idx < count; ++idx) min_val = data[idx] < min_val ? data[idx] : min_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == min_val) {
            *min_value = min_val;
            *min_index = idx;
            return;
        }
    }
    *min_value = min_val;
    *min_index = 0;
}

NK_INTERNAL void _nk_reduce_max_u64_haswell_contiguous( //
    nk_u64_t const *data, nk_size_t count, nk_u64_t *max_value, nk_size_t *max_index) {
    __m256i max_u64x4 = _mm256_setzero_si256();
    __m128i sign_bit_i64 = _mm_set1_epi64x((nk_i64_t)0x8000000000000000ull);
    nk_size_t idx = 0;
    for (; idx + 4 <= count; idx += 4) {
        __m256i data_u64x4 = _mm256_loadu_si256((__m256i const *)(data + idx));
        __m128i lo_data_u64x2 = _mm256_castsi256_si128(data_u64x4);
        __m128i hi_data_u64x2 = _mm256_extracti128_si256(data_u64x4, 1);
        __m128i lo_max_u64x2 = _mm256_castsi256_si128(max_u64x4);
        __m128i hi_max_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
        __m128i lo_cmp = _mm_cmpgt_epi64(_mm_xor_si128(lo_data_u64x2, sign_bit_i64),
                                         _mm_xor_si128(lo_max_u64x2, sign_bit_i64));
        __m128i hi_cmp = _mm_cmpgt_epi64(_mm_xor_si128(hi_data_u64x2, sign_bit_i64),
                                         _mm_xor_si128(hi_max_u64x2, sign_bit_i64));
        lo_max_u64x2 = _mm_blendv_epi8(lo_max_u64x2, lo_data_u64x2, lo_cmp);
        hi_max_u64x2 = _mm_blendv_epi8(hi_max_u64x2, hi_data_u64x2, hi_cmp);
        max_u64x4 = _mm256_setr_m128i(lo_max_u64x2, hi_max_u64x2);
    }
    nk_u64_t max_val = _nk_reduce_max_u64x4_haswell(max_u64x4);
    for (; idx < count; ++idx) max_val = data[idx] > max_val ? data[idx] : max_val;
    for (idx = 0; idx < count; ++idx) {
        if (data[idx] == max_val) {
            *max_value = max_val;
            *max_index = idx;
            return;
        }
    }
    *max_value = max_val;
    *max_index = 0;
}

// Integer public dispatchers

NK_PUBLIC void nk_reduce_add_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i8_t)) _nk_reduce_add_i8_haswell_contiguous(data, count, result);
    else nk_reduce_add_i8_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u8_t)) _nk_reduce_add_u8_haswell_contiguous(data, count, result);
    else nk_reduce_add_u8_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i16_t)) _nk_reduce_add_i16_haswell_contiguous(data, count, result);
    else nk_reduce_add_i16_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u16_t)) _nk_reduce_add_u16_haswell_contiguous(data, count, result);
    else nk_reduce_add_u16_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i32_t)) _nk_reduce_add_i32_haswell_contiguous(data, count, result);
    else nk_reduce_add_i32_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u32_t)) _nk_reduce_add_u32_haswell_contiguous(data, count, result);
    else nk_reduce_add_u32_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *result) {
    if (stride_bytes == sizeof(nk_i64_t)) _nk_reduce_add_i64_haswell_contiguous(data, count, result);
    else nk_reduce_add_i64_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_add_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *result) {
    if (stride_bytes == sizeof(nk_u64_t)) _nk_reduce_add_u64_haswell_contiguous(data, count, result);
    else nk_reduce_add_u64_serial(data, count, stride_bytes, result);
}

NK_PUBLIC void nk_reduce_min_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i8_t)) _nk_reduce_min_i8_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_i8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i8_haswell(                          //
    nk_i8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i8_t)) _nk_reduce_max_i8_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_i8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u8_t)) _nk_reduce_min_u8_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_u8_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u8_haswell(                          //
    nk_u8_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u8_t)) _nk_reduce_max_u8_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_u8_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i16_t)) _nk_reduce_min_i16_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_i16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i16_haswell(                          //
    nk_i16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i16_t)) _nk_reduce_max_i16_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_i16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u16_t)) _nk_reduce_min_u16_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_u16_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u16_haswell(                          //
    nk_u16_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u16_t)) _nk_reduce_max_u16_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_u16_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i32_t)) _nk_reduce_min_i32_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_i32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i32_haswell(                          //
    nk_i32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i32_t)) _nk_reduce_max_i32_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_i32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u32_t)) _nk_reduce_min_u32_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_u32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u32_haswell(                          //
    nk_u32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u32_t)) _nk_reduce_max_u32_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_u32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_i64_t)) _nk_reduce_min_i64_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_i64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_i64_haswell(                          //
    nk_i64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_i64_t)) _nk_reduce_max_i64_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_i64_serial(data, count, stride_bytes, max_value, max_index);
}

NK_PUBLIC void nk_reduce_min_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value, nk_size_t *min_index) {
    if (stride_bytes == sizeof(nk_u64_t)) _nk_reduce_min_u64_haswell_contiguous(data, count, min_value, min_index);
    else nk_reduce_min_u64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_PUBLIC void nk_reduce_max_u64_haswell(                          //
    nk_u64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *max_value, nk_size_t *max_index) {
    if (stride_bytes == sizeof(nk_u64_t)) _nk_reduce_max_u64_haswell_contiguous(data, count, max_value, max_index);
    else nk_reduce_max_u64_serial(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion // x86 Haswell Public Implementations

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL

#pragma endregion // x86 Haswell Implementations

#pragma region x86 Skylake Implementations

#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#pragma region x86 Skylake Internal Helpers

/** @brief Horizontal sum of 16 floats in a ZMM register (native f32 precision). */
NK_INTERNAL nk_f32_t _nk_reduce_add_f32x16_skylake(__m512 sum_f32x16) {
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
NK_INTERNAL nk_f64_t _nk_reduce_add_f64x8_skylake(__m512d sum_f64x8) {
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
NK_INTERNAL nk_f32_t _nk_reduce_min_f32x16_skylake(__m512 min_f32x16) {
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
NK_INTERNAL nk_f32_t _nk_reduce_max_f32x16_skylake(__m512 max_f32x16) {
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
NK_INTERNAL nk_f64_t _nk_reduce_min_f64x8_skylake(__m512d min_f64x8) {
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
NK_INTERNAL nk_f64_t _nk_reduce_max_f64x8_skylake(__m512d max_f64x8) {
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
NK_INTERNAL nk_i32_t _nk_reduce_add_i32x16_skylake(__m512i sum_i32x16) {
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
NK_INTERNAL nk_i64_t _nk_reduce_add_i64x8_skylake(__m512i sum_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(sum_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(sum_i64x8, 1);
    __m256i sum_i64x4 = _mm256_add_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    sum_i64x2 = _mm_add_epi64(sum_i64x2, _mm_shuffle_epi32(sum_i64x2, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si64(sum_i64x2);
}

/**
 *  @brief Returns AVX-512 mask for strided access of 8-bit elements (64-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  With 64 elements per register, useful for strides 2-16 (yielding 4+ elements per load).
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask64 _nk_stride_mask_b8x64(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask64)0x5555555555555555ull;  // 32 elems
    case 3: return (__mmask64)0x1249249249249249ull;  // 21 elems
    case 4: return (__mmask64)0x1111111111111111ull;  // 16 elems
    case 5: return (__mmask64)0x1084210842108421ull;  // 12 elems
    case 6: return (__mmask64)0x1041041041041041ull;  // 10 elems
    case 7: return (__mmask64)0x0408102040810204ull;  // 9 elems
    case 8: return (__mmask64)0x0101010101010101ull;  // 8 elems
    case 9: return (__mmask64)0x0080200802008020ull;  // 7 elems
    case 10: return (__mmask64)0x0040100401004010ull; // 6 elems
    case 11: return (__mmask64)0x0020080200802008ull; // 5 elems
    case 12: return (__mmask64)0x0010040100401004ull; // 5 elems
    case 13: return (__mmask64)0x0008020080200802ull; // 4 elems
    case 14: return (__mmask64)0x0004010040100401ull; // 4 elems
    case 15: return (__mmask64)0x0002008020080200ull; // 4 elems
    case 16: return (__mmask64)0x0001000100010001ull; // 4 elems
    default: return (__mmask64)0;
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 32-bit elements (16-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask16 _nk_stride_mask_b32x16(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask16)0x5555; // [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0] → 8 elems
    case 3: return (__mmask16)0x1249; // [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0] → 5 elems
    case 4: return (__mmask16)0x1111; // [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0] → 4 elems
    case 5: return (__mmask16)0x0421; // [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0] → 3 elems
    case 6: return (__mmask16)0x0041; // [1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0] → 2 elems
    case 7: return (__mmask16)0x0081; // [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0] → 2 elems
    case 8: return (__mmask16)0x0101; // [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0] → 2 elems
    default: return (__mmask16)0;     // Invalid stride - caller should use gather or serial
    }
}

/**
 *  @brief Returns AVX-512 mask for strided access of 64-bit elements (8-element register).
 *
 *  For column extraction from row-major matrices: stride N means every Nth element.
 *  Example: stride 4 extracts column 0 from a 4-column matrix.
 *  Mask bits set to 1 where (position % stride == 0).
 */
NK_INTERNAL __mmask8 _nk_stride_mask_b64x8(nk_size_t stride) {
    switch (stride) {
    case 2: return (__mmask8)0x55; // [1,0,1,0,1,0,1,0] → 4 elems
    case 3: return (__mmask8)0x49; // [1,0,0,1,0,0,1,0] → 3 elems
    case 4: return (__mmask8)0x11; // [1,0,0,0,1,0,0,0] → 2 elems
    case 5: return (__mmask8)0x21; // [1,0,0,0,0,1,0,0] → 2 elems
    case 6: return (__mmask8)0x41; // [1,0,0,0,0,0,1,0] → 2 elems
    case 7: return (__mmask8)0x01; // [1,0,0,0,0,0,0,0] → 1 elem
    case 8: return (__mmask8)0x01; // [1,0,0,0,0,0,0,0] → 1 elem
    default: return (__mmask8)0;
    }
}

/**
 *  @brief Returns initial logical index vector for 32-bit strided access (16-element register).
 *
 *  For min/max with index tracking: non-stride positions get 0 (don't matter, masked out).
 *  Stride positions get sequential logical indices: 0, 1, 2, ...
 */
NK_INTERNAL __m512i _nk_stride_logidx_i32x16(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm512_setr_epi32(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0); // 8 elems
    case 3: return _mm512_setr_epi32(0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 0); // 5 elems
    case 4: return _mm512_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0); // 4 elems
    case 5: return _mm512_setr_epi32(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0); // 3 elems
    case 6: return _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0); // 2 elems
    case 7: return _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0); // 2 elems
    case 8: return _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0); // 2 elems
    default: return _mm512_setzero_si512();
    }
}

/**
 *  @brief Returns initial logical index vector for 64-bit strided access (8-element register).
 *
 *  For min/max with index tracking: non-stride positions get 0 (don't matter, masked out).
 *  Stride positions get sequential logical indices: 0, 1, 2, ...
 */
NK_INTERNAL __m512i _nk_stride_logidx_i64x8(nk_size_t stride) {
    switch (stride) {
    case 2: return _mm512_setr_epi64(0, 0, 1, 0, 2, 0, 3, 0); // 4 elems
    case 3: return _mm512_setr_epi64(0, 0, 0, 1, 0, 0, 2, 0); // 3 elems
    case 4: return _mm512_setr_epi64(0, 0, 0, 0, 1, 0, 0, 0); // 2 elems
    case 5: return _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 0, 0); // 2 elems
    case 6: return _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 1, 0); // 2 elems
    case 7: return _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 0, 0); // 1 elem
    case 8: return _mm512_setr_epi64(0, 0, 0, 0, 0, 0, 0, 0); // 1 elem
    default: return _mm512_setzero_si512();
    }
}

/**
 *  @brief Returns number of logical elements per 16-scalar chunk for given stride.
 */
NK_INTERNAL nk_size_t _nk_stride_elems_b32x16(nk_size_t stride) {
    switch (stride) {
    case 2: return 8;
    case 3: return 5;
    case 4: return 4;
    case 5: return 3;
    case 6: return 2;
    case 7: return 2;
    case 8: return 2;
    default: return 0;
    }
}

/**
 *  @brief Returns number of logical elements per 8-scalar chunk for given stride.
 */
NK_INTERNAL nk_size_t _nk_stride_elems_b64x8(nk_size_t stride) {
    switch (stride) {
    case 2: return 4;
    case 3: return 3;
    case 4: return 2;
    case 5: return 2;
    case 6: return 2;
    case 7: return 1;
    case 8: return 1;
    default: return 0;
    }
}

/** @brief Horizontal min of 64 signed i8s in a ZMM register. */
NK_INTERNAL nk_i8_t _nk_reduce_min_i8x64_skylake(__m512i min_i8x64) {
    __m256i lo_i8x32 = _mm512_castsi512_si256(min_i8x64);
    __m256i hi_i8x32 = _mm512_extracti64x4_epi64(min_i8x64, 1);
    __m256i min_i8x32 = _mm256_min_epi8(lo_i8x32, hi_i8x32);
    __m128i lo_i8x16 = _mm256_castsi256_si128(min_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(min_i8x32, 1);
    __m128i min_i8x16 = _mm_min_epi8(lo_i8x16, hi_i8x16);
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shuffle_epi32(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_shufflelo_epi16(min_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i8x16 = _mm_min_epi8(min_i8x16, _mm_srli_epi16(min_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(min_i8x16);
}

/** @brief Horizontal max of 64 signed i8s in a ZMM register. */
NK_INTERNAL nk_i8_t _nk_reduce_max_i8x64_skylake(__m512i max_i8x64) {
    __m256i lo_i8x32 = _mm512_castsi512_si256(max_i8x64);
    __m256i hi_i8x32 = _mm512_extracti64x4_epi64(max_i8x64, 1);
    __m256i max_i8x32 = _mm256_max_epi8(lo_i8x32, hi_i8x32);
    __m128i lo_i8x16 = _mm256_castsi256_si128(max_i8x32);
    __m128i hi_i8x16 = _mm256_extracti128_si256(max_i8x32, 1);
    __m128i max_i8x16 = _mm_max_epi8(lo_i8x16, hi_i8x16);
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shuffle_epi32(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_shufflelo_epi16(max_i8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i8x16 = _mm_max_epi8(max_i8x16, _mm_srli_epi16(max_i8x16, 8));
    return (nk_i8_t)_mm_cvtsi128_si32(max_i8x16);
}

/** @brief Horizontal min of 64 unsigned u8s in a ZMM register. */
NK_INTERNAL nk_u8_t _nk_reduce_min_u8x64_skylake(__m512i min_u8x64) {
    __m256i lo_u8x32 = _mm512_castsi512_si256(min_u8x64);
    __m256i hi_u8x32 = _mm512_extracti64x4_epi64(min_u8x64, 1);
    __m256i min_u8x32 = _mm256_min_epu8(lo_u8x32, hi_u8x32);
    __m128i lo_u8x16 = _mm256_castsi256_si128(min_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(min_u8x32, 1);
    __m128i min_u8x16 = _mm_min_epu8(lo_u8x16, hi_u8x16);
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shuffle_epi32(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_shufflelo_epi16(min_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u8x16 = _mm_min_epu8(min_u8x16, _mm_srli_epi16(min_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(min_u8x16);
}

/** @brief Horizontal max of 64 unsigned u8s in a ZMM register. */
NK_INTERNAL nk_u8_t _nk_reduce_max_u8x64_skylake(__m512i max_u8x64) {
    __m256i lo_u8x32 = _mm512_castsi512_si256(max_u8x64);
    __m256i hi_u8x32 = _mm512_extracti64x4_epi64(max_u8x64, 1);
    __m256i max_u8x32 = _mm256_max_epu8(lo_u8x32, hi_u8x32);
    __m128i lo_u8x16 = _mm256_castsi256_si128(max_u8x32);
    __m128i hi_u8x16 = _mm256_extracti128_si256(max_u8x32, 1);
    __m128i max_u8x16 = _mm_max_epu8(lo_u8x16, hi_u8x16);
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shuffle_epi32(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_shufflelo_epi16(max_u8x16, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u8x16 = _mm_max_epu8(max_u8x16, _mm_srli_epi16(max_u8x16, 8));
    return (nk_u8_t)_mm_cvtsi128_si32(max_u8x16);
}

/** @brief Horizontal min of 32 signed i16s in a ZMM register. */
NK_INTERNAL nk_i16_t _nk_reduce_min_i16x32_skylake(__m512i min_i16x32) {
    __m256i lo_i16x16 = _mm512_castsi512_si256(min_i16x32);
    __m256i hi_i16x16 = _mm512_extracti64x4_epi64(min_i16x32, 1);
    __m256i min_i16x16 = _mm256_min_epi16(lo_i16x16, hi_i16x16);
    __m128i lo_i16x8 = _mm256_castsi256_si128(min_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(min_i16x16, 1);
    __m128i min_i16x8 = _mm_min_epi16(lo_i16x8, hi_i16x8);
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shuffle_epi32(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_i16x8 = _mm_min_epi16(min_i16x8, _mm_shufflelo_epi16(min_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(min_i16x8);
}

/** @brief Horizontal max of 32 signed i16s in a ZMM register. */
NK_INTERNAL nk_i16_t _nk_reduce_max_i16x32_skylake(__m512i max_i16x32) {
    __m256i lo_i16x16 = _mm512_castsi512_si256(max_i16x32);
    __m256i hi_i16x16 = _mm512_extracti64x4_epi64(max_i16x32, 1);
    __m256i max_i16x16 = _mm256_max_epi16(lo_i16x16, hi_i16x16);
    __m128i lo_i16x8 = _mm256_castsi256_si128(max_i16x16);
    __m128i hi_i16x8 = _mm256_extracti128_si256(max_i16x16, 1);
    __m128i max_i16x8 = _mm_max_epi16(lo_i16x8, hi_i16x8);
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shuffle_epi32(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_i16x8 = _mm_max_epi16(max_i16x8, _mm_shufflelo_epi16(max_i16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_i16_t)_mm_cvtsi128_si32(max_i16x8);
}

/** @brief Horizontal min of 32 unsigned u16s in a ZMM register. */
NK_INTERNAL nk_u16_t _nk_reduce_min_u16x32_skylake(__m512i min_u16x32) {
    __m256i lo_u16x16 = _mm512_castsi512_si256(min_u16x32);
    __m256i hi_u16x16 = _mm512_extracti64x4_epi64(min_u16x32, 1);
    __m256i min_u16x16 = _mm256_min_epu16(lo_u16x16, hi_u16x16);
    __m128i lo_u16x8 = _mm256_castsi256_si128(min_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(min_u16x16, 1);
    __m128i min_u16x8 = _mm_min_epu16(lo_u16x8, hi_u16x8);
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shuffle_epi32(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    min_u16x8 = _mm_min_epu16(min_u16x8, _mm_shufflelo_epi16(min_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(min_u16x8);
}

/** @brief Horizontal max of 32 unsigned u16s in a ZMM register. */
NK_INTERNAL nk_u16_t _nk_reduce_max_u16x32_skylake(__m512i max_u16x32) {
    __m256i lo_u16x16 = _mm512_castsi512_si256(max_u16x32);
    __m256i hi_u16x16 = _mm512_extracti64x4_epi64(max_u16x32, 1);
    __m256i max_u16x16 = _mm256_max_epu16(lo_u16x16, hi_u16x16);
    __m128i lo_u16x8 = _mm256_castsi256_si128(max_u16x16);
    __m128i hi_u16x8 = _mm256_extracti128_si256(max_u16x16, 1);
    __m128i max_u16x8 = _mm_max_epu16(lo_u16x8, hi_u16x8);
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shuffle_epi32(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    max_u16x8 = _mm_max_epu16(max_u16x8, _mm_shufflelo_epi16(max_u16x8, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u16_t)_mm_cvtsi128_si32(max_u16x8);
}

/** @brief Horizontal min of 16 signed i32s in a ZMM register. */
NK_INTERNAL nk_i32_t _nk_reduce_min_i32x16_skylake(__m512i min_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(min_i32x16);
    __m256i hi_i32x8 = _mm512_extracti64x4_epi64(min_i32x16, 1);
    __m256i min_i32x8 = _mm256_min_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(min_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(min_i32x8, 1);
    __m128i min_i32x4 = _mm_min_epi32(lo_i32x4, hi_i32x4);
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_i32x4 = _mm_min_epi32(min_i32x4, _mm_shuffle_epi32(min_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(min_i32x4);
}

/** @brief Horizontal max of 16 signed i32s in a ZMM register. */
NK_INTERNAL nk_i32_t _nk_reduce_max_i32x16_skylake(__m512i max_i32x16) {
    __m256i lo_i32x8 = _mm512_castsi512_si256(max_i32x16);
    __m256i hi_i32x8 = _mm512_extracti64x4_epi64(max_i32x16, 1);
    __m256i max_i32x8 = _mm256_max_epi32(lo_i32x8, hi_i32x8);
    __m128i lo_i32x4 = _mm256_castsi256_si128(max_i32x8);
    __m128i hi_i32x4 = _mm256_extracti128_si256(max_i32x8, 1);
    __m128i max_i32x4 = _mm_max_epi32(lo_i32x4, hi_i32x4);
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_i32x4 = _mm_max_epi32(max_i32x4, _mm_shuffle_epi32(max_i32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(max_i32x4);
}

/** @brief Horizontal min of 16 unsigned u32s in a ZMM register. */
NK_INTERNAL nk_u32_t _nk_reduce_min_u32x16_skylake(__m512i min_u32x16) {
    __m256i lo_u32x8 = _mm512_castsi512_si256(min_u32x16);
    __m256i hi_u32x8 = _mm512_extracti64x4_epi64(min_u32x16, 1);
    __m256i min_u32x8 = _mm256_min_epu32(lo_u32x8, hi_u32x8);
    __m128i lo_u32x4 = _mm256_castsi256_si128(min_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(min_u32x8, 1);
    __m128i min_u32x4 = _mm_min_epu32(lo_u32x4, hi_u32x4);
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    min_u32x4 = _mm_min_epu32(min_u32x4, _mm_shuffle_epi32(min_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(min_u32x4);
}

/** @brief Horizontal max of 16 unsigned u32s in a ZMM register. */
NK_INTERNAL nk_u32_t _nk_reduce_max_u32x16_skylake(__m512i max_u32x16) {
    __m256i lo_u32x8 = _mm512_castsi512_si256(max_u32x16);
    __m256i hi_u32x8 = _mm512_extracti64x4_epi64(max_u32x16, 1);
    __m256i max_u32x8 = _mm256_max_epu32(lo_u32x8, hi_u32x8);
    __m128i lo_u32x4 = _mm256_castsi256_si128(max_u32x8);
    __m128i hi_u32x4 = _mm256_extracti128_si256(max_u32x8, 1);
    __m128i max_u32x4 = _mm_max_epu32(lo_u32x4, hi_u32x4);
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(2, 3, 0, 1)));
    max_u32x4 = _mm_max_epu32(max_u32x4, _mm_shuffle_epi32(max_u32x4, _MM_SHUFFLE(1, 0, 3, 2)));
    return (nk_u32_t)_mm_cvtsi128_si32(max_u32x4);
}

/** @brief Horizontal min of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t _nk_reduce_min_i64x8_skylake(__m512i min_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(min_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(min_i64x8, 1);
    __m256i min_i64x4 = _mm256_min_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(min_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(min_i64x4, 1);
    __m128i min_i64x2 = _mm_min_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(min_i64x2, min_i64x2);
    __m128i final_i64 = _mm_min_epi64(min_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal max of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t _nk_reduce_max_i64x8_skylake(__m512i max_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(max_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(max_i64x8, 1);
    __m256i max_i64x4 = _mm256_max_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(max_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(max_i64x4, 1);
    __m128i max_i64x2 = _mm_max_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(max_i64x2, max_i64x2);
    __m128i final_i64 = _mm_max_epi64(max_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal min of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t _nk_reduce_min_u64x8_skylake(__m512i min_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(min_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(min_u64x8, 1);
    __m256i min_u64x4 = _mm256_min_epu64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(min_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(min_u64x4, 1);
    __m128i min_u64x2 = _mm_min_epu64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(min_u64x2, min_u64x2);
    __m128i final_u64 = _mm_min_epu64(min_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

/** @brief Horizontal max of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t _nk_reduce_max_u64x8_skylake(__m512i max_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(max_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(max_u64x8, 1);
    __m256i max_u64x4 = _mm256_max_epu64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(max_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(max_u64x4, 1);
    __m128i max_u64x2 = _mm_max_epu64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(max_u64x2, max_u64x2);
    __m128i final_u64 = _mm_max_epu64(max_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

/** @brief Horizontal sum of 8 signed i64s in a ZMM register. */
NK_INTERNAL nk_i64_t _nk_reduce_add_i64x8_skylake(__m512i sum_i64x8) {
    __m256i lo_i64x4 = _mm512_castsi512_si256(sum_i64x8);
    __m256i hi_i64x4 = _mm512_extracti64x4_epi64(sum_i64x8, 1);
    __m256i sum_i64x4 = _mm256_add_epi64(lo_i64x4, hi_i64x4);
    __m128i lo_i64x2 = _mm256_castsi256_si128(sum_i64x4);
    __m128i hi_i64x2 = _mm256_extracti128_si256(sum_i64x4, 1);
    __m128i sum_i64x2 = _mm_add_epi64(lo_i64x2, hi_i64x2);
    __m128i hi_lane_i64 = _mm_unpackhi_epi64(sum_i64x2, sum_i64x2);
    __m128i final_i64 = _mm_add_epi64(sum_i64x2, hi_lane_i64);
    return _mm_cvtsi128_si64(final_i64);
}

/** @brief Horizontal sum of 8 unsigned u64s in a ZMM register. */
NK_INTERNAL nk_u64_t _nk_reduce_add_u64x8_skylake(__m512i sum_u64x8) {
    __m256i lo_u64x4 = _mm512_castsi512_si256(sum_u64x8);
    __m256i hi_u64x4 = _mm512_extracti64x4_epi64(sum_u64x8, 1);
    __m256i sum_u64x4 = _mm256_add_epi64(lo_u64x4, hi_u64x4);
    __m128i lo_u64x2 = _mm256_castsi256_si128(sum_u64x4);
    __m128i hi_u64x2 = _mm256_extracti128_si256(sum_u64x4, 1);
    __m128i sum_u64x2 = _mm_add_epi64(lo_u64x2, hi_u64x2);
    __m128i hi_lane_u64 = _mm_unpackhi_epi64(sum_u64x2, sum_u64x2);
    __m128i final_u64 = _mm_add_epi64(sum_u64x2, hi_lane_u64);
    return (nk_u64_t)_mm_cvtsi128_si64(final_u64);
}

#pragma endregion // x86 Skylake Internal Helpers

#pragma region x86 Skylake Public Implementations

NK_INTERNAL void _nk_reduce_add_f32_skylake_contiguous( //
    nk_f32_t const *data, nk_size_t count, nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(data_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(data_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __m512 tail_f32x16 = _mm512_maskz_loadu_ps(tail_mask, data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(tail_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(tail_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        if (remaining > 8) sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    *result = _nk_reduce_add_f64x8_skylake(sum_f64x8);
}

NK_INTERNAL void _nk_reduce_add_f32_skylake_gather(                //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f32_t));
    __m512i indices_i32x16 = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                                _mm512_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 gathered_f32x16 = _mm512_i32gather_ps(indices_i32x16, data + idx_scalars * stride_elements,
                                                     sizeof(nk_f32_t));
        __m256 lo_f32x8 = _mm512_castps512_ps256(gathered_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(gathered_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    nk_f64_t sum = _nk_reduce_add_f64x8_skylake(sum_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(nk_f32_t const *)ptr;
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_f32_skylake_strided(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Masked load zeros out non-stride elements; zeros don't affect the sum
    __mmask16 stride_mask_m16 = _nk_stride_mask_b32x16(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(stride_mask_m16, data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(data_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(data_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    // Masked tail: combine stride mask with tail mask
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_maskz_loadu_ps(load_mask_m16, data + idx_scalars);
        __m256 lo_f32x8 = _mm512_castps512_ps256(data_f32x16);
        __m256 hi_f32x8 = _mm512_extractf32x8_ps(data_f32x16, 1);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(lo_f32x8));
        if (remaining > 8) sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_cvtps_pd(hi_f32x8));
    }
    *result = _nk_reduce_add_f64x8_skylake(sum_f64x8);
}

NK_PUBLIC void nk_reduce_add_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_add_f32_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) _nk_reduce_add_f32_skylake_contiguous(data, count, result);
    else if (stride_elements <= 8) _nk_reduce_add_f32_skylake_strided(data, count, stride_elements, result);
    else _nk_reduce_add_f32_skylake_gather(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_add_f64_skylake_contiguous( //
    nk_f64_t const *data, nk_size_t count, nk_f64_t *result) {
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, data_f64x8);
    }
    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __m512d tail_f64x8 = _mm512_maskz_loadu_pd(tail_mask, data + idx_scalars);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, tail_f64x8);
    }
    *result = _nk_reduce_add_f64x8_skylake(sum_f64x8);
}

NK_INTERNAL void _nk_reduce_add_f64_skylake_gather(                //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_i32_t stride_elements = (nk_i32_t)(stride_bytes / sizeof(nk_f64_t));
    __m256i indices_i32x8 = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32(stride_elements));
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d gathered_f64x8 = _mm512_i32gather_pd(indices_i32x8, data + idx_scalars * stride_elements,
                                                     sizeof(nk_f64_t));
        sum_f64x8 = _mm512_add_pd(sum_f64x8, gathered_f64x8);
    }
    nk_f64_t sum = _nk_reduce_add_f64x8_skylake(sum_f64x8);
    unsigned char const *ptr = (unsigned char const *)(data + idx_scalars * stride_elements);
    for (; idx_scalars < count; ++idx_scalars, ptr += stride_bytes) sum += *(nk_f64_t const *)ptr;
    *result = sum;
}

NK_INTERNAL void _nk_reduce_add_f64_skylake_strided(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *result) {
    // Masked load zeros out non-stride elements; zeros don't affect the sum
    __mmask8 stride_mask_m8 = _nk_stride_mask_b64x8(stride_elements);
    __m512d sum_f64x8 = _mm512_setzero_pd();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;
    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_maskz_loadu_pd(stride_mask_m8, data + idx_scalars);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, data_f64x8);
    }
    // Masked tail: combine stride mask with tail mask
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_maskz_loadu_pd(load_mask_m8, data + idx_scalars);
        sum_f64x8 = _mm512_add_pd(sum_f64x8, data_f64x8);
    }
    *result = _nk_reduce_add_f64x8_skylake(sum_f64x8);
}

NK_PUBLIC void nk_reduce_add_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *result) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_add_f64_serial(data, count, stride_bytes, result);
    else if (stride_elements == 1) _nk_reduce_add_f64_skylake_contiguous(data, count, result);
    else if (stride_elements <= 8) _nk_reduce_add_f64_skylake_strided(data, count, stride_elements, result);
    else _nk_reduce_add_f64_skylake_gather(data, count, stride_bytes, result);
}

NK_INTERNAL void _nk_reduce_min_f32_skylake_contiguous( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512 min_f32x16 = _mm512_loadu_ps(data);
    __m512i min_idx_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_idx_i32x16 = _mm512_set1_epi32(16);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 lt_mask = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask, current_idx_i32x16);
        current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
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
    nk_f32_t min_val = _nk_reduce_min_f32x16_skylake(min_f32x16);

    // Find the first lane that matches the minimum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_idx_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void _nk_reduce_min_f32_skylake_strided(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    // Masked load with +inf for non-stride elements; track logical indices
    __mmask16 stride_mask_m16 = _nk_stride_mask_b32x16(stride_elements);
    __m512 pos_inf_f32x16 = _mm512_set1_ps(__builtin_huge_valf());
    __m512 min_f32x16 = pos_inf_f32x16;
    __m512i min_idx_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i32x16 = _nk_stride_logidx_i32x16(stride_elements);
    nk_size_t elems_per_chunk = _nk_stride_elems_b32x16(stride_elements);
    __m512i step_i32x16 = _mm512_set1_epi32((nk_i32_t)elems_per_chunk);

    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_mask_loadu_ps(pos_inf_f32x16, stride_mask_m16, data + idx_scalars);
        __mmask16 lt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask_m16, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask_m16, logical_idx_i32x16);
        logical_idx_i32x16 = _mm512_add_epi32(logical_idx_i32x16, step_i32x16);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_mask_loadu_ps(pos_inf_f32x16, load_mask_m16, data + idx_scalars);
        __mmask16 lt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, min_f32x16, _CMP_LT_OQ);
        min_f32x16 = _mm512_mask_mov_ps(min_f32x16, lt_mask_m16, data_f32x16);
        min_idx_i32x16 = _mm512_mask_mov_epi32(min_idx_i32x16, lt_mask_m16, logical_idx_i32x16);
    }

    // Horizontal reduction
    nk_f32_t min_val = _nk_reduce_min_f32x16_skylake(min_f32x16);
    __mmask16 eq_mask_m16 = _mm512_cmp_ps_mask(min_f32x16, _mm512_set1_ps(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m16);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, min_idx_i32x16);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_min_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 16)
        _nk_reduce_min_f32_skylake_contiguous(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        _nk_reduce_min_f32_skylake_strided(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f32_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void _nk_reduce_max_f32_skylake_contiguous( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512 max_f32x16 = _mm512_loadu_ps(data);
    __m512i max_idx_i32x16 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i current_idx_i32x16 = _mm512_set1_epi32(16);
    __m512i step_i32x16 = _mm512_set1_epi32(16);

    nk_size_t idx_scalars = 16;
    for (; idx_scalars + 16 <= count; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_loadu_ps(data + idx_scalars);
        __mmask16 gt_mask = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask, current_idx_i32x16);
        current_idx_i32x16 = _mm512_add_epi32(current_idx_i32x16, step_i32x16);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
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
    nk_f32_t max_val = _nk_reduce_max_f32x16_skylake(max_f32x16);

    // Find the first lane that matches the maximum
    __mmask16 eq_mask = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    // Extract the index from that lane
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_idx_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void _nk_reduce_max_f32_skylake_strided(                  //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    // Masked load with -inf for non-stride elements; track logical indices
    __mmask16 stride_mask_m16 = _nk_stride_mask_b32x16(stride_elements);
    __m512 neg_inf_f32x16 = _mm512_set1_ps(-__builtin_huge_valf());
    __m512 max_f32x16 = neg_inf_f32x16;
    __m512i max_idx_i32x16 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i32x16 = _nk_stride_logidx_i32x16(stride_elements);
    nk_size_t elems_per_chunk = _nk_stride_elems_b32x16(stride_elements);
    __m512i step_i32x16 = _mm512_set1_epi32((nk_i32_t)elems_per_chunk);

    for (; idx_scalars + 16 <= total_scalars; idx_scalars += 16) {
        __m512 data_f32x16 = _mm512_mask_loadu_ps(neg_inf_f32x16, stride_mask_m16, data + idx_scalars);
        __mmask16 gt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask_m16, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask_m16, logical_idx_i32x16);
        logical_idx_i32x16 = _mm512_add_epi32(logical_idx_i32x16, step_i32x16);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask16 tail_mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)remaining);
        __mmask16 load_mask_m16 = stride_mask_m16 & tail_mask_m16;
        __m512 data_f32x16 = _mm512_mask_loadu_ps(neg_inf_f32x16, load_mask_m16, data + idx_scalars);
        __mmask16 gt_mask_m16 = _mm512_cmp_ps_mask(data_f32x16, max_f32x16, _CMP_GT_OQ);
        max_f32x16 = _mm512_mask_mov_ps(max_f32x16, gt_mask_m16, data_f32x16);
        max_idx_i32x16 = _mm512_mask_mov_epi32(max_idx_i32x16, gt_mask_m16, logical_idx_i32x16);
    }

    // Horizontal reduction
    nk_f32_t max_val = _nk_reduce_max_f32x16_skylake(max_f32x16);
    __mmask16 eq_mask_m16 = _mm512_cmp_ps_mask(max_f32x16, _mm512_set1_ps(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m16);
    nk_i32_t indices[16];
    _mm512_storeu_si512(indices, max_idx_i32x16);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_max_f32_skylake(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (!aligned) nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 16)
        _nk_reduce_max_f32_skylake_contiguous(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        _nk_reduce_max_f32_skylake_strided(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f32_serial(data, count, stride_bytes, max_value, max_index);
}

NK_INTERNAL void _nk_reduce_min_f64_skylake_contiguous( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Single-pass: track both min value and index in SIMD
    __m512d min_f64x8 = _mm512_loadu_pd(data);
    __m512i min_idx_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_idx_i64x8 = _mm512_set1_epi64(8);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 lt_mask = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask, current_idx_i64x8);
        current_idx_i64x8 = _mm512_add_epi64(current_idx_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
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
    nk_f64_t min_val = _nk_reduce_min_f64x8_skylake(min_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_idx_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void _nk_reduce_min_f64_skylake_strided(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    // Masked load with +inf for non-stride elements; track logical indices
    __mmask8 stride_mask_m8 = _nk_stride_mask_b64x8(stride_elements);
    __m512d pos_inf_f64x8 = _mm512_set1_pd(__builtin_huge_val());
    __m512d min_f64x8 = pos_inf_f64x8;
    __m512i min_idx_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i64x8 = _nk_stride_logidx_i64x8(stride_elements);
    nk_size_t elems_per_chunk = _nk_stride_elems_b64x8(stride_elements);
    __m512i step_i64x8 = _mm512_set1_epi64((nk_i64_t)elems_per_chunk);

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_mask_loadu_pd(pos_inf_f64x8, stride_mask_m8, data + idx_scalars);
        __mmask8 lt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask_m8, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask_m8, logical_idx_i64x8);
        logical_idx_i64x8 = _mm512_add_epi64(logical_idx_i64x8, step_i64x8);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_mask_loadu_pd(pos_inf_f64x8, load_mask_m8, data + idx_scalars);
        __mmask8 lt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, min_f64x8, _CMP_LT_OQ);
        min_f64x8 = _mm512_mask_mov_pd(min_f64x8, lt_mask_m8, data_f64x8);
        min_idx_i64x8 = _mm512_mask_mov_epi64(min_idx_i64x8, lt_mask_m8, logical_idx_i64x8);
    }

    // Horizontal reduction
    nk_f64_t min_val = _nk_reduce_min_f64x8_skylake(min_f64x8);
    __mmask8 eq_mask_m8 = _mm512_cmp_pd_mask(min_f64x8, _mm512_set1_pd(min_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m8);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, min_idx_i64x8);

    *min_value = min_val;
    *min_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_min_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
    else if (stride_elements == 1 && count >= 8)
        _nk_reduce_min_f64_skylake_contiguous(data, count, min_value, min_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        _nk_reduce_min_f64_skylake_strided(data, count, stride_elements, min_value, min_index);
    else nk_reduce_min_f64_serial(data, count, stride_bytes, min_value, min_index);
}

NK_INTERNAL void _nk_reduce_max_f64_skylake_contiguous( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Single-pass: track both max value and index in SIMD
    __m512d max_f64x8 = _mm512_loadu_pd(data);
    __m512i max_idx_i64x8 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i current_idx_i64x8 = _mm512_set1_epi64(8);
    __m512i step_i64x8 = _mm512_set1_epi64(8);

    nk_size_t idx_scalars = 8;
    for (; idx_scalars + 8 <= count; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_loadu_pd(data + idx_scalars);
        __mmask8 gt_mask = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask, current_idx_i64x8);
        current_idx_i64x8 = _mm512_add_epi64(current_idx_i64x8, step_i64x8);
    }

    // Handle tail with masked load
    nk_size_t remaining = count - idx_scalars;
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
    nk_f64_t max_val = _nk_reduce_max_f64x8_skylake(max_f64x8);
    __mmask8 eq_mask = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask);

    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_idx_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_INTERNAL void _nk_reduce_max_f64_skylake_strided(                  //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_elements, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    // Masked load with -inf for non-stride elements; track logical indices
    __mmask8 stride_mask_m8 = _nk_stride_mask_b64x8(stride_elements);
    __m512d neg_inf_f64x8 = _mm512_set1_pd(-__builtin_huge_val());
    __m512d max_f64x8 = neg_inf_f64x8;
    __m512i max_idx_i64x8 = _mm512_setzero_si512();
    nk_size_t idx_scalars = 0;
    nk_size_t total_scalars = count * stride_elements;

    // Precomputed logical index vector and step for this stride
    __m512i logical_idx_i64x8 = _nk_stride_logidx_i64x8(stride_elements);
    nk_size_t elems_per_chunk = _nk_stride_elems_b64x8(stride_elements);
    __m512i step_i64x8 = _mm512_set1_epi64((nk_i64_t)elems_per_chunk);

    for (; idx_scalars + 8 <= total_scalars; idx_scalars += 8) {
        __m512d data_f64x8 = _mm512_mask_loadu_pd(neg_inf_f64x8, stride_mask_m8, data + idx_scalars);
        __mmask8 gt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask_m8, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask_m8, logical_idx_i64x8);
        logical_idx_i64x8 = _mm512_add_epi64(logical_idx_i64x8, step_i64x8);
    }

    // Masked tail
    nk_size_t remaining = total_scalars - idx_scalars;
    if (remaining > 0) {
        __mmask8 tail_mask_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)remaining);
        __mmask8 load_mask_m8 = stride_mask_m8 & tail_mask_m8;
        __m512d data_f64x8 = _mm512_mask_loadu_pd(neg_inf_f64x8, load_mask_m8, data + idx_scalars);
        __mmask8 gt_mask_m8 = _mm512_cmp_pd_mask(data_f64x8, max_f64x8, _CMP_GT_OQ);
        max_f64x8 = _mm512_mask_mov_pd(max_f64x8, gt_mask_m8, data_f64x8);
        max_idx_i64x8 = _mm512_mask_mov_epi64(max_idx_i64x8, gt_mask_m8, logical_idx_i64x8);
    }

    // Horizontal reduction
    nk_f64_t max_val = _nk_reduce_max_f64x8_skylake(max_f64x8);
    __mmask8 eq_mask_m8 = _mm512_cmp_pd_mask(max_f64x8, _mm512_set1_pd(max_val), _CMP_EQ_OQ);
    unsigned int first_lane = _tzcnt_u32(eq_mask_m8);
    nk_i64_t indices[8];
    _mm512_storeu_si512(indices, max_idx_i64x8);

    *max_value = max_val;
    *max_index = (nk_size_t)indices[first_lane];
}

NK_PUBLIC void nk_reduce_max_f64_skylake(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (!aligned) nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
    else if (stride_elements == 1 && count >= 8)
        _nk_reduce_max_f64_skylake_contiguous(data, count, max_value, max_index);
    else if (stride_elements >= 2 && stride_elements <= 8)
        _nk_reduce_max_f64_skylake_strided(data, count, stride_elements, max_value, max_index);
    else nk_reduce_max_f64_serial(data, count, stride_bytes, max_value, max_index);
}

#pragma endregion // x86 Skylake Public Implementations

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // _NK_TARGET_X86

#pragma endregion // x86 Skylake Implementations

#pragma region Compile-Time Dispatch

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

#pragma endregion // Compile-Time Dispatch

#ifdef __cplusplus
}
#endif

#endif // NK_REDUCE_H
