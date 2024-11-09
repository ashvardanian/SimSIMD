/**
 *  @file       elementwise.h
 *  @brief      SIMD-accelerated mixed-precision element-wise operations.
 *  @author     Ash Vardanian
 *  @date       October 16, 2024
 *
 *  Contains following @b Unary/Binary/Ternary element-wise operations:
 *  - Scale (Multiply) with Shift: R[i] = Alpha * A[i] + Beta
 *  - Sum (Add): R[i] = A[i] + B[i]
 *  - WSum or Weighted-Sum: R[i] = Alpha * A[i] + Beta * B[i]
 *  - FMA or Fused-Multiply-Add: R[i] = Alpha * A[i] * B[i] + Beta * C[i]
 *
 *  This tiny set of operations if enough to implement a wide range of algorithms.
 *  To scale a vector by a scalar, just call WSum with $Beta$ = 0.
 *  To sum two vectors, just call WSum with $Alpha$ = $Beta$ = 1.
 *  To average two vectors, just call WSum with $Alpha$ = $Beta$ = 0.5.
 *  To multiply vectors element-wise, just call FMA with $Beta$ = 0.
 *
 *  For datatypes:
 *  - 64-bit IEEE floating point numbers
 *  - 32-bit IEEE floating point numbers
 *  - 16-bit IEEE floating point numbers
 *  - 16-bit brain floating point numbers
 *  - 8-bit unsigned integers... for binary data
 *  - 8-bit signed integers... for signal processing
 *  For many operations higher-precision integral types are also supported:
 *  - 16-bit signed & unsigned integers
 *  - 32-bit signed & unsigned integers
 *  - 64-bit signed & unsigned integers
 *
 *  For hardware architectures:
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire
 *
 *  We use `f16` for `i8` and `u8` arithmetic. This is because Arm received `f16` support earlier than `bf16`.
 *  For example, Apple M1 has `f16` support and `bf16` was only added in M2. On the other hand, on paper,
 *  AMD Genoa has `bf16` support, and `f16` is only available on Intel Sapphire Rapids and newer.
 *  Sadly, the SIMD support for `bf16` is limited to mixed-precision dot-products, which makes it useless here.
 *
 *  x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
 *  Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef SIMSIMD_ELEMENTWISE_H
#define SIMSIMD_ELEMENTWISE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

/*  Serial backends for all numeric types.
 *  By default they use 32-bit arithmetic, unless the arguments themselves contain 64-bit floats.
 *  For double-precision computation check out the "*_accurate" variants of those "*_serial" functions.
 */
SIMSIMD_PUBLIC void simsimd_scale_f64_serial(simsimd_f64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f32_serial(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f16_serial(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_bf16_serial(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i8_serial(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u8_serial(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i16_serial(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u16_serial(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i32_serial(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u32_serial(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i64_serial(simsimd_i64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u64_serial(simsimd_u64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result);

SIMSIMD_PUBLIC void simsimd_sum_f64_serial(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_sum_f32_serial(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_f16_serial(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_bf16_serial(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i8_serial(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u8_serial(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i16_serial(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_size_t n, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u16_serial(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_size_t n, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i32_serial(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_size_t n, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u32_serial(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i64_serial(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_size_t n, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u64_serial(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_size_t n, simsimd_u64_t *result);

SIMSIMD_PUBLIC void simsimd_wsum_f64_serial(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_f32_serial(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_f16_serial(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_bf16_serial(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_i8_serial(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_u8_serial(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);

SIMSIMD_PUBLIC void simsimd_fma_f64_serial(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f32_serial(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f16_serial(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_bf16_serial(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i8_serial(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u8_serial(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i16_serial(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u16_serial(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i32_serial(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u32_serial(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i64_serial(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u64_serial(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result);

/*  SIMD-powered backends for Arm NEON, mostly using 32-bit arithmetic over 128-bit words.
 *  By far the most portable backend, covering most Arm v8 devices, over a billion phones, and almost all
 *  server CPUs produced before 2023.
 */
SIMSIMD_PUBLIC void simsimd_scale_f32_neon(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f16_neon(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_bf16_neon(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i8_neon(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u8_neon(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i16_neon(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u16_neon(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i32_neon(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u32_neon(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i64_neon(simsimd_i64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u64_neon(simsimd_u64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result);

SIMSIMD_PUBLIC void simsimd_sum_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i16_neon(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_size_t n, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u16_neon(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_size_t n, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i32_neon(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_size_t n, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u32_neon(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i64_neon(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_size_t n, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u64_neon(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_size_t n, simsimd_u64_t *result);

SIMSIMD_PUBLIC void simsimd_wsum_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);

SIMSIMD_PUBLIC void simsimd_fma_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i16_neon(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u16_neon(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i32_neon(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u32_neon(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i64_neon(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u64_neon(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result);

/*  SIMD-powered backends for AVX2 CPUs of Haswell generation and newer, using 32-bit arithmetic over 256-bit words.
 *  First demonstrated in 2011, at least one Haswell-based processor was still being sold in 2022 — the Pentium G3420.
 *  Practically all modern x86 CPUs support AVX2, FMA, and F16C, making it a perfect baseline for SIMD algorithms.
 *  On other hand, there is no need to implement AVX2 versions of `f32` and `f64` functions, as those are
 *  properly vectorized by recent compilers.
 */
SIMSIMD_PUBLIC void simsimd_scale_f64_haswell(simsimd_f64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f32_haswell(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f16_haswell(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_bf16_haswell(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i8_haswell(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u8_haswell(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i16_haswell(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u16_haswell(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i32_haswell(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u32_haswell(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);

SIMSIMD_PUBLIC void simsimd_sum_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_sum_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i16_haswell(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_size_t n, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u16_haswell(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_size_t n, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i32_haswell(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_size_t n, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u32_haswell(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n, simsimd_u32_t *result);

SIMSIMD_PUBLIC void simsimd_wsum_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);

SIMSIMD_PUBLIC void simsimd_fma_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i16_haswell(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u16_haswell(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i32_haswell(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u32_haswell(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);

/*  SIMD-powered backends for older AVX512 CPUs.
 *  Unlike the distance metrics, the SIMD implementation of FMA and WSum benefits from aligned stores.
 *  Assuming the size of ZMM register matches the width of the cache line, we skip the unaligned head
 *  and tail of the output buffer, and only use aligned stores in the main loop.
 *  Also, unlike Haswell, Skylake-X CPUs natively support converting between more numeric types.
 */
SIMSIMD_PUBLIC void simsimd_scale_f64_skylake(simsimd_f64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f32_skylake(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_f16_skylake(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_bf16_skylake(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i8_skylake(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u8_skylake(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i16_skylake(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u16_skylake(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i32_skylake(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u32_skylake(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i64_skylake(simsimd_i64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u64_skylake(simsimd_u64_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result);

SIMSIMD_PUBLIC void simsimd_sum_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_sum_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_bf16_skylake(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_bf16_t *result);

SIMSIMD_PUBLIC void simsimd_wsum_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_bf16_skylake(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);

SIMSIMD_PUBLIC void simsimd_fma_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_f16_skylake(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_bf16_skylake(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i8_skylake(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u8_skylake(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i16_skylake(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u16_skylake(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i32_skylake(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u32_skylake(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i64_skylake(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u64_skylake(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result);

/*  SIMD-powered backends for AVX512 CPUs with integer math support.
 *  Interestingly, Ice Lake only has saturating arithmetic for 8-bit and 16-bit integers, but not for 32-bit and 64-bit.
 */
SIMSIMD_PUBLIC void simsimd_sum_i8_ice(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u8_ice(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_u8_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i16_ice(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_size_t n, simsimd_i16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u16_ice(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_size_t n, simsimd_u16_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i32_ice(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_size_t n, simsimd_i32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u32_ice(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n, simsimd_u32_t *result);
SIMSIMD_PUBLIC void simsimd_sum_i64_ice(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_size_t n, simsimd_i64_t *result);
SIMSIMD_PUBLIC void simsimd_sum_u64_ice(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_size_t n, simsimd_u64_t *result);

/*  SIMD-powered backends for newer AVX512 CPUs with native F16 support.
 *  Unlike the distance metrics, the SIMD implementation of FMA and WSum benefits from aligned stores.
 *  Assuming the size of ZMM register matches the width of the cache line, we skip the unaligned head
 *  and tail of the output buffer, and only use aligned stores in the main loop.
 */
SIMSIMD_PUBLIC void simsimd_scale_f16_sapphire(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i8_sapphire(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u8_sapphire(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);

SIMSIMD_PUBLIC void simsimd_scale_f16_sapphire(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_scale_i8_sapphire(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_scale_u8_sapphire(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);

SIMSIMD_PUBLIC void simsimd_sum_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_f16_t *result);

SIMSIMD_PUBLIC void simsimd_wsum_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_i8_sapphire(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_wsum_u8_sapphire(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);

SIMSIMD_PUBLIC void simsimd_fma_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result);
SIMSIMD_PUBLIC void simsimd_fma_i8_sapphire(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result);
SIMSIMD_PUBLIC void simsimd_fma_u8_sapphire(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result);

// clang-format on

#define SIMSIMD_MAKE_SCALE(name, input_type, accumulator_type, load_and_convert, convert_and_store)              \
    SIMSIMD_PUBLIC void simsimd_scale_##input_type##_##name(simsimd_##input_type##_t const *a, simsimd_size_t n, \
                                                            simsimd_distance_t alpha, simsimd_distance_t beta,   \
                                                            simsimd_##input_type##_t *result) {                  \
        simsimd_##accumulator_type##_t alpha_cast = (simsimd_##accumulator_type##_t)alpha;                       \
        simsimd_##accumulator_type##_t beta_cast = (simsimd_##accumulator_type##_t)beta;                         \
        simsimd_##accumulator_type##_t ai, sum;                                                                  \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                        \
            sum = (simsimd_##accumulator_type##_t)(alpha_cast * ai + beta_cast);                                 \
            convert_and_store(&sum, result + i);                                                                 \
        }                                                                                                        \
    }
#define SIMSIMD_MAKE_SUM(name, input_type, accumulator_type, load_and_convert, convert_and_store)              \
    SIMSIMD_PUBLIC void simsimd_sum_##input_type##_##name(simsimd_##input_type##_t const *a,                   \
                                                          simsimd_##input_type##_t const *b, simsimd_size_t n, \
                                                          simsimd_##input_type##_t *result) {                  \
        simsimd_##accumulator_type##_t ai, bi, sum;                                                            \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                              \
            load_and_convert(a + i, &ai);                                                                      \
            load_and_convert(b + i, &bi);                                                                      \
            sum = ai + bi;                                                                                     \
            convert_and_store(&sum, result + i);                                                               \
        }                                                                                                      \
    }

#define SIMSIMD_MAKE_WSUM(name, input_type, accumulator_type, load_and_convert, convert_and_store) \
    SIMSIMD_PUBLIC void simsimd_wsum_##input_type##_##name(                                        \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_size_t n,    \
        simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_##input_type##_t *result) {     \
        simsimd_##accumulator_type##_t alpha_cast = (simsimd_##accumulator_type##_t)alpha;         \
        simsimd_##accumulator_type##_t beta_cast = (simsimd_##accumulator_type##_t)beta;           \
        simsimd_##accumulator_type##_t ai, bi, ai_scaled, bi_scaled, sum;                          \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                  \
            load_and_convert(a + i, &ai);                                                          \
            load_and_convert(b + i, &bi);                                                          \
            ai_scaled = ai * alpha_cast;                                                           \
            bi_scaled = bi * beta_cast;                                                            \
            sum = ai_scaled + bi_scaled;                                                           \
            convert_and_store(&sum, result + i);                                                   \
        }                                                                                          \
    }

#define SIMSIMD_MAKE_FMA(name, input_type, accumulator_type, load_and_convert, convert_and_store)                \
    SIMSIMD_PUBLIC void simsimd_fma_##input_type##_##name(                                                       \
        simsimd_##input_type##_t const *a, simsimd_##input_type##_t const *b, simsimd_##input_type##_t const *c, \
        simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_##input_type##_t *result) { \
        simsimd_##accumulator_type##_t alpha_cast = (simsimd_##accumulator_type##_t)alpha;                       \
        simsimd_##accumulator_type##_t beta_cast = (simsimd_##accumulator_type##_t)beta;                         \
        simsimd_##accumulator_type##_t ai, bi, ci, abi_scaled, ci_scaled, sum;                                   \
        for (simsimd_size_t i = 0; i != n; ++i) {                                                                \
            load_and_convert(a + i, &ai);                                                                        \
            load_and_convert(b + i, &bi);                                                                        \
            load_and_convert(c + i, &ci);                                                                        \
            abi_scaled = ai * bi * alpha_cast;                                                                   \
            ci_scaled = ci * beta_cast;                                                                          \
            sum = abi_scaled + ci_scaled;                                                                        \
            convert_and_store(&sum, result + i);                                                                 \
        }                                                                                                        \
    }

SIMSIMD_MAKE_SUM(serial, f64, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_sum_f64_serial
SIMSIMD_MAKE_SUM(serial, f32, f32, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_sum_f32_serial
SIMSIMD_MAKE_SUM(serial, f16, f32, simsimd_f16_to_f32, simsimd_f32_to_f16)         // simsimd_sum_f16_serial
SIMSIMD_MAKE_SUM(serial, bf16, f32, simsimd_bf16_to_f32, simsimd_f32_to_bf16)      // simsimd_sum_bf16_serial
SIMSIMD_MAKE_SUM(serial, i8, i64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_i64_to_i8)      // simsimd_sum_i8_serial
SIMSIMD_MAKE_SUM(serial, u8, i64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_i64_to_u8)      // simsimd_sum_u8_serial
SIMSIMD_MAKE_SUM(serial, i16, i64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_i64_to_i16)    // simsimd_sum_i16_serial
SIMSIMD_MAKE_SUM(serial, u16, i64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_i64_to_u16)    // simsimd_sum_u16_serial
SIMSIMD_MAKE_SUM(serial, i32, i64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_i64_to_i32)    // simsimd_sum_i32_serial
SIMSIMD_MAKE_SUM(serial, u32, i64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_i64_to_u32)    // simsimd_sum_u32_serial
SIMSIMD_MAKE_SUM(serial, i64, i64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_sum_i64_serial
SIMSIMD_MAKE_SUM(serial, u64, u64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_sum_u64_serial

SIMSIMD_MAKE_SUM(accurate, f32, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_sum_f32_accurate
SIMSIMD_MAKE_SUM(accurate, f16, f64, _simsimd_f16_to_f64, _simsimd_f64_to_f16)       // simsimd_sum_f16_accurate
SIMSIMD_MAKE_SUM(accurate, bf16, f64, _simsimd_bf16_to_f64, _simsimd_f64_to_bf16)    // simsimd_sum_bf16_accurate

SIMSIMD_MAKE_SCALE(serial, f64, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_scale_f64_serial
SIMSIMD_MAKE_SCALE(serial, f32, f32, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_scale_f32_serial
SIMSIMD_MAKE_SCALE(serial, f16, f32, simsimd_f16_to_f32, simsimd_f32_to_f16)         // simsimd_scale_f16_serial
SIMSIMD_MAKE_SCALE(serial, bf16, f32, simsimd_bf16_to_f32, simsimd_f32_to_bf16)      // simsimd_scale_bf16_serial
SIMSIMD_MAKE_SCALE(serial, i8, f32, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f32_to_i8)      // simsimd_scale_i8_serial
SIMSIMD_MAKE_SCALE(serial, u8, f32, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f32_to_u8)      // simsimd_scale_u8_serial
SIMSIMD_MAKE_SCALE(serial, i16, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i16)    // simsimd_scale_i16_serial
SIMSIMD_MAKE_SCALE(serial, u16, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u16)    // simsimd_scale_u16_serial
SIMSIMD_MAKE_SCALE(serial, i32, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i32)    // simsimd_scale_i32_serial
SIMSIMD_MAKE_SCALE(serial, u32, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u32)    // simsimd_scale_u32_serial
SIMSIMD_MAKE_SCALE(serial, i64, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i64)    // simsimd_scale_i64_serial
SIMSIMD_MAKE_SCALE(serial, u64, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u64)    // simsimd_scale_u64_serial

SIMSIMD_MAKE_SCALE(accurate, f32, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_scale_f32_accurate
SIMSIMD_MAKE_SCALE(accurate, f16, f64, _simsimd_f16_to_f64, _simsimd_f64_to_f16)       // simsimd_scale_f16_accurate
SIMSIMD_MAKE_SCALE(accurate, bf16, f64, _simsimd_bf16_to_f64, _simsimd_f64_to_bf16)    // simsimd_scale_bf16_accurate
SIMSIMD_MAKE_SCALE(accurate, i8, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i8)      // simsimd_scale_i8_accurate
SIMSIMD_MAKE_SCALE(accurate, u8, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u8)      // simsimd_scale_u8_accurate

SIMSIMD_MAKE_WSUM(serial, f64, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_wsum_f64_serial
SIMSIMD_MAKE_WSUM(serial, f32, f32, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_wsum_f32_serial
SIMSIMD_MAKE_WSUM(serial, f16, f32, simsimd_f16_to_f32, simsimd_f32_to_f16)         // simsimd_wsum_f16_serial
SIMSIMD_MAKE_WSUM(serial, bf16, f32, simsimd_bf16_to_f32, simsimd_f32_to_bf16)      // simsimd_wsum_bf16_serial
SIMSIMD_MAKE_WSUM(serial, i8, f32, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f32_to_i8)      // simsimd_wsum_i8_serial
SIMSIMD_MAKE_WSUM(serial, u8, f32, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f32_to_u8)      // simsimd_wsum_u8_serial

SIMSIMD_MAKE_WSUM(accurate, f32, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_wsum_f32_accurate
SIMSIMD_MAKE_WSUM(accurate, f16, f64, _simsimd_f16_to_f64, _simsimd_f64_to_f16)       // simsimd_wsum_f16_accurate
SIMSIMD_MAKE_WSUM(accurate, bf16, f64, _simsimd_bf16_to_f64, _simsimd_f64_to_bf16)    // simsimd_wsum_bf16_accurate
SIMSIMD_MAKE_WSUM(accurate, i8, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i8)      // simsimd_wsum_i8_accurate
SIMSIMD_MAKE_WSUM(accurate, u8, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u8)      // simsimd_wsum_u8_accurate

SIMSIMD_MAKE_FMA(serial, f64, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_fma_f64_serial
SIMSIMD_MAKE_FMA(serial, f32, f32, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_fma_f32_serial
SIMSIMD_MAKE_FMA(serial, f16, f32, simsimd_f16_to_f32, simsimd_f32_to_f16)         // simsimd_fma_f16_serial
SIMSIMD_MAKE_FMA(serial, bf16, f32, simsimd_bf16_to_f32, simsimd_f32_to_bf16)      // simsimd_fma_bf16_serial
SIMSIMD_MAKE_FMA(serial, i8, f32, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f32_to_i8)      // simsimd_fma_i8_serial
SIMSIMD_MAKE_FMA(serial, u8, f32, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f32_to_u8)      // simsimd_fma_u8_serial
SIMSIMD_MAKE_FMA(serial, i16, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i16)    // simsimd_fma_i16_serial
SIMSIMD_MAKE_FMA(serial, u16, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u16)    // simsimd_fma_u16_serial
SIMSIMD_MAKE_FMA(serial, i32, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i32)    // simsimd_fma_i32_serial
SIMSIMD_MAKE_FMA(serial, u32, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u32)    // simsimd_fma_u32_serial
SIMSIMD_MAKE_FMA(serial, i64, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i64)    // simsimd_fma_i64_serial
SIMSIMD_MAKE_FMA(serial, u64, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u64)    // simsimd_fma_u64_serial

SIMSIMD_MAKE_FMA(accurate, f32, f64, _SIMSIMD_ASSIGN_1_TO_2, _SIMSIMD_ASSIGN_1_TO_2) // simsimd_fma_f32_accurate
SIMSIMD_MAKE_FMA(accurate, f16, f64, _simsimd_f16_to_f64, _simsimd_f64_to_f16)       // simsimd_fma_f16_accurate
SIMSIMD_MAKE_FMA(accurate, bf16, f64, _simsimd_bf16_to_f64, _simsimd_f64_to_bf16)    // simsimd_fma_bf16_accurate
SIMSIMD_MAKE_FMA(accurate, i8, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_i8)      // simsimd_fma_i8_accurate
SIMSIMD_MAKE_FMA(accurate, u8, f64, _SIMSIMD_ASSIGN_1_TO_2, _simsimd_f64_to_u8)      // simsimd_fma_u8_accurate

#if _SIMSIMD_TARGET_X86
#if SIMSIMD_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_sum_f32_haswell(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 sum_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

SIMSIMD_PUBLIC void simsimd_scale_f32_haswell(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_f32_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        _mm256_storeu_ps(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_f32 * a[i] + beta_f32;
}

SIMSIMD_PUBLIC void simsimd_wsum_f32_haswell(                         //
    simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f32_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f32_haswell(a, n, alpha, 0, result); }
        else { simsimd_scale_f32_haswell(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 a_scaled_vec = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
        _mm256_storeu_ps(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_f32 * a[i] + beta_f32 * b[i];
}

SIMSIMD_PUBLIC void simsimd_sum_f64_haswell(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                            simsimd_f64_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d sum_vec = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

SIMSIMD_PUBLIC void simsimd_scale_f64_haswell(simsimd_f64_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_f64_t *result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d sum_vec = _mm256_fmadd_pd(a_vec, alpha_vec, beta_vec);
        _mm256_storeu_pd(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha * a[i] + beta;
}

SIMSIMD_PUBLIC void simsimd_wsum_f64_haswell(                         //
    simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f64_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f64_haswell(a, n, alpha, 0, result); }
        else { simsimd_scale_f64_haswell(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d a_scaled_vec = _mm256_mul_pd(a_vec, alpha_vec);
        __m256d sum_vec = _mm256_fmadd_pd(b_vec, beta_vec, a_scaled_vec);
        _mm256_storeu_pd(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha * a[i] + beta * b[i];
}

SIMSIMD_PUBLIC void simsimd_sum_f16_haswell(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                            simsimd_f16_t *result) {

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_f16 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_vec = _mm256_cvtph_ps(a_f16);
        __m256 b_vec = _mm256_cvtph_ps(b_f16);
        __m256 sum_vec = _mm256_add_ps(a_vec, b_vec);
        __m128i sum_f16 = _mm256_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), sum_f16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi;
        simsimd_f16_to_f32(a + i, &ai);
        simsimd_f16_to_f32(b + i, &bi);
        simsimd_f32_t sum = ai + bi;
        simsimd_f32_to_f16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_f16_haswell(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_f16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m256 a_vec = _mm256_cvtph_ps(a_f16);
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        __m128i sum_f16 = _mm256_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), sum_f16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai;
        simsimd_f16_to_f32(a + i, &ai);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        simsimd_f32_to_f16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_f16_haswell(                         //
    simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f16_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f16_haswell(a, n, alpha, 0, result); }
        else { simsimd_scale_f16_haswell(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_f16 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_vec = _mm256_cvtph_ps(a_f16);
        __m256 b_vec = _mm256_cvtph_ps(b_f16);
        __m256 a_scaled_vec = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
        __m128i sum_f16 = _mm256_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), sum_f16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi;
        simsimd_f16_to_f32(a + i, &ai);
        simsimd_f16_to_f32(b + i, &bi);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        simsimd_f32_to_f16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_bf16_haswell(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                             simsimd_bf16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_bf16 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_vec = _simsimd_bf16x8_to_f32x8_haswell(a_bf16);
        __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell(b_bf16);
        __m256 sum_vec = _mm256_add_ps(a_vec, b_vec);
        __m128i sum_bf16 = _simsimd_f32x8_to_bf16x8_haswell(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_bf16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_bf16_to_f32(b + i, &bi);
        simsimd_f32_t sum = ai + bi;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_bf16_haswell(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                               simsimd_distance_t beta, simsimd_bf16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m256 a_vec = _simsimd_bf16x8_to_f32x8_haswell(a_bf16);
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        __m128i sum_bf16 = _simsimd_f32x8_to_bf16x8_haswell(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_bf16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_bf16_haswell(                          //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_bf16_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_bf16_haswell(a, n, alpha, 0, result); }
        else { simsimd_scale_bf16_haswell(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_bf16 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m256 a_vec = _simsimd_bf16x8_to_f32x8_haswell(a_bf16);
        __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell(b_bf16);
        __m256 a_scaled_vec = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
        __m128i sum_bf16 = _simsimd_f32x8_to_bf16x8_haswell(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_bf16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_bf16_to_f32(b + i, &bi);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_f32_haswell(                                //
    simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 c_vec = _mm256_loadu_ps(c + i);
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        _mm256_storeu_ps(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f64_haswell(                                //
    simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_loadu_pd(a + i);
        __m256d b_vec = _mm256_loadu_pd(b + i);
        __m256d c_vec = _mm256_loadu_pd(c + i);
        __m256d ab_vec = _mm256_mul_pd(a_vec, b_vec);
        __m256d ab_scaled_vec = _mm256_mul_pd(ab_vec, alpha_vec);
        __m256d sum_vec = _mm256_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
        _mm256_storeu_pd(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha * a[i] * b[i] + beta * c[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f16_haswell(                                //
    simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_f16 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m128i c_f16 = _mm_lddqu_si128((__m128i const *)(c + i));
        __m256 a_vec = _mm256_cvtph_ps(a_f16);
        __m256 b_vec = _mm256_cvtph_ps(b_f16);
        __m256 c_vec = _mm256_cvtph_ps(c_f16);
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        __m128i sum_f16 = _mm256_cvtps_ph(sum_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(result + i), sum_f16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi, ci;
        simsimd_f16_to_f32(a + i, &ai);
        simsimd_f16_to_f32(b + i, &bi);
        simsimd_f16_to_f32(c + i, &ci);
        simsimd_f32_t sum = alpha * ai * bi + beta * ci;
        simsimd_f32_to_f16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_bf16_haswell(                                  //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_bf16 = _mm_lddqu_si128((__m128i const *)(a + i));
        __m128i b_bf16 = _mm_lddqu_si128((__m128i const *)(b + i));
        __m128i c_bf16 = _mm_lddqu_si128((__m128i const *)(c + i));
        __m256 a_vec = _simsimd_bf16x8_to_f32x8_haswell(a_bf16);
        __m256 b_vec = _simsimd_bf16x8_to_f32x8_haswell(b_bf16);
        __m256 c_vec = _simsimd_bf16x8_to_f32x8_haswell(c_bf16);
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        __m128i sum_bf16 = _simsimd_f32x8_to_bf16x8_haswell(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_bf16);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi, ci;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_bf16_to_f32(b + i, &bi);
        simsimd_bf16_to_f32(c + i, &ci);
        simsimd_f32_t sum = alpha * ai * bi + beta * ci;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_i8_haswell(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                           simsimd_i8_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi8(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        simsimd_f32_t sum = ai + bi;
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_i8_haswell(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                             simsimd_distance_t beta, simsimd_i8_t *result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    int sum_i32s[8], a_i32s[8];

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        // The normal part.
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        // Instead of serial calls to expensive `_simsimd_f32_to_u8`, convert and clip with SIMD.
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(-128));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, sum_i32_vec);
        result[i + 0] = (simsimd_i8_t)sum_i32s[0];
        result[i + 1] = (simsimd_i8_t)sum_i32s[1];
        result[i + 2] = (simsimd_i8_t)sum_i32s[2];
        result[i + 3] = (simsimd_i8_t)sum_i32s[3];
        result[i + 4] = (simsimd_i8_t)sum_i32s[4];
        result[i + 5] = (simsimd_i8_t)sum_i32s[5];
        result[i + 6] = (simsimd_i8_t)sum_i32s[6];
        result[i + 7] = (simsimd_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i];
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_i8_haswell(                        //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_i8_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_i8_haswell(a, n, alpha, 0, result); }
        else { simsimd_scale_i8_haswell(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    int sum_i32s[8], a_i32s[8], b_i32s[8];

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        // The normal part.
        __m256 a_scaled_vec = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
        // Instead of serial calls to expensive `_simsimd_f32_to_u8`, convert and clip with SIMD.
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(-128));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, sum_i32_vec);
        result[i + 0] = (simsimd_i8_t)sum_i32s[0];
        result[i + 1] = (simsimd_i8_t)sum_i32s[1];
        result[i + 2] = (simsimd_i8_t)sum_i32s[2];
        result[i + 3] = (simsimd_i8_t)sum_i32s[3];
        result[i + 4] = (simsimd_i8_t)sum_i32s[4];
        result[i + 5] = (simsimd_i8_t)sum_i32s[5];
        result[i + 6] = (simsimd_i8_t)sum_i32s[6];
        result[i + 7] = (simsimd_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_u8_haswell(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                           simsimd_u8_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu8(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        simsimd_f32_t sum = ai + bi;
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_u8_haswell(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                             simsimd_distance_t beta, simsimd_u8_t *result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    int sum_i32s[8], a_i32s[8];

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        // The normal part.
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        // Instead of serial calls to expensive `_simsimd_f32_to_u8`, convert and clip with SIMD.
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(0));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, sum_i32_vec);
        result[i + 0] = (simsimd_u8_t)sum_i32s[0];
        result[i + 1] = (simsimd_u8_t)sum_i32s[1];
        result[i + 2] = (simsimd_u8_t)sum_i32s[2];
        result[i + 3] = (simsimd_u8_t)sum_i32s[3];
        result[i + 4] = (simsimd_u8_t)sum_i32s[4];
        result[i + 5] = (simsimd_u8_t)sum_i32s[5];
        result[i + 6] = (simsimd_u8_t)sum_i32s[6];
        result[i + 7] = (simsimd_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i];
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_u8_haswell(                        //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_u8_haswell(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_u8_haswell(a, n, alpha, 0, result); }
        else { simsimd_scale_u8_haswell(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    int sum_i32s[8], a_i32s[8], b_i32s[8];

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        // The normal part.
        __m256 a_scaled_vec = _mm256_mul_ps(a_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
        // Instead of serial calls to expensive `_simsimd_f32_to_u8`, convert and clip with SIMD.
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(0));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, sum_i32_vec);
        result[i + 0] = (simsimd_u8_t)sum_i32s[0];
        result[i + 1] = (simsimd_u8_t)sum_i32s[1];
        result[i + 2] = (simsimd_u8_t)sum_i32s[2];
        result[i + 3] = (simsimd_u8_t)sum_i32s[3];
        result[i + 4] = (simsimd_u8_t)sum_i32s[4];
        result[i + 5] = (simsimd_u8_t)sum_i32s[5];
        result[i + 6] = (simsimd_u8_t)sum_i32s[6];
        result[i + 7] = (simsimd_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i];
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i8_haswell(                                                //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    int sum_i32s[8], a_i32s[8], b_i32s[8], c_i32s[8];

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        c_i32s[0] = c[i + 0], c_i32s[1] = c[i + 1], c_i32s[2] = c[i + 2], c_i32s[3] = c[i + 3], //
            c_i32s[4] = c[i + 4], c_i32s[5] = c[i + 5], c_i32s[6] = c[i + 6], c_i32s[7] = c[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        __m256 c_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)c_i32s));
        // The normal part.
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        // Instead of serial calls to expensive `_simsimd_f32_to_u8`, convert and clip with SIMD.
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(-128));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(127));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, sum_i32_vec);
        result[i + 0] = (simsimd_i8_t)sum_i32s[0];
        result[i + 1] = (simsimd_i8_t)sum_i32s[1];
        result[i + 2] = (simsimd_i8_t)sum_i32s[2];
        result[i + 3] = (simsimd_i8_t)sum_i32s[3];
        result[i + 4] = (simsimd_i8_t)sum_i32s[4];
        result[i + 5] = (simsimd_i8_t)sum_i32s[5];
        result[i + 6] = (simsimd_i8_t)sum_i32s[6];
        result[i + 7] = (simsimd_i8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i], ci = c[i];
        simsimd_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u8_haswell(                                                //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);
    int sum_i32s[8], a_i32s[8], b_i32s[8], c_i32s[8];

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        //? Handling loads and stores with SIMD is tricky. Not because of upcasting, but the
        //? downcasting at the end of the loop. In AVX2 it's a drag! Keep it for another day.
        a_i32s[0] = a[i + 0], a_i32s[1] = a[i + 1], a_i32s[2] = a[i + 2], a_i32s[3] = a[i + 3], //
            a_i32s[4] = a[i + 4], a_i32s[5] = a[i + 5], a_i32s[6] = a[i + 6], a_i32s[7] = a[i + 7];
        b_i32s[0] = b[i + 0], b_i32s[1] = b[i + 1], b_i32s[2] = b[i + 2], b_i32s[3] = b[i + 3], //
            b_i32s[4] = b[i + 4], b_i32s[5] = b[i + 5], b_i32s[6] = b[i + 6], b_i32s[7] = b[i + 7];
        c_i32s[0] = c[i + 0], c_i32s[1] = c[i + 1], c_i32s[2] = c[i + 2], c_i32s[3] = c[i + 3], //
            c_i32s[4] = c[i + 4], c_i32s[5] = c[i + 5], c_i32s[6] = c[i + 6], c_i32s[7] = c[i + 7];
        //! This can be done at least 50% faster if we convert 8-bit integers to floats instead
        //! of relying on the slow `_mm256_cvtepi32_ps` instruction.
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)a_i32s));
        __m256 b_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)b_i32s));
        __m256 c_vec = _mm256_cvtepi32_ps(_mm256_lddqu_si256((__m256i *)c_i32s));
        // The normal part.
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        // Instead of serial calls to expensive `_simsimd_f32_to_u8`, convert and clip with SIMD.
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(0));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(255));
        // Export into a serial buffer.
        _mm256_storeu_si256((__m256i *)sum_i32s, sum_i32_vec);
        result[i + 0] = (simsimd_u8_t)sum_i32s[0];
        result[i + 1] = (simsimd_u8_t)sum_i32s[1];
        result[i + 2] = (simsimd_u8_t)sum_i32s[2];
        result[i + 3] = (simsimd_u8_t)sum_i32s[3];
        result[i + 4] = (simsimd_u8_t)sum_i32s[4];
        result[i + 5] = (simsimd_u8_t)sum_i32s[5];
        result[i + 6] = (simsimd_u8_t)sum_i32s[6];
        result[i + 7] = (simsimd_u8_t)sum_i32s[7];
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i], ci = c[i];
        simsimd_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_i16_haswell(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_size_t n,
                                            simsimd_i16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi16(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_i64_t ai = a[i], bi = b[i];
        simsimd_i64_t sum = ai + bi;
        _simsimd_i64_to_i16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_i16_haswell(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_i16_t *result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(-32768));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(32767));
        // Casting down to 16-bit integers is tricky!
        __m128i sum_i16_vec =
            _mm_packs_epi32(_mm256_castsi256_si128(sum_i32_vec), _mm256_extracti128_si256(sum_i32_vec, 1));
        _mm_storeu_si128((__m128i *)(result + i), sum_i16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i];
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        _simsimd_f32_to_i16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i16_haswell(                                                  //
    simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 b_vec = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(b + i))));
        __m256 c_vec = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_lddqu_si128((__m128i *)(c + i))));
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_set1_epi32(-32768));
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(32767));
        // Casting down to 16-bit integers is tricky!
        __m128i sum_i16_vec =
            _mm_packs_epi32(_mm256_castsi256_si128(sum_i32_vec), _mm256_extracti128_si256(sum_i32_vec, 1));
        _mm_storeu_si128((__m128i *)(result + i), sum_i16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i], ci = c[i];
        simsimd_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        _simsimd_f32_to_i16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_u16_haswell(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_size_t n,
                                            simsimd_u16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu16(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_u64_t ai = a[i], bi = b[i];
        simsimd_u64_t sum = ai + bi;
        _simsimd_u64_to_u16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_u16_haswell(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_u16_t *result) {

    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 sum_vec = _mm256_fmadd_ps(a_vec, alpha_vec, beta_vec);
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_setzero_si256());
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(65535));
        // Casting down to 16-bit integers is tricky!
        __m128i sum_u16_vec =
            _mm_packs_epi32(_mm256_castsi256_si128(sum_i32_vec), _mm256_extracti128_si256(sum_i32_vec, 1));
        _mm_storeu_si128((__m128i *)(result + i), sum_u16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i];
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        _simsimd_f32_to_u16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u16_haswell(                                                  //
    simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    __m256 alpha_vec = _mm256_set1_ps(alpha_f32);
    __m256 beta_vec = _mm256_set1_ps(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(a + i))));
        __m256 b_vec = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(b + i))));
        __m256 c_vec = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_lddqu_si128((__m128i *)(c + i))));
        __m256 ab_vec = _mm256_mul_ps(a_vec, b_vec);
        __m256 ab_scaled_vec = _mm256_mul_ps(ab_vec, alpha_vec);
        __m256 sum_vec = _mm256_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
        __m256i sum_i32_vec = _mm256_cvtps_epi32(sum_vec);
        sum_i32_vec = _mm256_max_epi32(sum_i32_vec, _mm256_setzero_si256());
        sum_i32_vec = _mm256_min_epi32(sum_i32_vec, _mm256_set1_epi32(65535));
        // Casting down to 16-bit integers is tricky!
        __m128i sum_u16_vec =
            _mm_packs_epi32(_mm256_castsi256_si128(sum_i32_vec), _mm256_extracti128_si256(sum_i32_vec, 1));
        _mm_storeu_si128((__m128i *)(result + i), sum_u16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai = a[i], bi = b[i], ci = c[i];
        simsimd_f32_t sum = alpha_f32 * ai * bi + beta_f32 * ci;
        _simsimd_f32_to_u16(&sum, result + i);
    }
}

SIMSIMD_INTERNAL __m256i _mm256_adds_epi32_haswell(__m256i a, __m256i b) {
    // There are many flavors of addition with saturation in AVX2: i8, u8, i16, and u16.
    // But not for larger numeric types. We have to do it manually.
    __m256i result = _mm256_add_epi32(a, b);
    // Detect positive overflow
    __m256i overflow_mask = _mm256_and_si256(          //
        _mm256_cmpgt_epi32(a, _mm256_setzero_si256()), // a > 0
        _mm256_cmpgt_epi32(b, _mm256_setzero_si256())  // b > 0
    );
    overflow_mask = _mm256_and_si256(overflow_mask, _mm256_cmpgt_epi32(result, a));

    // Detect negative overflow
    __m256i underflow_mask = _mm256_and_si256(         //
        _mm256_cmpgt_epi32(_mm256_setzero_si256(), a), // a < 0
        _mm256_cmpgt_epi32(_mm256_setzero_si256(), b)  // b < 0
    );
    underflow_mask = _mm256_and_si256(underflow_mask, _mm256_cmpgt_epi32(a, result));
    // Set overflowed results to INT32_MAX
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(2147483647), overflow_mask);
    // Set underflowed results to INT32_MIN
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(-2147483647 - 1), underflow_mask);
    return result;
}

SIMSIMD_PUBLIC void simsimd_sum_i32_haswell(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_size_t n,
                                            simsimd_i32_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epi32_haswell(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_i64_t ai = a[i], bi = b[i];
        simsimd_i64_t sum = ai + bi;
        _simsimd_i64_to_i32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_i32_haswell(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_i32_t *result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d sum_vec = _mm256_fmadd_pd(a_vec, alpha_vec, beta_vec);
        __m128i sum_i32_vec = _mm256_cvtpd_epi32(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_i32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i];
        simsimd_f64_t sum = alpha * ai + beta;
        _simsimd_f64_to_i32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i32_haswell(                                                  //
    simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d b_vec = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(b + i)));
        __m256d c_vec = _mm256_cvtepi32_pd(_mm_lddqu_si128((__m128i *)(c + i)));
        __m256d ab_vec = _mm256_mul_pd(a_vec, b_vec);
        __m256d ab_scaled_vec = _mm256_mul_pd(ab_vec, alpha_vec);
        __m256d sum_vec = _mm256_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
        __m128i sum_i32_vec = _mm256_cvtpd_epi32(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_i32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i], bi = b[i], ci = c[i];
        simsimd_f64_t sum = alpha * ai * bi + beta * ci;
        _simsimd_f64_to_i32(&sum, result + i);
    }
}

SIMSIMD_INTERNAL __m256i _mm256_adds_epu32_haswell(__m256i a, __m256i b) {
    __m256i result = _mm256_add_epi32(a, b);
    // Detect overflow: if result < a, then an overflow occurred (for unsigned addition).
    __m256i overflow_mask = _mm256_cmpgt_epi32(a, result);
    // Set overflowed results to UINT32_MAX (0xFFFFFFFF)
    result = _mm256_blendv_epi8(result, _mm256_set1_epi32(4294967295u), overflow_mask);
    return result;
}

SIMSIMD_INTERNAL __m256d _mm256_cvtepu32_pd_haswell(__m128i a) {
    // Converting unsigned 32-bit integers to double-precision floats isn't trivial in AVX2.
    // Let's convert the lower 31 bits to a double-precision float.
    // And then conditionally add 2^31 to the result if the MSB is set.
    __m256d result = _mm256_cvtepi32_pd(_mm_and_si128(a, _mm_set1_epi32(0x7FFFFFFF)));
    int should_increment = (_mm_movemask_epi8(a) & 0x8888);
    should_increment = should_increment / 0x8888; // Transform something like 0b1000100010001000 to 0b1111
    __m256d incremented = _mm256_add_pd(result, _mm256_set1_pd(2147483648.0));
    result = _mm256_blend_pd(result, incremented, should_increment);
    return result;
}

SIMSIMD_INTERNAL __m128i _mm256_cvtpd_epu32_haswell(__m256d a) {
    //? For now let's avoid SIMD and just use serial conversion.
    simsimd_u32_t result[4];
    result[0] = (simsimd_u32_t)a[0];
    result[1] = (simsimd_u32_t)a[1];
    result[2] = (simsimd_u32_t)a[2];
    result[3] = (simsimd_u32_t)a[3];
    return _mm_lddqu_si128((__m128i *)result);
}

SIMSIMD_PUBLIC void simsimd_sum_u32_haswell(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                            simsimd_u32_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_vec = _mm256_lddqu_si256((__m256i *)(a + i));
        __m256i b_vec = _mm256_lddqu_si256((__m256i *)(b + i));
        __m256i sum_vec = _mm256_adds_epu32_haswell(a_vec, b_vec);
        _mm256_storeu_si256((__m256i *)(result + i), sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_i64_t ai = a[i], bi = b[i];
        simsimd_i64_t sum = ai + bi;
        _simsimd_i64_to_u32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_u32_haswell(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_u32_t *result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d sum_vec = _mm256_fmadd_pd(a_vec, alpha_vec, beta_vec);
        __m128i sum_u32_vec = _mm256_cvtpd_epu32_haswell(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_u32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i];
        simsimd_f64_t sum = alpha * ai + beta;
        _simsimd_f64_to_u32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u32_haswell(                                                  //
    simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result) {
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    __m256d beta_vec = _mm256_set1_pd(beta);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_vec = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(a + i)));
        __m256d b_vec = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(b + i)));
        __m256d c_vec = _mm256_cvtepu32_pd_haswell(_mm_lddqu_si128((__m128i *)(c + i)));
        __m256d ab_vec = _mm256_mul_pd(a_vec, b_vec);
        __m256d ab_scaled_vec = _mm256_mul_pd(ab_vec, alpha_vec);
        __m256d sum_vec = _mm256_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
        __m128i sum_u32_vec = _mm256_cvtpd_epu32_haswell(sum_vec);
        _mm_storeu_si128((__m128i *)(result + i), sum_u32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t ai = a[i], bi = b[i], ci = c[i];
        simsimd_f64_t sum = alpha * ai * bi + beta * ci;
        _simsimd_f64_to_u32(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_HASWELL

#if SIMSIMD_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_sum_f64_skylake(simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n,
                                            simsimd_f64_t *result) {
    __m512d a_vec, b_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_sum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    sum_vec = _mm512_add_pd(a_vec, b_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto simsimd_sum_f64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_f64_skylake(simsimd_f64_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_f64_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512d a_vec, b_vec, a_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_scale_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        a += 8, n -= 8;
    }
    sum_vec = _mm512_fmadd_pd(a_vec, alpha_vec, beta_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto simsimd_scale_f64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_wsum_f64_skylake(                         //
    simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f64_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f64_skylake(a, n, alpha, 0, result); }
        else { simsimd_scale_f64_skylake(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512d a_vec, b_vec, a_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_wsum_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        a += 8, b += 8, n -= 8;
    }
    a_scaled_vec = _mm512_mul_pd(a_vec, alpha_vec);
    sum_vec = _mm512_fmadd_pd(b_vec, beta_vec, a_scaled_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto simsimd_wsum_f64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_sum_f32_skylake(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                            simsimd_f32_t *result) {
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

simsimd_sum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto simsimd_sum_f32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_f32_skylake(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_f32_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m512 a_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

simsimd_scale_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        a += 16, n -= 16;
    }
    sum_vec = _mm512_fmadd_ps(a_vec, alpha_vec, beta_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto simsimd_scale_f32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_wsum_f32_skylake(                         //
    simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f32_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f32_skylake(a, n, alpha, 0, result); }
        else { simsimd_scale_f32_skylake(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m512 a_vec, b_vec, a_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_wsum_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        a += 16, b += 16, n -= 16;
    }
    a_scaled_vec = _mm512_mul_ps(a_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto simsimd_wsum_f32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_sum_bf16_skylake(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                             simsimd_bf16_t *result) {
    __m256i a_bf16_vec, b_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_sum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16_vec = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        b_bf16_vec = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_vec = _simsimd_bf16x16_to_f32x16_skylake(a_bf16_vec);
    b_vec = _simsimd_bf16x16_to_f32x16_skylake(b_bf16_vec);
    sum_vec = _mm512_add_ps(a_vec, b_vec);
    sum_bf16_vec = _simsimd_f32x16_to_bf16x16_skylake(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto simsimd_sum_bf16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_bf16_skylake(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                               simsimd_distance_t beta, simsimd_bf16_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_bf16_vec, b_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_wsum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        a += 16, n -= 16;
    }
    a_vec = _simsimd_bf16x16_to_f32x16_skylake(a_bf16_vec);
    sum_vec = _mm512_fmadd_ps(a_vec, alpha_vec, beta_vec);
    sum_bf16_vec = _simsimd_f32x16_to_bf16x16_skylake(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto simsimd_wsum_bf16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_wsum_bf16_skylake(                          //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_bf16_skylake(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_bf16_skylake(a, n, alpha, 0, result); }
        else { simsimd_scale_bf16_skylake(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_bf16_vec, b_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, a_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_wsum_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16_vec = _mm256_maskz_loadu_epi16(mask, b);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        b_bf16_vec = _mm256_loadu_epi16(b);
        a += 16, b += 16, n -= 16;
    }
    a_vec = _simsimd_bf16x16_to_f32x16_skylake(a_bf16_vec);
    b_vec = _simsimd_bf16x16_to_f32x16_skylake(b_bf16_vec);
    a_scaled_vec = _mm512_mul_ps(a_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(b_vec, beta_vec, a_scaled_vec);
    sum_bf16_vec = _simsimd_f32x16_to_bf16x16_skylake(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto simsimd_wsum_bf16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_f64_skylake(                                                  //
    simsimd_f64_t const *a, simsimd_f64_t const *b, simsimd_f64_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f64_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512d a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_fma_f64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_pd(mask, a);
        b_vec = _mm512_maskz_loadu_pd(mask, b);
        c_vec = _mm512_maskz_loadu_pd(mask, c);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_pd(a);
        b_vec = _mm512_loadu_pd(b);
        c_vec = _mm512_loadu_pd(c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    ab_vec = _mm512_mul_pd(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_pd(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
    _mm512_mask_storeu_pd(result, mask, sum_vec);
    result += 8;
    if (n) goto simsimd_fma_f64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_f32_skylake(                                                  //
    simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m512 a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_fma_f32_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_vec = _mm512_maskz_loadu_ps(mask, a);
        b_vec = _mm512_maskz_loadu_ps(mask, b);
        c_vec = _mm512_maskz_loadu_ps(mask, c);
        n = 0;
    }
    else {
        a_vec = _mm512_loadu_ps(a);
        b_vec = _mm512_loadu_ps(b);
        c_vec = _mm512_loadu_ps(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    ab_vec = _mm512_mul_ps(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_ps(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
    _mm512_mask_storeu_ps(result, mask, sum_vec);
    result += 16;
    if (n) goto simsimd_fma_f32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_bf16_skylake(                                                    //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_bf16_vec, b_bf16_vec, c_bf16_vec, sum_bf16_vec;
    __m512 a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_fma_bf16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_bf16_vec = _mm256_maskz_loadu_epi16(mask, b);
        c_bf16_vec = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_bf16_vec = _mm256_loadu_epi16(a);
        b_bf16_vec = _mm256_loadu_epi16(b);
        c_bf16_vec = _mm256_loadu_epi16(c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_vec = _simsimd_bf16x16_to_f32x16_skylake(a_bf16_vec);
    b_vec = _simsimd_bf16x16_to_f32x16_skylake(b_bf16_vec);
    c_vec = _simsimd_bf16x16_to_f32x16_skylake(c_bf16_vec);
    ab_vec = _mm512_mul_ps(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_ps(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
    sum_bf16_vec = _simsimd_f32x16_to_bf16x16_skylake(sum_vec);
    _mm256_mask_storeu_epi16(result, mask, sum_bf16_vec);
    result += 16;
    if (n) goto simsimd_fma_bf16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_i8_skylake(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                             simsimd_distance_t beta, simsimd_i8_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m128i a_i8_vec, sum_i8_vec;
    __m512 a_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

simsimd_scale_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8_vec = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_i8_vec = _mm_lddqu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8_vec));
    sum_vec = _mm512_fmadd_ps(a_vec, alpha_vec, beta_vec);
    sum_i8_vec = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(sum_vec));
    _mm_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 16;
    if (n) goto simsimd_scale_i8_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_i8_skylake(                                                //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m128i a_i8_vec, b_i8_vec, c_i8_vec, sum_i8_vec;
    __m512 a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_fma_i8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i8_vec = _mm_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm_maskz_loadu_epi8(mask, b);
        c_i8_vec = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_i8_vec = _mm_lddqu_si128((__m128i *)a);
        b_i8_vec = _mm_lddqu_si128((__m128i *)b);
        c_i8_vec = _mm_lddqu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(a_i8_vec));
    b_vec = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b_i8_vec));
    c_vec = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c_i8_vec));
    ab_vec = _mm512_mul_ps(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_ps(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
    sum_i8_vec = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(sum_vec));
    _mm_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 16;
    if (n) goto simsimd_fma_i8_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_u8_skylake(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                             simsimd_distance_t beta, simsimd_u8_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m128i a_u8_vec, sum_u8_vec;
    __m512 a_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

simsimd_scale_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8_vec = _mm_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8_vec = _mm_lddqu_si128((__m128i *)a);
        a += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8_vec));
    sum_vec = _mm512_fmadd_ps(a_vec, alpha_vec, beta_vec);
    sum_u8_vec = _mm512_cvtepi32_epi8(_mm512_cvtps_epu32(sum_vec));
    _mm_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 16;
    if (n) goto simsimd_scale_u8_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_u8_skylake(                                                //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m128i a_u8_vec, b_u8_vec, c_u8_vec, sum_u8_vec;
    __m512 a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_fma_u8_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u8_vec = _mm_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm_maskz_loadu_epi8(mask, b);
        c_u8_vec = _mm_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8_vec = _mm_lddqu_si128((__m128i *)a);
        b_u8_vec = _mm_lddqu_si128((__m128i *)b);
        c_u8_vec = _mm_lddqu_si128((__m128i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(a_u8_vec));
    b_vec = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(b_u8_vec));
    c_vec = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c_u8_vec));
    ab_vec = _mm512_mul_ps(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_ps(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
    sum_u8_vec = _mm512_cvtepi32_epi8(_mm512_cvtps_epu32(sum_vec));
    _mm_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 16;
    if (n) goto simsimd_fma_u8_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_i16_skylake(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_i16_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_i16_vec, sum_i16_vec;
    __m512 a_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

simsimd_scale_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_i16_vec = _mm256_lddqu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16_vec));
    sum_vec = _mm512_fmadd_ps(a_vec, alpha_vec, beta_vec);
    sum_i16_vec = _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(sum_vec));
    _mm256_mask_storeu_epi16(result, mask, sum_i16_vec);
    result += 16;
    if (n) goto simsimd_scale_i16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_i16_skylake(                                                  //
    simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_i16_vec, b_i16_vec, c_i16_vec, sum_i16_vec;
    __m512 a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_fma_i16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_i16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_i16_vec = _mm256_maskz_loadu_epi16(mask, b);
        c_i16_vec = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_i16_vec = _mm256_lddqu_si256((__m256i *)a);
        b_i16_vec = _mm256_lddqu_si256((__m256i *)b);
        c_i16_vec = _mm256_lddqu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(a_i16_vec));
    b_vec = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(b_i16_vec));
    c_vec = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(c_i16_vec));
    ab_vec = _mm512_mul_ps(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_ps(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
    sum_i16_vec = _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(sum_vec));
    _mm256_mask_storeu_epi16(result, mask, sum_i16_vec);
    result += 16;
    if (n) goto simsimd_fma_i16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_u16_skylake(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_u16_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_u16_vec, sum_u16_vec;
    __m512 a_vec, sum_vec;
    __mmask16 mask = 0xFFFF;

simsimd_scale_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16_vec = _mm256_maskz_loadu_epi16(mask, a);
        n = 0;
    }
    else {
        a_u16_vec = _mm256_lddqu_si256((__m256i *)a);
        a += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16_vec));
    sum_vec = _mm512_fmadd_ps(a_vec, alpha_vec, beta_vec);
    sum_u16_vec = _mm512_cvtepi32_epi16(_mm512_cvtps_epu32(sum_vec));
    _mm256_mask_storeu_epi16(result, mask, sum_u16_vec);
    result += 16;
    if (n) goto simsimd_scale_u16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_u16_skylake(                                                  //
    simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    __m256i a_u16_vec, b_u16_vec, c_u16_vec, sum_u16_vec;
    __m512 a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask16 mask = 0xFFFF;
simsimd_fma_u16_skylake_cycle:
    if (n < 16) {
        mask = (__mmask16)_bzhi_u32(0xFFFFFFFF, n);
        a_u16_vec = _mm256_maskz_loadu_epi16(mask, a);
        b_u16_vec = _mm256_maskz_loadu_epi16(mask, b);
        c_u16_vec = _mm256_maskz_loadu_epi16(mask, c);
        n = 0;
    }
    else {
        a_u16_vec = _mm256_lddqu_si256((__m256i *)a);
        b_u16_vec = _mm256_lddqu_si256((__m256i *)b);
        c_u16_vec = _mm256_lddqu_si256((__m256i *)c);
        a += 16, b += 16, c += 16, n -= 16;
    }
    a_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(a_u16_vec));
    b_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(b_u16_vec));
    c_vec = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(c_u16_vec));
    ab_vec = _mm512_mul_ps(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_ps(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_ps(c_vec, beta_vec, ab_scaled_vec);
    sum_u16_vec = _mm512_cvtepi32_epi16(_mm512_cvtps_epu32(sum_vec));
    _mm256_mask_storeu_epi16(result, mask, sum_u16_vec);
    result += 16;
    if (n) goto simsimd_fma_u16_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_i32_skylake(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_i32_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m256i a_i32_vec, sum_i32_vec;
    __m512d a_vec, sum_vec;
    __mmask8 mask = 0xFF;

simsimd_scale_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32_vec = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_i32_vec = _mm256_lddqu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_vec = _mm512_cvtepi32_pd(a_i32_vec);
    sum_vec = _mm512_fmadd_pd(a_vec, alpha_vec, beta_vec);
    sum_i32_vec = _mm512_cvtpd_epi32(sum_vec);
    _mm256_mask_storeu_epi32(result, mask, sum_i32_vec);
    result += 8;
    if (n) goto simsimd_scale_i32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_i32_skylake(                                                  //
    simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m256i a_i32_vec, b_i32_vec, c_i32_vec, sum_i32_vec;
    __m512d a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_fma_i32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i32_vec = _mm256_maskz_loadu_epi32(mask, a);
        b_i32_vec = _mm256_maskz_loadu_epi32(mask, b);
        c_i32_vec = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_i32_vec = _mm256_lddqu_si256((__m256i *)a);
        b_i32_vec = _mm256_lddqu_si256((__m256i *)b);
        c_i32_vec = _mm256_lddqu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_vec = _mm512_cvtepi32_pd(a_i32_vec);
    b_vec = _mm512_cvtepi32_pd(b_i32_vec);
    c_vec = _mm512_cvtepi32_pd(c_i32_vec);
    ab_vec = _mm512_mul_pd(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_pd(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
    sum_i32_vec = _mm512_cvtpd_epi32(sum_vec);
    _mm256_mask_storeu_epi32(result, mask, sum_i32_vec);
    result += 8;
    if (n) goto simsimd_fma_i32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_u32_skylake(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_u32_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m256i a_u32_vec, sum_u32_vec;
    __m512d a_vec, sum_vec;
    __mmask8 mask = 0xFF;

simsimd_scale_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32_vec = _mm256_maskz_loadu_epi32(mask, a);
        n = 0;
    }
    else {
        a_u32_vec = _mm256_lddqu_si256((__m256i *)a);
        a += 8, n -= 8;
    }
    a_vec = _mm512_cvtepu32_pd(a_u32_vec);
    sum_vec = _mm512_fmadd_pd(a_vec, alpha_vec, beta_vec);
    sum_u32_vec = _mm512_cvtpd_epu32(sum_vec);
    _mm256_mask_storeu_epi32(result, mask, sum_u32_vec);
    result += 8;
    if (n) goto simsimd_scale_u32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_u32_skylake(                                                  //
    simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m256i a_u32_vec, b_u32_vec, c_u32_vec, sum_u32_vec;
    __m512d a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_fma_u32_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u32_vec = _mm256_maskz_loadu_epi32(mask, a);
        b_u32_vec = _mm256_maskz_loadu_epi32(mask, b);
        c_u32_vec = _mm256_maskz_loadu_epi32(mask, c);
        n = 0;
    }
    else {
        a_u32_vec = _mm256_lddqu_si256((__m256i *)a);
        b_u32_vec = _mm256_lddqu_si256((__m256i *)b);
        c_u32_vec = _mm256_lddqu_si256((__m256i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_vec = _mm512_cvtepu32_pd(a_u32_vec);
    b_vec = _mm512_cvtepu32_pd(b_u32_vec);
    c_vec = _mm512_cvtepu32_pd(c_u32_vec);
    ab_vec = _mm512_mul_pd(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_pd(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
    sum_u32_vec = _mm512_cvtpd_epu32(sum_vec);
    _mm256_mask_storeu_epi32(result, mask, sum_u32_vec);
    result += 8;
    if (n) goto simsimd_fma_u32_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_i64_skylake(simsimd_i64_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_i64_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512i a_i64_vec, sum_i64_vec;
    __m512d a_vec, sum_vec;
    __mmask8 mask = 0xFF;

simsimd_scale_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64_vec = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_i64_vec = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_vec = _mm512_cvtepi64_pd(a_i64_vec);
    sum_vec = _mm512_fmadd_pd(a_vec, alpha_vec, beta_vec);
    sum_i64_vec = _mm512_cvtpd_epi64(sum_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_i64_vec);
    result += 8;
    if (n) goto simsimd_scale_i64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_i64_skylake(                                                  //
    simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512i a_i64_vec, b_i64_vec, c_i64_vec, sum_i64_vec;
    __m512d a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_fma_i64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_i64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_i64_vec = _mm512_maskz_loadu_epi64(mask, b);
        c_i64_vec = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_i64_vec = _mm512_loadu_si512((__m512i *)a);
        b_i64_vec = _mm512_loadu_si512((__m512i *)b);
        c_i64_vec = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_vec = _mm512_cvtepi64_pd(a_i64_vec);
    b_vec = _mm512_cvtepi64_pd(b_i64_vec);
    c_vec = _mm512_cvtepi64_pd(c_i64_vec);
    ab_vec = _mm512_mul_pd(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_pd(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
    sum_i64_vec = _mm512_cvtpd_epi64(sum_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_i64_vec);
    result += 8;
    if (n) goto simsimd_fma_i64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_u64_skylake(simsimd_u64_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_u64_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512i a_u64_vec, sum_u64_vec;
    __m512d a_vec, sum_vec;
    __mmask8 mask = 0xFF;

simsimd_scale_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64_vec = _mm512_maskz_loadu_epi64(mask, a);
        n = 0;
    }
    else {
        a_u64_vec = _mm512_loadu_si512((__m512i *)a);
        a += 8, n -= 8;
    }
    a_vec = _mm512_cvtepu64_pd(a_u64_vec);
    sum_vec = _mm512_fmadd_pd(a_vec, alpha_vec, beta_vec);
    sum_u64_vec = _mm512_cvtpd_epu64(sum_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_u64_vec);
    result += 8;
    if (n) goto simsimd_scale_u64_skylake_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_u64_skylake(                                                  //
    simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result) {
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    __m512d beta_vec = _mm512_set1_pd(beta);
    __m512i a_u64_vec, b_u64_vec, c_u64_vec, sum_u64_vec;
    __m512d a_vec, b_vec, c_vec, ab_vec, ab_scaled_vec, sum_vec;
    __mmask8 mask = 0xFF;
simsimd_fma_u64_skylake_cycle:
    if (n < 8) {
        mask = (__mmask8)_bzhi_u32(0xFFFFFFFF, n);
        a_u64_vec = _mm512_maskz_loadu_epi64(mask, a);
        b_u64_vec = _mm512_maskz_loadu_epi64(mask, b);
        c_u64_vec = _mm512_maskz_loadu_epi64(mask, c);
        n = 0;
    }
    else {
        a_u64_vec = _mm512_loadu_si512((__m512i *)a);
        b_u64_vec = _mm512_loadu_si512((__m512i *)b);
        c_u64_vec = _mm512_loadu_si512((__m512i *)c);
        a += 8, b += 8, c += 8, n -= 8;
    }
    a_vec = _mm512_cvtepu64_pd(a_u64_vec);
    b_vec = _mm512_cvtepu64_pd(b_u64_vec);
    c_vec = _mm512_cvtepu64_pd(c_u64_vec);
    ab_vec = _mm512_mul_pd(a_vec, b_vec);
    ab_scaled_vec = _mm512_mul_pd(ab_vec, alpha_vec);
    sum_vec = _mm512_fmadd_pd(c_vec, beta_vec, ab_scaled_vec);
    sum_u64_vec = _mm512_cvtpd_epu64(sum_vec);
    _mm512_mask_storeu_epi64(result, mask, sum_u64_vec);
    result += 8;
    if (n) goto simsimd_fma_u64_skylake_cycle;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SKYLAKE

#if SIMSIMD_TARGET_SAPPHIRE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

SIMSIMD_PUBLIC void simsimd_sum_f16_sapphire(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                             simsimd_f16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512h a_f16_vec, b_f16_vec;
    __m512h sum_f16_vec;
simsimd_sum_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_loadu_ph(a);
        b_f16_vec = _mm512_loadu_ph(b);
        a += 32, b += 32, n -= 32;
    }
    sum_f16_vec = _mm512_add_ph(a_f16_vec, b_f16_vec);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto simsimd_sum_f16_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_f16_sapphire(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                               simsimd_distance_t beta, simsimd_f16_t *result) {
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512h a_f16_vec, b_f16_vec;
    __m512h sum_f16_vec;
simsimd_scale_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_loadu_ph(a);
        a += 32, n -= 32;
    }
    sum_f16_vec = _mm512_fmadd_ph(a_f16_vec, alpha_vec, beta_vec);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto simsimd_scale_f16_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_wsum_f16_sapphire(                        //
    simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f16_sapphire(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f16_sapphire(a, n, alpha, 0, result); }
        else { simsimd_scale_f16_sapphire(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512h a_f16_vec, b_f16_vec;
    __m512h a_scaled_f16_vec, sum_f16_vec;
simsimd_wsum_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_loadu_ph(a);
        b_f16_vec = _mm512_loadu_ph(b);
        a += 32, b += 32, n -= 32;
    }
    a_scaled_f16_vec = _mm512_mul_ph(a_f16_vec, alpha_vec);
    sum_f16_vec = _mm512_fmadd_ph(b_f16_vec, beta_vec, a_scaled_f16_vec);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto simsimd_wsum_f16_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_f16_sapphire(                                                 //
    simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result) {

    __mmask32 mask = 0xFFFFFFFF;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512h a_f16_vec, b_f16_vec, c_f16_vec;
    __m512h ab_f16_vec, ab_scaled_f16_vec, sum_f16_vec;
simsimd_fma_f16_sapphire_cycle:
    if (n < 32) {
        mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, n);
        a_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, a));
        b_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, b));
        c_f16_vec = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, c));
        n = 0;
    }
    else {
        a_f16_vec = _mm512_loadu_ph(a);
        b_f16_vec = _mm512_loadu_ph(b);
        c_f16_vec = _mm512_loadu_ph(c);
        a += 32, b += 32, c += 32, n -= 32;
    }
    ab_f16_vec = _mm512_mul_ph(a_f16_vec, b_f16_vec);
    ab_scaled_f16_vec = _mm512_mul_ph(ab_f16_vec, alpha_vec);
    sum_f16_vec = _mm512_fmadd_ph(c_f16_vec, beta_vec, ab_scaled_f16_vec);
    _mm512_mask_storeu_epi16(result, mask, _mm512_castph_si512(sum_f16_vec));
    result += 32;
    if (n) goto simsimd_fma_f16_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_sum_u8_sapphire(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                            simsimd_u8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512i a_u8_vec, b_u8_vec, sum_u8_vec;
simsimd_sum_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        b_u8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_u8_vec = _mm512_adds_epu8(a_u8_vec, b_u8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto simsimd_sum_u8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_u8_sapphire(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_u8_t *result) {
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512i a_u8_vec, b_u8_vec, sum_u8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec;
    __m512h a_scaled_f16_low_vec, a_scaled_f16_high_vec, sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;
simsimd_scale_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        a += 64, n -= 64;
    }
    // Upcast:
    a_f16_low_vec = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8_vec, _mm512_setzero_si512()));
    a_f16_high_vec = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8_vec, _mm512_setzero_si512()));
    // Scale:
    sum_f16_low_vec = _mm512_fmadd_ph(a_f16_low_vec, alpha_vec, beta_vec);
    sum_f16_high_vec = _mm512_fmadd_ph(a_f16_high_vec, alpha_vec, beta_vec);
    // Downcast:
    sum_i16_low_vec = _mm512_cvtph_epi16(sum_f16_low_vec);
    sum_i16_high_vec = _mm512_cvtph_epi16(sum_f16_high_vec);
    sum_u8_vec = _mm512_packus_epi16(sum_i16_low_vec, sum_i16_high_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto simsimd_scale_u8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_wsum_u8_sapphire(                       //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_u8_sapphire(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_u8_sapphire(a, n, alpha, 0, result); }
        else { simsimd_scale_u8_sapphire(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512i a_u8_vec, b_u8_vec, sum_u8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec, b_f16_low_vec, b_f16_high_vec;
    __m512h a_scaled_f16_low_vec, a_scaled_f16_high_vec, sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;
simsimd_wsum_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        b_u8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    // Upcast:
    a_f16_low_vec = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8_vec, _mm512_setzero_si512()));
    a_f16_high_vec = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8_vec, _mm512_setzero_si512()));
    b_f16_low_vec = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(b_u8_vec, _mm512_setzero_si512()));
    b_f16_high_vec = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(b_u8_vec, _mm512_setzero_si512()));
    // Scale:
    a_scaled_f16_low_vec = _mm512_mul_ph(a_f16_low_vec, alpha_vec);
    a_scaled_f16_high_vec = _mm512_mul_ph(a_f16_high_vec, alpha_vec);
    // Add:
    sum_f16_low_vec = _mm512_fmadd_ph(b_f16_low_vec, beta_vec, a_scaled_f16_low_vec);
    sum_f16_high_vec = _mm512_fmadd_ph(b_f16_high_vec, beta_vec, a_scaled_f16_high_vec);
    // Downcast:
    sum_i16_low_vec = _mm512_cvtph_epi16(sum_f16_low_vec);
    sum_i16_high_vec = _mm512_cvtph_epi16(sum_f16_high_vec);
    sum_u8_vec = _mm512_packus_epi16(sum_i16_low_vec, sum_i16_high_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto simsimd_wsum_u8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_sum_i8_sapphire(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                            simsimd_i8_t *result) {

    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512i a_i8_vec, b_i8_vec, sum_i8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec, b_f16_low_vec, b_f16_high_vec;
    __m512h a_scaled_f16_low_vec, a_scaled_f16_high_vec, sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;

simsimd_sum_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        b_i8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    sum_i8_vec = _mm512_adds_epi8(a_i8_vec, b_i8_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto simsimd_sum_i8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_scale_i8_sapphire(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                              simsimd_distance_t beta, simsimd_i8_t *result) {

    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512i a_i8_vec, sum_i8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec;
    __m512h sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;
simsimd_wsum_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        a += 64, n -= 64;
    }
    // Upcast:
    a_f16_low_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_vec)));
    a_f16_high_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_vec, 1)));
    // Scale:
    sum_f16_low_vec = _mm512_fmadd_ph(a_f16_low_vec, alpha_vec, beta_vec);
    sum_f16_high_vec = _mm512_fmadd_ph(a_f16_high_vec, alpha_vec, beta_vec);
    // Downcast:
    sum_i16_low_vec = _mm512_cvtph_epi16(sum_f16_low_vec);
    sum_i16_high_vec = _mm512_cvtph_epi16(sum_f16_high_vec);
    sum_i8_vec = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(sum_i16_low_vec)),
                                    _mm512_cvtsepi16_epi8(sum_i16_high_vec), 1);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto simsimd_wsum_i8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_wsum_i8_sapphire(                       //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_i8_sapphire(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_i8_sapphire(a, n, alpha, 0, result); }
        else { simsimd_scale_i8_sapphire(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    __mmask64 mask = 0xFFFFFFFFFFFFFFFFull;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512i a_i8_vec, b_i8_vec, sum_i8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec, b_f16_low_vec, b_f16_high_vec;
    __m512h a_scaled_f16_low_vec, a_scaled_f16_high_vec, sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;
simsimd_wsum_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm512_maskz_loadu_epi8(mask, b);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        b_i8_vec = _mm512_loadu_epi8(b);
        a += 64, b += 64, n -= 64;
    }
    // Upcast:
    a_f16_low_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_vec)));
    a_f16_high_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_vec, 1)));
    b_f16_low_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_vec)));
    b_f16_high_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_vec, 1)));
    // Scale:
    a_scaled_f16_low_vec = _mm512_mul_ph(a_f16_low_vec, alpha_vec);
    a_scaled_f16_high_vec = _mm512_mul_ph(a_f16_high_vec, alpha_vec);
    // Add:
    sum_f16_low_vec = _mm512_fmadd_ph(b_f16_low_vec, beta_vec, a_scaled_f16_low_vec);
    sum_f16_high_vec = _mm512_fmadd_ph(b_f16_high_vec, beta_vec, a_scaled_f16_high_vec);
    // Downcast:
    sum_i16_low_vec = _mm512_cvtph_epi16(sum_f16_low_vec);
    sum_i16_high_vec = _mm512_cvtph_epi16(sum_f16_high_vec);
    sum_i8_vec = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(sum_i16_low_vec)),
                                    _mm512_cvtsepi16_epi8(sum_i16_high_vec), 1);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto simsimd_wsum_i8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_i8_sapphire(                                               //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {

    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512i a_i8_vec, b_i8_vec, c_i8_vec, sum_i8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec, b_f16_low_vec, b_f16_high_vec;
    __m512h c_f16_low_vec, c_f16_high_vec, ab_f16_low_vec, ab_f16_high_vec;
    __m512h ab_scaled_f16_low_vec, ab_scaled_f16_high_vec, sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;
simsimd_fma_i8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_i8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_i8_vec = _mm512_maskz_loadu_epi8(mask, b);
        c_i8_vec = _mm512_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_i8_vec = _mm512_loadu_epi8(a);
        b_i8_vec = _mm512_loadu_epi8(b);
        c_i8_vec = _mm512_loadu_epi8(c);
        a += 64, b += 64, c += 64, n -= 64;
    }
    // Upcast:
    a_f16_low_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(a_i8_vec)));
    a_f16_high_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_i8_vec, 1)));
    b_f16_low_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(b_i8_vec)));
    b_f16_high_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_i8_vec, 1)));
    c_f16_low_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(c_i8_vec)));
    c_f16_high_vec = _mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(c_i8_vec, 1)));
    // Multiply:
    ab_f16_low_vec = _mm512_mul_ph(a_f16_low_vec, b_f16_low_vec);
    ab_f16_high_vec = _mm512_mul_ph(a_f16_high_vec, b_f16_high_vec);
    // Scale:
    ab_scaled_f16_low_vec = _mm512_mul_ph(ab_f16_low_vec, alpha_vec);
    ab_scaled_f16_high_vec = _mm512_mul_ph(ab_f16_high_vec, alpha_vec);
    // Add:
    sum_f16_low_vec = _mm512_fmadd_ph(c_f16_low_vec, beta_vec, ab_scaled_f16_low_vec);
    sum_f16_high_vec = _mm512_fmadd_ph(c_f16_high_vec, beta_vec, ab_scaled_f16_high_vec);
    // Downcast:
    sum_i16_low_vec = _mm512_cvtph_epi16(sum_f16_low_vec);
    sum_i16_high_vec = _mm512_cvtph_epi16(sum_f16_high_vec);
    sum_i8_vec = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtsepi16_epi8(sum_i16_low_vec)),
                                    _mm512_cvtsepi16_epi8(sum_i16_high_vec), 1);
    _mm512_mask_storeu_epi8(result, mask, sum_i8_vec);
    result += 64;
    if (n) goto simsimd_fma_i8_sapphire_cycle;
}

SIMSIMD_PUBLIC void simsimd_fma_u8_sapphire(                                               //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {

    __mmask64 mask = 0xFFFFFFFFFFFFFFFF;
    __m512h alpha_vec = _mm512_set1_ph((_Float16)alpha);
    __m512h beta_vec = _mm512_set1_ph((_Float16)beta);
    __m512i a_u8_vec, b_u8_vec, c_u8_vec, sum_u8_vec;
    __m512h a_f16_low_vec, a_f16_high_vec, b_f16_low_vec, b_f16_high_vec;
    __m512h c_f16_low_vec, c_f16_high_vec, ab_f16_low_vec, ab_f16_high_vec;
    __m512h ab_scaled_f16_low_vec, ab_scaled_f16_high_vec, sum_f16_low_vec, sum_f16_high_vec;
    __m512i sum_i16_low_vec, sum_i16_high_vec;
simsimd_fma_u8_sapphire_cycle:
    if (n < 64) {
        mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFFull, n);
        a_u8_vec = _mm512_maskz_loadu_epi8(mask, a);
        b_u8_vec = _mm512_maskz_loadu_epi8(mask, b);
        c_u8_vec = _mm512_maskz_loadu_epi8(mask, c);
        n = 0;
    }
    else {
        a_u8_vec = _mm512_loadu_epi8(a);
        b_u8_vec = _mm512_loadu_epi8(b);
        c_u8_vec = _mm512_loadu_epi8(c);
        a += 64, b += 64, c += 64, n -= 64;
    }
    // Upcast:
    a_f16_low_vec = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(a_u8_vec, _mm512_setzero_si512()));
    a_f16_high_vec = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(a_u8_vec, _mm512_setzero_si512()));
    b_f16_low_vec = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(b_u8_vec, _mm512_setzero_si512()));
    b_f16_high_vec = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(b_u8_vec, _mm512_setzero_si512()));
    c_f16_low_vec = _mm512_cvtepi16_ph(_mm512_unpacklo_epi8(c_u8_vec, _mm512_setzero_si512()));
    c_f16_high_vec = _mm512_cvtepi16_ph(_mm512_unpackhi_epi8(c_u8_vec, _mm512_setzero_si512()));
    // Multiply:
    ab_f16_low_vec = _mm512_mul_ph(a_f16_low_vec, b_f16_low_vec);
    ab_f16_high_vec = _mm512_mul_ph(a_f16_high_vec, b_f16_high_vec);
    // Scale:
    ab_scaled_f16_low_vec = _mm512_mul_ph(ab_f16_low_vec, alpha_vec);
    ab_scaled_f16_high_vec = _mm512_mul_ph(ab_f16_high_vec, alpha_vec);
    // Add:
    sum_f16_low_vec = _mm512_fmadd_ph(c_f16_low_vec, beta_vec, ab_scaled_f16_low_vec);
    sum_f16_high_vec = _mm512_fmadd_ph(c_f16_high_vec, beta_vec, ab_scaled_f16_high_vec);
    // Downcast:
    sum_i16_low_vec = _mm512_cvtph_epi16(sum_f16_low_vec);
    sum_i16_high_vec = _mm512_cvtph_epi16(sum_f16_high_vec);
    sum_u8_vec = _mm512_packus_epi16(sum_i16_low_vec, sum_i16_high_vec);
    _mm512_mask_storeu_epi8(result, mask, sum_u8_vec);
    result += 64;
    if (n) goto simsimd_fma_u8_sapphire_cycle;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_SAPPHIRE
#endif // _SIMSIMD_TARGET_X86

#if _SIMSIMD_TARGET_ARM
#if SIMSIMD_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_sum_f32_neon(simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n,
                                         simsimd_f32_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = a[i] + b[i];
}

SIMSIMD_PUBLIC void simsimd_scale_f32_neon(simsimd_f32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_f32_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    float32x4_t alpha_vec = vdupq_n_f32(alpha_f32);
    float32x4_t beta_vec = vdupq_n_f32(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t sum_vec = vfmaq_n_f32(beta_vec, a_vec, alpha_f32);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_f32 * a[i] + beta_f32;
}

SIMSIMD_PUBLIC void simsimd_wsum_f32_neon(                            //
    simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f32_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f32_neon(a, n, alpha, 0, result); }
        else { simsimd_scale_f32_neon(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t a_scaled_vec = vmulq_n_f32(a_vec, alpha_f32);
        float32x4_t b_scaled_vec = vmulq_n_f32(b_vec, beta_f32);
        float32x4_t sum_vec = vaddq_f32(a_scaled_vec, b_scaled_vec);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_f32 * a[i] + beta_f32 * b[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f32_neon(                                   //
    simsimd_f32_t const *a, simsimd_f32_t const *b, simsimd_f32_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f32_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t c_vec = vld1q_f32(c + i);
        float32x4_t ab_vec = vmulq_f32(a_vec, b_vec);
        float32x4_t ab_scaled_vec = vmulq_n_f32(ab_vec, alpha_f32);
        float32x4_t sum_vec = vfmaq_n_f32(ab_scaled_vec, c_vec, beta_f32);
        vst1q_f32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) result[i] = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
}

SIMSIMD_PUBLIC void simsimd_sum_i16_neon(simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_size_t n,
                                         simsimd_i16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t a_vec = vld1q_s16(a + i);
        int16x8_t b_vec = vld1q_s16(b + i);
        int16x8_t sum_vec = vqaddq_s16(a_vec, b_vec);
        vst1q_s16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _simsimd_i16_sadd(a + i, b + i, result + i);
}

SIMSIMD_PUBLIC void simsimd_scale_i16_neon(simsimd_i16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_i16_t *result) {
    float32_t alpha_f32 = (float32_t)alpha;
    float32_t beta_f32 = (float32_t)beta;
    float32x4_t alpha_vec = vdupq_n_f32(alpha_f32);
    float32x4_t beta_vec = vdupq_n_f32(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16_vec = vld1_s16(a + i);
        float32x4_t a_vec = vcvtq_f32_s32(vmovl_s16(a_i16_vec));
        float32x4_t sum_vec = vfmaq_f32(beta_vec, a_vec, alpha_vec);
        int16x4_t sum_i16_vec = vqmovn_s32(vcvtaq_s32_f32(sum_vec));
        vst1_s16(result + i, sum_i16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f32 * a[i] + beta_f32;
        _simsimd_f32_to_i16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i16_neon(                                   //
    simsimd_i16_t const *a, simsimd_i16_t const *b, simsimd_i16_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i16_t *result) {
    float32_t alpha_f32 = (float32_t)alpha;
    float32_t beta_f32 = (float32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int16x4_t a_i16_vec = vld1_s16(a + i);
        int16x4_t b_i16_vec = vld1_s16(b + i);
        int16x4_t c_i16_vec = vld1_s16(c + i);
        float32x4_t a_vec = vcvtq_f32_s32(vmovl_s16(a_i16_vec));
        float32x4_t b_vec = vcvtq_f32_s32(vmovl_s16(b_i16_vec));
        float32x4_t c_vec = vcvtq_f32_s32(vmovl_s16(c_i16_vec));
        float32x4_t ab_vec = vmulq_f32(a_vec, b_vec);
        float32x4_t ab_scaled_vec = vmulq_n_f32(ab_vec, alpha_f32);
        float32x4_t sum_vec = vfmaq_n_f32(ab_scaled_vec, c_vec, beta_f32);
        int16x4_t sum_i16_vec = vqmovn_s32(vcvtaq_s32_f32(sum_vec));
        vst1_s16(result + i, sum_i16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        _simsimd_f32_to_i16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_u16_neon(simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_size_t n,
                                         simsimd_u16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint16x8_t a_vec = vld1q_u16(a + i);
        uint16x8_t b_vec = vld1q_u16(b + i);
        uint16x8_t sum_vec = vqaddq_u16(a_vec, b_vec);
        vst1q_u16(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _simsimd_u16_sadd(a + i, b + i, result + i);
}

SIMSIMD_PUBLIC void simsimd_scale_u16_neon(simsimd_u16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_u16_t *result) {
    float32_t alpha_f32 = (float32_t)alpha;
    float32_t beta_f32 = (float32_t)beta;
    float32x4_t alpha_vec = vdupq_n_f32(alpha_f32);
    float32x4_t beta_vec = vdupq_n_f32(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16_vec = vld1_u16(a + i);
        float32x4_t a_vec = vcvtq_f32_u32(vmovl_u16(a_u16_vec));
        float32x4_t sum_vec = vfmaq_f32(beta_vec, a_vec, alpha_vec);
        uint16x4_t sum_u16_vec = vqmovn_u32(vcvtaq_u32_f32(sum_vec));
        vst1_u16(result + i, sum_u16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f32 * a[i] + beta_f32;
        _simsimd_f32_to_u16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u16_neon(                                   //
    simsimd_u16_t const *a, simsimd_u16_t const *b, simsimd_u16_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u16_t *result) {
    float32_t alpha_f32 = (float32_t)alpha;
    float32_t beta_f32 = (float32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint16x4_t a_u16_vec = vld1_u16(a + i);
        uint16x4_t b_u16_vec = vld1_u16(b + i);
        uint16x4_t c_u16_vec = vld1_u16(c + i);
        float32x4_t a_vec = vcvtq_f32_u32(vmovl_u16(a_u16_vec));
        float32x4_t b_vec = vcvtq_f32_u32(vmovl_u16(b_u16_vec));
        float32x4_t c_vec = vcvtq_f32_u32(vmovl_u16(c_u16_vec));
        float32x4_t ab_vec = vmulq_f32(a_vec, b_vec);
        float32x4_t ab_scaled_vec = vmulq_n_f32(ab_vec, alpha_f32);
        float32x4_t sum_vec = vfmaq_n_f32(ab_scaled_vec, c_vec, beta_f32);
        uint16x4_t sum_u16_vec = vqmovn_u32(vcvtaq_u32_f32(sum_vec));
        vst1_u16(result + i, sum_u16_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f32 * a[i] * b[i] + beta_f32 * c[i];
        _simsimd_f32_to_u16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_i32_neon(simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_size_t n,
                                         simsimd_i32_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int32x4_t a_vec = vld1q_s32(a + i);
        int32x4_t b_vec = vld1q_s32(b + i);
        int32x4_t sum_vec = vqaddq_s32(a_vec, b_vec);
        vst1q_s32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _simsimd_i32_sadd(a + i, b + i, result + i);
}

SIMSIMD_PUBLIC void simsimd_scale_i32_neon(simsimd_i32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_i32_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;
    float64x2_t alpha_vec = vdupq_n_f64(alpha_f64);
    float64x2_t beta_vec = vdupq_n_f64(beta_f64);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32_vec = vld1_s32(a + i);
        float64x2_t a_vec = vcvtq_f64_s64(vmovl_s32(a_i32_vec));
        float64x2_t sum_vec = vfmaq_f64(beta_vec, a_vec, alpha_vec);
        int32x2_t sum_i32_vec = vqmovn_s64(vcvtaq_s64_f64(sum_vec));
        vst1_s32(result + i, sum_i32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] + beta_f64;
        _simsimd_f64_to_i32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i32_neon(                                   //
    simsimd_i32_t const *a, simsimd_i32_t const *b, simsimd_i32_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i32_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int32x2_t a_i32_vec = vld1_s32(a + i);
        int32x2_t b_i32_vec = vld1_s32(b + i);
        int32x2_t c_i32_vec = vld1_s32(c + i);
        float64x2_t a_vec = vcvtq_f64_s64(vmovl_s32(a_i32_vec));
        float64x2_t b_vec = vcvtq_f64_s64(vmovl_s32(b_i32_vec));
        float64x2_t c_vec = vcvtq_f64_s64(vmovl_s32(c_i32_vec));
        float64x2_t ab_vec = vmulq_f64(a_vec, b_vec);
        float64x2_t ab_scaled_vec = vmulq_n_f64(ab_vec, alpha_f64);
        float64x2_t sum_vec = vfmaq_n_f64(ab_scaled_vec, c_vec, beta_f64);
        int32x2_t sum_i32_vec = vqmovn_s64(vcvtaq_s64_f64(sum_vec));
        vst1_s32(result + i, sum_i32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] * b[i] + beta_f64 * c[i];
        _simsimd_f64_to_i32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_u32_neon(simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_size_t n,
                                         simsimd_u32_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t a_vec = vld1q_u32(a + i);
        uint32x4_t b_vec = vld1q_u32(b + i);
        uint32x4_t sum_vec = vqaddq_u32(a_vec, b_vec);
        vst1q_u32(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _simsimd_u32_sadd(a + i, b + i, result + i);
}

SIMSIMD_PUBLIC void simsimd_scale_u32_neon(simsimd_u32_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_u32_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;
    float64x2_t alpha_vec = vdupq_n_f64(alpha_f64);
    float64x2_t beta_vec = vdupq_n_f64(beta_f64);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32_vec = vld1_u32(a + i);
        float64x2_t a_vec = vcvtq_f64_u64(vmovl_u32(a_u32_vec));
        float64x2_t sum_vec = vfmaq_f64(beta_vec, a_vec, alpha_vec);
        uint32x2_t sum_u32_vec = vqmovn_u64(vcvtaq_u64_f64(sum_vec));
        vst1_u32(result + i, sum_u32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] + beta_f64;
        _simsimd_f64_to_u32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u32_neon(                                   //
    simsimd_u32_t const *a, simsimd_u32_t const *b, simsimd_u32_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u32_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint32x2_t a_u32_vec = vld1_u32(a + i);
        uint32x2_t b_u32_vec = vld1_u32(b + i);
        uint32x2_t c_u32_vec = vld1_u32(c + i);
        float64x2_t a_vec = vcvtq_f64_u64(vmovl_u32(a_u32_vec));
        float64x2_t b_vec = vcvtq_f64_u64(vmovl_u32(b_u32_vec));
        float64x2_t c_vec = vcvtq_f64_u64(vmovl_u32(c_u32_vec));
        float64x2_t ab_vec = vmulq_f64(a_vec, b_vec);
        float64x2_t ab_scaled_vec = vmulq_n_f64(ab_vec, alpha_f64);
        float64x2_t sum_vec = vfmaq_n_f64(ab_scaled_vec, c_vec, beta_f64);
        uint32x2_t sum_u32_vec = vqmovn_u64(vcvtaq_u64_f64(sum_vec));
        vst1_u32(result + i, sum_u32_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] * b[i] + beta_f64 * c[i];
        _simsimd_f64_to_u32(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_i64_neon(simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_size_t n,
                                         simsimd_i64_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_vec = vld1q_s64(a + i);
        int64x2_t b_vec = vld1q_s64(b + i);
        int64x2_t sum_vec = vqaddq_s64(a_vec, b_vec);
        vst1q_s64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _simsimd_i64_sadd(a + i, b + i, result + i);
}

SIMSIMD_PUBLIC void simsimd_scale_i64_neon(simsimd_i64_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_i64_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;
    float64x2_t alpha_vec = vdupq_n_f64(alpha_f64);
    float64x2_t beta_vec = vdupq_n_f64(beta_f64);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64_vec = vld1q_s64(a + i);
        float64x2_t a_vec = vcvtq_f64_s64(a_i64_vec);
        float64x2_t sum_vec = vfmaq_f64(beta_vec, a_vec, alpha_vec);
        int64x2_t sum_i64_vec = vcvtaq_s64_f64(sum_vec);
        vst1q_s64(result + i, sum_i64_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] + beta_f64;
        _simsimd_f64_to_i64(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i64_neon(                                   //
    simsimd_i64_t const *a, simsimd_i64_t const *b, simsimd_i64_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i64_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        int64x2_t a_i64_vec = vld1q_s64(a + i);
        int64x2_t b_i64_vec = vld1q_s64(b + i);
        int64x2_t c_i64_vec = vld1q_s64(c + i);
        float64x2_t a_vec = vcvtq_f64_s64(a_i64_vec);
        float64x2_t b_vec = vcvtq_f64_s64(b_i64_vec);
        float64x2_t c_vec = vcvtq_f64_s64(c_i64_vec);
        float64x2_t ab_vec = vmulq_f64(a_vec, b_vec);
        float64x2_t ab_scaled_vec = vmulq_n_f64(ab_vec, alpha_f64);
        float64x2_t sum_vec = vfmaq_n_f64(ab_scaled_vec, c_vec, beta_f64);
        int64x2_t sum_i64_vec = vcvtaq_s64_f64(sum_vec);
        vst1q_s64(result + i, sum_i64_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] * b[i] + beta_f64 * c[i];
        _simsimd_f64_to_i64(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_u64_neon(simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_size_t n,
                                         simsimd_u64_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_vec = vld1q_u64(a + i);
        uint64x2_t b_vec = vld1q_u64(b + i);
        uint64x2_t sum_vec = vqaddq_u64(a_vec, b_vec);
        vst1q_u64(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) _simsimd_u64_sadd(a + i, b + i, result + i);
}

SIMSIMD_PUBLIC void simsimd_scale_u64_neon(simsimd_u64_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_u64_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;
    float64x2_t alpha_vec = vdupq_n_f64(alpha_f64);
    float64x2_t beta_vec = vdupq_n_f64(beta_f64);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64_vec = vld1q_u64(a + i);
        float64x2_t a_vec = vcvtq_f64_u64(a_u64_vec);
        float64x2_t sum_vec = vfmaq_f64(beta_vec, a_vec, alpha_vec);
        uint64x2_t sum_u64_vec = vcvtaq_u64_f64(sum_vec);
        vst1q_u64(result + i, sum_u64_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] + beta_f64;
        _simsimd_f64_to_u64(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u64_neon(                                   //
    simsimd_u64_t const *a, simsimd_u64_t const *b, simsimd_u64_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u64_t *result) {
    float64_t alpha_f64 = (float64_t)alpha;
    float64_t beta_f64 = (float64_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        uint64x2_t a_u64_vec = vld1q_u64(a + i);
        uint64x2_t b_u64_vec = vld1q_u64(b + i);
        uint64x2_t c_u64_vec = vld1q_u64(c + i);
        float64x2_t a_vec = vcvtq_f64_u64(a_u64_vec);
        float64x2_t b_vec = vcvtq_f64_u64(b_u64_vec);
        float64x2_t c_vec = vcvtq_f64_u64(c_u64_vec);
        float64x2_t ab_vec = vmulq_f64(a_vec, b_vec);
        float64x2_t ab_scaled_vec = vmulq_n_f64(ab_vec, alpha_f64);
        float64x2_t sum_vec = vfmaq_n_f64(ab_scaled_vec, c_vec, beta_f64);
        uint64x2_t sum_u64_vec = vcvtaq_u64_f64(sum_vec);
        vst1q_u64(result + i, sum_u64_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f64_t sum = alpha_f64 * a[i] * b[i] + beta_f64 * c[i];
        _simsimd_f64_to_u64(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON

#if SIMSIMD_TARGET_NEON_BF16
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_sum_bf16_neon(simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n,
                                          simsimd_bf16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t sum_vec = vaddq_f32(a_vec, b_vec);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_bf16_to_f32(b + i, &bi);
        simsimd_f32_t sum = ai + bi;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_bf16_neon(simsimd_bf16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                            simsimd_distance_t beta, simsimd_bf16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;
    float32x4_t alpha_vec = vdupq_n_f32(alpha_f32);
    float32x4_t beta_vec = vdupq_n_f32(beta_f32);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t sum_vec = vfmaq_f32(beta_vec, a_vec, alpha_vec);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_bf16_neon(                             //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_bf16_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_bf16_neon(a, n, alpha, 0, result); }
        else { simsimd_scale_bf16_neon(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t a_scaled_vec = vmulq_n_f32(a_vec, alpha_f32);
        float32x4_t b_scaled_vec = vmulq_n_f32(b_vec, beta_f32);
        float32x4_t sum_vec = vaddq_f32(a_scaled_vec, b_scaled_vec);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_bf16_to_f32(b + i, &bi);
        simsimd_f32_t sum = alpha_f32 * ai + beta_f32 * bi;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_bf16_neon(                                     //
    simsimd_bf16_t const *a, simsimd_bf16_t const *b, simsimd_bf16_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_bf16_t *result) {
    simsimd_f32_t alpha_f32 = (simsimd_f32_t)alpha;
    simsimd_f32_t beta_f32 = (simsimd_f32_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t a_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)a + i));
        float32x4_t b_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)b + i));
        float32x4_t c_vec = vcvt_f32_bf16(vld1_bf16((bfloat16_t const *)c + i));
        float32x4_t ab_vec = vmulq_f32(a_vec, b_vec);
        float32x4_t ab_scaled_vec = vmulq_n_f32(ab_vec, alpha_f32);
        float32x4_t sum_vec = vfmaq_n_f32(ab_scaled_vec, c_vec, beta_f32);
        vst1_bf16((bfloat16_t *)result + i, vcvt_bf16_f32(sum_vec));
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t ai, bi, ci;
        simsimd_bf16_to_f32(a + i, &ai);
        simsimd_bf16_to_f32(b + i, &bi);
        simsimd_bf16_to_f32(c + i, &ci);
        simsimd_f32_t sum = alpha * ai * bi + beta * ci;
        simsimd_f32_to_bf16(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_BF16

#if SIMSIMD_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

SIMSIMD_PUBLIC void simsimd_sum_f16_neon(simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n,
                                         simsimd_f16_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_vec = vld1q_f16((float16_t const *)b + i);
        float16x8_t sum_vec = vaddq_f16(a_vec, b_vec);
        vst1q_f16((float16_t *)result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) ((float16_t *)result)[i] = ((float16_t const *)a)[i] + ((float16_t const *)b)[i];
}

SIMSIMD_PUBLIC void simsimd_scale_f16_neon(simsimd_f16_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                           simsimd_distance_t beta, simsimd_f16_t *result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;
    float16x8_t alpha_vec = vdupq_n_f16(alpha_f16);
    float16x8_t beta_vec = vdupq_n_f16(beta_f16);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16((float16_t const *)a + i);
        float16x8_t sum_vec = vfmaq_f16(beta_vec, a_vec, alpha_vec);
        vst1q_f16((float16_t *)result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] + beta_f16;
}

SIMSIMD_PUBLIC void simsimd_wsum_f16_neon(                            //
    simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result) {
    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_f16_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_f16_neon(a, n, alpha, 0, result); }
        else { simsimd_scale_f16_neon(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_vec = vld1q_f16((float16_t const *)b + i);
        float16x8_t a_scaled_vec = vmulq_n_f16(a_vec, alpha_f16);
        float16x8_t b_scaled_vec = vmulq_n_f16(b_vec, beta_f16);
        float16x8_t sum_vec = vaddq_f16(a_scaled_vec, b_scaled_vec);
        vst1q_f16((float16_t *)result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t *)result)[i] = alpha_f16 * ((float16_t const *)a)[i] + beta_f16 * ((float16_t const *)b)[i];
}

SIMSIMD_PUBLIC void simsimd_fma_f16_neon(                                   //
    simsimd_f16_t const *a, simsimd_f16_t const *b, simsimd_f16_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_f16_t *result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t a_vec = vld1q_f16((float16_t const *)a + i);
        float16x8_t b_vec = vld1q_f16((float16_t const *)b + i);
        float16x8_t c_vec = vld1q_f16((float16_t const *)c + i);
        float16x8_t ab_vec = vmulq_f16(a_vec, b_vec);
        float16x8_t ab_scaled_vec = vmulq_n_f16(ab_vec, alpha_f16);
        float16x8_t sum_vec = vfmaq_n_f16(ab_scaled_vec, c_vec, beta_f16);
        vst1q_f16((float16_t *)result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i)
        ((float16_t *)result)[i] =
            alpha_f16 * ((float16_t const *)a)[i] * ((float16_t const *)b)[i] + beta_f16 * ((float16_t const *)c)[i];
}

SIMSIMD_PUBLIC void simsimd_sum_u8_neon(simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n,
                                        simsimd_u8_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);
        uint8x16_t sum_vec = vqaddq_u8(a_vec, b_vec);
        vst1q_u8(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = (simsimd_f32_t)a[i] + b[i];
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_u8_neon(simsimd_u8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                          simsimd_distance_t beta, simsimd_u8_t *result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;
    float16x8_t alpha_vec = vdupq_n_f16(alpha_f16);
    float16x8_t beta_vec = vdupq_n_f16(beta_f16);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8_vec = vld1_u8(a + i);
        float16x8_t a_vec = vcvtq_f16_u16(vmovl_u8(a_u8_vec));
        float16x8_t sum_vec = vfmaq_f16(beta_vec, a_vec, alpha_vec);
        uint8x8_t sum_u8_vec = vqmovn_u16(vcvtaq_u16_f16(sum_vec));
        vst1_u8(result + i, sum_u8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f16 * a[i] + beta_f16;
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_u8_neon(                           //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_u8_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_u8_neon(a, n, alpha, 0, result); }
        else { simsimd_scale_u8_neon(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8_vec = vld1_u8(a + i);
        uint8x8_t b_u8_vec = vld1_u8(b + i);
        float16x8_t a_vec = vcvtq_f16_u16(vmovl_u8(a_u8_vec));
        float16x8_t b_vec = vcvtq_f16_u16(vmovl_u8(b_u8_vec));
        float16x8_t a_scaled_vec = vmulq_n_f16(a_vec, alpha_f16);
        float16x8_t b_scaled_vec = vmulq_n_f16(b_vec, beta_f16);
        float16x8_t sum_vec = vaddq_f16(a_scaled_vec, b_scaled_vec);
        uint8x8_t sum_u8_vec = vqmovn_u16(vcvtaq_u16_f16(sum_vec));
        vst1_u8(result + i, sum_u8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f16 * a[i] + beta_f16 * b[i];
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_u8_neon(                                 //
    simsimd_u8_t const *a, simsimd_u8_t const *b, simsimd_u8_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_u8_t *result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint8x8_t a_u8_vec = vld1_u8(a + i);
        uint8x8_t b_u8_vec = vld1_u8(b + i);
        uint8x8_t c_u8_vec = vld1_u8(c + i);
        float16x8_t a_vec = vcvtq_f16_u16(vmovl_u8(a_u8_vec));
        float16x8_t b_vec = vcvtq_f16_u16(vmovl_u8(b_u8_vec));
        float16x8_t c_vec = vcvtq_f16_u16(vmovl_u8(c_u8_vec));
        float16x8_t ab_vec = vmulq_f16(a_vec, b_vec);
        float16x8_t ab_scaled_vec = vmulq_n_f16(ab_vec, alpha_f16);
        float16x8_t sum_vec = vfmaq_n_f16(ab_scaled_vec, c_vec, beta_f16);
        uint8x8_t sum_u8_vec = vqmovn_u16(vcvtaq_u16_f16(sum_vec));
        vst1_u8(result + i, sum_u8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f16 * a[i] * b[i] + beta_f16 * c[i];
        _simsimd_f32_to_u8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_sum_i8_neon(simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n,
                                        simsimd_i8_t *result) {
    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);
        int8x16_t sum_vec = vqaddq_s8(a_vec, b_vec);
        vst1q_s8(result + i, sum_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = (simsimd_f32_t)a[i] + b[i];
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_scale_i8_neon(simsimd_i8_t const *a, simsimd_size_t n, simsimd_distance_t alpha,
                                          simsimd_distance_t beta, simsimd_i8_t *result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;
    float16x8_t alpha_vec = vdupq_n_f16(alpha_f16);
    float16x8_t beta_vec = vdupq_n_f16(beta_f16);

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8_vec = vld1_s8(a + i);
        float16x8_t a_vec = vcvtq_f16_s16(vmovl_s8(a_i8_vec));
        float16x8_t sum_vec = vfmaq_f16(beta_vec, a_vec, alpha_vec);
        int8x8_t sum_i8_vec = vqmovn_s16(vcvtaq_s16_f16(sum_vec));
        vst1_s8(result + i, sum_i8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f16 * a[i] + beta_f16;
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_wsum_i8_neon(                           //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_size_t n, //
    simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {

    // There are are several special cases we may want to implement:
    // 1. Simple addition, when both weights are equal to 1.0.
    if (alpha == 1 && beta == 1) {
        // In this case we can avoid expensive multiplications.
        simsimd_sum_i8_neon(a, b, n, result);
        return;
    }
    // 2. Just scaling, when one of the weights is equal to zero.
    else if (alpha == 0 || beta == 0) {
        // In this case we can avoid half of the load instructions.
        if (beta == 0) { simsimd_scale_i8_neon(a, n, alpha, 0, result); }
        else { simsimd_scale_i8_neon(b, n, beta, 0, result); }
        return;
    }

    // The general case.
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8_vec = vld1_s8(a + i);
        int8x8_t b_i8_vec = vld1_s8(b + i);
        float16x8_t a_vec = vcvtq_f16_s16(vmovl_s8(a_i8_vec));
        float16x8_t b_vec = vcvtq_f16_s16(vmovl_s8(b_i8_vec));
        float16x8_t a_scaled_vec = vmulq_n_f16(a_vec, alpha_f16);
        float16x8_t b_scaled_vec = vmulq_n_f16(b_vec, beta_f16);
        float16x8_t sum_vec = vaddq_f16(a_scaled_vec, b_scaled_vec);
        int8x8_t sum_i8_vec = vqmovn_s16(vcvtaq_s16_f16(sum_vec));
        vst1_s8(result + i, sum_i8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f16 * a[i] + beta_f16 * b[i];
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

SIMSIMD_PUBLIC void simsimd_fma_i8_neon(                                 //
    simsimd_i8_t const *a, simsimd_i8_t const *b, simsimd_i8_t const *c, //
    simsimd_size_t n, simsimd_distance_t alpha, simsimd_distance_t beta, simsimd_i8_t *result) {
    float16_t alpha_f16 = (float16_t)alpha;
    float16_t beta_f16 = (float16_t)beta;

    // The main loop:
    simsimd_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t a_i8_vec = vld1_s8(a + i);
        int8x8_t b_i8_vec = vld1_s8(b + i);
        int8x8_t c_i8_vec = vld1_s8(c + i);
        float16x8_t a_vec = vcvtq_f16_s16(vmovl_s8(a_i8_vec));
        float16x8_t b_vec = vcvtq_f16_s16(vmovl_s8(b_i8_vec));
        float16x8_t c_vec = vcvtq_f16_s16(vmovl_s8(c_i8_vec));
        float16x8_t ab_vec = vmulq_f16(a_vec, b_vec);
        float16x8_t ab_scaled_vec = vmulq_n_f16(ab_vec, alpha_f16);
        float16x8_t sum_vec = vfmaq_n_f16(ab_scaled_vec, c_vec, beta_f16);
        int8x8_t sum_i8_vec = vqmovn_s16(vcvtaq_s16_f16(sum_vec));
        vst1_s8(result + i, sum_i8_vec);
    }

    // The tail:
    for (; i < n; ++i) {
        simsimd_f32_t sum = alpha_f16 * a[i] * b[i] + beta_f16 * c[i];
        _simsimd_f32_to_i8(&sum, result + i);
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SIMSIMD_TARGET_NEON_F16
#endif // _SIMSIMD_TARGET_ARM

#ifdef __cplusplus
}
#endif

#endif
