/**
 *  @brief SIMD-accelerated Batched Dot Products for Haswell.
 *  @file include/numkong/dots/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/dots.h
 *
 *  @section haswell_dots_instructions Key AVX2/FMA GEMM Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_fmadd_ps/pd          VFMADD (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_mul_ps               VMULPS (YMM, YMM, YMM)          5cy         0.5/cy      p01
 *      _mm256_add_ps               VADDPS (YMM, YMM, YMM)          3cy         1/cy        p01
 *      _mm256_cvtph_ps             VCVTPH2PS (YMM, XMM)            5cy         1/cy        p01
 *      _mm256_madd_epi16           VPMADDWD (YMM, YMM, YMM)        5cy         1/cy        p0
 *
 *  GEMM kernels use tiled dot products with 4-way parallel accumulation to hide FMA latency.
 *  Type-specific tile sizes: f32/f64 use depth_simd_dimensions=4, f16/bf16 use depth_simd_dimensions=8,
 *  i8/u8/fp8 use depth_simd_dimensions=16. Integer dot products use VPMADDWD for efficient i16 pair
 *  multiplication with i32 accumulation.
 */
#ifndef NK_DOTS_HASWELL_H
#define NK_DOTS_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/cast.h" // `nk_f16_to_f32`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/* F32 GEMM: depth_simd_dimensions=4 (4 f32s = 16 bytes for f32->f64 upcast accumulation) */
nk_define_cross_pack_size_(dots, f32, haswell, f32, f32, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f32, haswell, f32, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f32, haswell, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_haswell_t, nk_b128_vec_t,
                           nk_dot_f32x4_init_haswell, nk_load_b128_haswell_, nk_partial_load_b32x4_serial_,
                           nk_dot_f32x4_update_haswell, nk_dot_f32x4_finalize_haswell, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f32, haswell, f32, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_haswell_t, nk_b128_vec_t,
                        nk_dot_f32x4_init_haswell, nk_load_b128_haswell_, nk_partial_load_b32x4_serial_,
                        nk_load_b128_haswell_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_haswell,
                        nk_dot_f32x4_finalize_haswell, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F64 GEMM: depth_simd_dimensions=4 (4 f64s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, f64, haswell, f64, f64, /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f64, haswell, f64, f64, nk_assign_from_to_, /*depth_simd_dimensions=*/4,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f64, haswell, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_haswell_t, nk_b256_vec_t,
                           nk_dot_f64x4_init_haswell, nk_load_b256_haswell_, nk_partial_load_b64x4_serial_,
                           nk_dot_f64x4_update_haswell, nk_dot_f64x4_finalize_haswell, nk_partial_store_b64x4_serial_,
                           /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f64, haswell, f64, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_haswell_t, nk_b256_vec_t,
                        nk_dot_f64x4_init_haswell, nk_load_b256_haswell_, nk_partial_load_b64x4_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b64x4_serial_, nk_dot_f64x4_update_haswell,
                        nk_dot_f64x4_finalize_haswell, nk_partial_store_b64x4_serial_,
                        /*depth_simd_dimensions=*/4, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=8 (8 f16s = 16 bytes = 128-bit input) → upcasted to 8×f32 (256-bit) */
nk_define_cross_pack_size_(dots, f16, haswell, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, haswell, f16, f32, nk_f16_to_f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1) // Store as F32
nk_define_cross_symmetric_(dots, f16, haswell, f16, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_f16x8_to_f32x8_haswell_,
                           nk_partial_load_f16x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                           nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, haswell, f16, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_f16x8_to_f32x8_haswell_,
                        nk_partial_load_f16x8_to_f32x8_haswell_, nk_load_b256_haswell_, nk_partial_load_b32x8_serial_,
                        nk_dot_through_f32_update_haswell_, nk_dot_through_f32_finalize_haswell_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=8 (8 bf16s = 16 bytes = 128-bit input) → upcasted to 8×f32 (256-bit) */
nk_define_cross_pack_size_(dots, bf16, haswell, bf16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, bf16, haswell, bf16, f32, nk_bf16_to_f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1) // Store as F32
nk_define_cross_symmetric_(dots, bf16, haswell, bf16, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_bf16x8_to_f32x8_haswell_,
                           nk_partial_load_bf16x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                           nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, bf16, haswell, bf16, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_bf16x8_to_f32x8_haswell_,
                        nk_partial_load_bf16x8_to_f32x8_haswell_, nk_load_b256_haswell_, nk_partial_load_b32x8_serial_,
                        nk_dot_through_f32_update_haswell_, nk_dot_through_f32_finalize_haswell_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=8 (8 e4m3s = 8 bytes) → upcasted to 8×f32 (256-bit) */
nk_define_cross_pack_size_(dots, e4m3, haswell, e4m3, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e4m3, haswell, e4m3, f32, nk_e4m3_to_f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e4m3, haswell, e4m3, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e4m3x8_to_f32x8_haswell_,
                           nk_partial_load_e4m3x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                           nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e4m3, haswell, e4m3, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e4m3x8_to_f32x8_haswell_,
                        nk_partial_load_e4m3x8_to_f32x8_haswell_, nk_load_b256_haswell_, nk_partial_load_b32x8_serial_,
                        nk_dot_through_f32_update_haswell_, nk_dot_through_f32_finalize_haswell_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=8 (8 e5m2s = 8 bytes) → upcasted to 8×f32 (256-bit) */
nk_define_cross_pack_size_(dots, e5m2, haswell, e5m2, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e5m2, haswell, e5m2, f32, nk_e5m2_to_f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e5m2, haswell, e5m2, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e5m2x8_to_f32x8_haswell_,
                           nk_partial_load_e5m2x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                           nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e5m2, haswell, e5m2, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e5m2x8_to_f32x8_haswell_,
                        nk_partial_load_e5m2x8_to_f32x8_haswell_, nk_load_b256_haswell_, nk_partial_load_b32x8_serial_,
                        nk_dot_through_f32_update_haswell_, nk_dot_through_f32_finalize_haswell_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E2M3 GEMM: depth_simd_dimensions=8 (8 e2m3s = 8 bytes) → upcasted to 8×f32 (256-bit) */
nk_define_cross_pack_size_(dots, e2m3, haswell, e2m3, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, haswell, e2m3, f32, nk_e2m3_to_f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, haswell, e2m3, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e2m3x8_to_f32x8_haswell_,
                           nk_partial_load_e2m3x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                           nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, haswell, e2m3, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e2m3x8_to_f32x8_haswell_,
                        nk_partial_load_e2m3x8_to_f32x8_haswell_, nk_load_b256_haswell_, nk_partial_load_b32x8_serial_,
                        nk_dot_through_f32_update_haswell_, nk_dot_through_f32_finalize_haswell_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E3M2 GEMM: depth_simd_dimensions=8 (8 e3m2s = 8 bytes) → upcasted to 8×f32 (256-bit) */
nk_define_cross_pack_size_(dots, e3m2, haswell, e3m2, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e3m2, haswell, e3m2, f32, nk_e3m2_to_f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, haswell, e3m2, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                           nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e3m2x8_to_f32x8_haswell_,
                           nk_partial_load_e3m2x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                           nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, haswell, e3m2, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_,
                        nk_b128_vec_t, nk_dot_through_f32_init_haswell_, nk_load_e3m2x8_to_f32x8_haswell_,
                        nk_partial_load_e3m2x8_to_f32x8_haswell_, nk_load_b256_haswell_, nk_partial_load_b32x8_serial_,
                        nk_dot_through_f32_update_haswell_, nk_dot_through_f32_finalize_haswell_,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* I8 GEMM: depth_simd_dimensions=16 (16 i8s = 16 bytes = 128-bit input) */
nk_define_cross_pack_size_(dots, i8, haswell, i8, i8, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, haswell, i8, i8, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, haswell, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_haswell_t, nk_b128_vec_t,
                           nk_dot_i8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                           nk_dot_i8x16_update_haswell, nk_dot_i8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, haswell, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_haswell_t, nk_b128_vec_t,
                        nk_dot_i8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_haswell_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_haswell,
                        nk_dot_i8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=16 (16 u8s = 16 bytes = 128-bit input) */
nk_define_cross_pack_size_(dots, u8, haswell, u8, u8, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, haswell, u8, u8, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, haswell, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_haswell_t, nk_b128_vec_t,
                           nk_dot_u8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                           nk_dot_u8x16_update_haswell, nk_dot_u8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, haswell, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_haswell_t, nk_b128_vec_t,
                        nk_dot_u8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_haswell_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_haswell,
                        nk_dot_u8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* I4 GEMM: depth_simd_dimensions=32 (32 nibbles = 16 bytes = 128-bit input)
 * Note: dimensions_per_value=2 because 2 nibbles (i4 values) are packed per byte */
nk_define_cross_pack_size_(dots, i4, haswell, i4x2, i4x2, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, i4, haswell, i4x2, i4x2, nk_assign_from_to_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, i4, haswell, i4x2, i32, nk_b128_vec_t, nk_dot_i4x32_state_haswell_t, nk_b128_vec_t,
                           nk_dot_i4x32_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                           nk_dot_i4x32_update_haswell, nk_dot_i4x32_finalize_haswell, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, i4, haswell, i4x2, i4x2, i32, nk_b128_vec_t, nk_dot_i4x32_state_haswell_t, nk_b128_vec_t,
                        nk_dot_i4x32_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_haswell_, nk_partial_load_b8x16_serial_, nk_dot_i4x32_update_haswell,
                        nk_dot_i4x32_finalize_haswell, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

/* U4 GEMM: depth_simd_dimensions=32 (32 nibbles = 16 bytes = 128-bit input)
 * Note: dimensions_per_value=2 because 2 nibbles (u4 values) are packed per byte */
nk_define_cross_pack_size_(dots, u4, haswell, u4x2, u4x2, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_pack_(dots, u4, haswell, u4x2, u4x2, nk_assign_from_to_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2)
nk_define_cross_symmetric_(dots, u4, haswell, u4x2, u32, nk_b128_vec_t, nk_dot_u4x32_state_haswell_t, nk_b128_vec_t,
                           nk_dot_u4x32_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                           nk_dot_u4x32_update_haswell, nk_dot_u4x32_finalize_haswell, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_packed_(dots, u4, haswell, u4x2, u4x2, u32, nk_b128_vec_t, nk_dot_u4x32_state_haswell_t, nk_b128_vec_t,
                        nk_dot_u4x32_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_haswell_, nk_partial_load_b8x16_serial_, nk_dot_u4x32_update_haswell,
                        nk_dot_u4x32_finalize_haswell, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_DOTS_HASWELL_H
