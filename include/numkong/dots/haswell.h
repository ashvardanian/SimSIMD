/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/dots/haswell.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
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
 *  Type-specific tile sizes: f32/f64 use simd_width=4, f16/bf16 use simd_width=8, i8/u8/fp8 use simd_width=16.
 *  Integer dot products use VPMADDWD for efficient i16 pair multiplication with i32 accumulation.
 */
#ifndef NK_DOTS_HASWELL_H
#define NK_DOTS_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* F32 GEMM: simd_width=4 (4 f32s = 16 bytes for f32->f64 upcast accumulation) */
nk_define_dots_pack_size_(haswell, f32, f32, f32)
nk_define_dots_pack_(haswell, f32, f32, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(f32_haswell, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_haswell_t, nk_b128_vec_t,
                          nk_dot_f32x4_init_haswell, nk_load_b128_haswell_, nk_partial_load_b32x4_serial_,
                          nk_dot_f32x4_update_haswell, nk_dot_f32x4_finalize_haswell,
                          /*simd_width=*/4)
nk_define_dots_packed_(f32_haswell, f32, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_haswell_t, nk_b128_vec_t,
                       nk_dot_f32x4_init_haswell, nk_load_b128_haswell_, nk_partial_load_b32x4_serial_,
                       nk_load_b128_haswell_, nk_partial_load_b32x4_serial_, nk_dot_f32x4_update_haswell,
                       nk_dot_f32x4_finalize_haswell, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/4)

/* F64 GEMM: simd_width=4 (4 f64s = 32 bytes = AVX2 register width) */
nk_define_dots_pack_size_(haswell, f64, f64, f64)
nk_define_dots_pack_(haswell, f64, f64, f64, nk_assign_from_to_)
nk_define_dots_symmetric_(f64_haswell, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_haswell_t, nk_b256_vec_t,
                          nk_dot_f64x4_init_haswell, nk_load_b256_haswell_, nk_partial_load_b64x4_serial_,
                          nk_dot_f64x4_update_haswell, nk_dot_f64x4_finalize_haswell,
                          /*simd_width=*/4)
nk_define_dots_packed_(f64_haswell, f64, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_haswell_t, nk_b256_vec_t,
                       nk_dot_f64x4_init_haswell, nk_load_b256_haswell_, nk_partial_load_b64x4_serial_,
                       nk_load_b256_haswell_, nk_partial_load_b64x4_serial_, nk_dot_f64x4_update_haswell,
                       nk_dot_f64x4_finalize_haswell, nk_partial_store_b64x4_serial_,
                       /*simd_width=*/4)

/* F16 GEMM: simd_width=8 (8 f16s = 16 bytes = 128-bit input) → upcasted to 8×f32 (256-bit) */
nk_define_dots_pack_size_(haswell, f16, f32, f32)
nk_define_dots_pack_(haswell, f16, f32, f32, nk_f16_to_f32) // Store as F32
nk_define_dots_symmetric_(f16_haswell, f16, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_haswell_, nk_load_f16x8_to_f32x8_haswell_,
                          nk_dots_partial_load_f16x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                          nk_dot_through_f32_finalize_haswell_,
                          /*simd_width=*/8)
nk_define_dots_packed_(f16_haswell, f16, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_haswell_, nk_load_f16x8_to_f32x8_haswell_,
                       nk_dots_partial_load_f16x8_to_f32x8_haswell_, nk_load_b256_haswell_,
                       nk_partial_load_b32x8_serial_, nk_dot_through_f32_update_haswell_,
                       nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* BF16 GEMM: simd_width=8 (8 bf16s = 16 bytes = 128-bit input) → upcasted to 8×f32 (256-bit) */
nk_define_dots_pack_size_(haswell, bf16, f32, f32)
nk_define_dots_pack_(haswell, bf16, f32, f32, nk_bf16_to_f32) // Store as F32
nk_define_dots_symmetric_(bf16_haswell, bf16, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_haswell_, nk_load_bf16x8_to_f32x8_haswell_,
                          nk_dots_partial_load_bf16x8_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                          nk_dot_through_f32_finalize_haswell_,
                          /*simd_width=*/8)
nk_define_dots_packed_(bf16_haswell, bf16, f32, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_haswell_, nk_load_bf16x8_to_f32x8_haswell_,
                       nk_dots_partial_load_bf16x8_to_f32x8_haswell_, nk_load_b256_haswell_,
                       nk_partial_load_b32x8_serial_, nk_dot_through_f32_update_haswell_,
                       nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* E4M3 GEMM: simd_width=8 (8 e4m3s = 8 bytes) → upcasted to 8×f32 (256-bit) */
nk_define_dots_pack_size_(haswell, e4m3, e4m3, f32)
nk_define_dots_pack_(haswell, e4m3, e4m3, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(e4m3_haswell, e4m3, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_haswell_, nk_load_e4m3x16_to_f32x8_haswell_,
                          nk_dots_partial_load_e4m3x16_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                          nk_dot_through_f32_finalize_haswell_,
                          /*simd_width=*/8)
nk_define_dots_packed_(e4m3_haswell, e4m3, e4m3, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_haswell_, nk_load_e4m3x16_to_f32x8_haswell_,
                       nk_dots_partial_load_e4m3x16_to_f32x8_haswell_, nk_load_b256_haswell_,
                       nk_partial_load_b32x8_serial_, nk_dot_through_f32_update_haswell_,
                       nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* E5M2 GEMM: simd_width=8 (8 e5m2s = 8 bytes) → upcasted to 8×f32 (256-bit) */
nk_define_dots_pack_size_(haswell, e5m2, e5m2, f32)
nk_define_dots_pack_(haswell, e5m2, e5m2, f32, nk_assign_from_to_)
nk_define_dots_symmetric_(e5m2_haswell, e5m2, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_haswell_, nk_load_e5m2x16_to_f32x8_haswell_,
                          nk_dots_partial_load_e5m2x16_to_f32x8_haswell_, nk_dot_through_f32_update_haswell_,
                          nk_dot_through_f32_finalize_haswell_,
                          /*simd_width=*/8)
nk_define_dots_packed_(e5m2_haswell, e5m2, e5m2, f32, nk_b256_vec_t, nk_dot_through_f32_state_haswell_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_haswell_, nk_load_e5m2x16_to_f32x8_haswell_,
                       nk_dots_partial_load_e5m2x16_to_f32x8_haswell_, nk_load_b256_haswell_,
                       nk_partial_load_b32x8_serial_, nk_dot_through_f32_update_haswell_,
                       nk_dot_through_f32_finalize_haswell_, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/8)

/* I8 GEMM: simd_width=16 (16 i8s = 16 bytes = 128-bit input) */
nk_define_dots_pack_size_(haswell, i8, i8, i32)
nk_define_dots_pack_(haswell, i8, i8, i32, nk_assign_from_to_)
nk_define_dots_symmetric_(i8_haswell, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_haswell_t, nk_b128_vec_t,
                          nk_dot_i8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                          nk_dot_i8x16_update_haswell, nk_dot_i8x16_finalize_haswell,
                          /*simd_width=*/16)
nk_define_dots_packed_(i8_haswell, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_haswell_t, nk_b128_vec_t,
                       nk_dot_i8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_haswell_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_haswell,
                       nk_dot_i8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

/* U8 GEMM: simd_width=16 (16 u8s = 16 bytes = 128-bit input) */
nk_define_dots_pack_size_(haswell, u8, u8, u32)
nk_define_dots_pack_(haswell, u8, u8, u32, nk_assign_from_to_)
nk_define_dots_symmetric_(u8_haswell, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_haswell_t, nk_b128_vec_t,
                          nk_dot_u8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                          nk_dot_u8x16_update_haswell, nk_dot_u8x16_finalize_haswell,
                          /*simd_width=*/16)
nk_define_dots_packed_(u8_haswell, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_haswell_t, nk_b128_vec_t,
                       nk_dot_u8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_haswell_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_haswell,
                       nk_dot_u8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_DOTS_HASWELL_H
