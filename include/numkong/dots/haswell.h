/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/dots/haswell.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_HASWELL_H
#define NK_DOTS_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// F32 GEMM: k_tile=4 (4 f32s = 16 bytes for f32->f64 upcast accumulation)
nk_make_dots_pack_size_(haswell, f32, f32)
nk_make_dots_pack_(haswell, f32, f32)
nk_make_dots_packed_vectors_(f32_haswell, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_haswell_t, nk_b128_vec_t,
                             nk_dot_f32x4_init_haswell, nk_load_b128_haswell_, nk_partial_load_b32x4_serial_,
                             nk_dot_f32x4_update_haswell, nk_dot_f32x4_finalize_haswell, nk_partial_store_b32x4_serial_,
                             /*k_tile=*/4)

// F64 GEMM: k_tile=4 (4 f64s = 32 bytes = AVX2 register width)
nk_make_dots_pack_size_(haswell, f64, f64)
nk_make_dots_pack_(haswell, f64, f64)
nk_make_dots_packed_vectors_(f64_haswell, f64, f64, nk_b256_vec_t, nk_dot_f64x4_state_haswell_t, nk_b256_vec_t,
                             nk_dot_f64x4_init_haswell, nk_load_b256_haswell_, nk_partial_load_b64x4_serial_,
                             nk_dot_f64x4_update_haswell, nk_dot_f64x4_finalize_haswell, nk_partial_store_b64x4_serial_,
                             /*k_tile=*/4)

// F16 GEMM: k_tile=8 (8 f16s = 16 bytes = 128-bit input)
nk_make_dots_pack_size_(haswell, f16, f32)
nk_make_dots_pack_(haswell, f16, f32)
nk_make_dots_packed_vectors_(f16_haswell, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_haswell_t, nk_b128_vec_t,
                             nk_dot_f16x8_init_haswell, nk_load_b128_haswell_, nk_partial_load_b16x8_serial_,
                             nk_dot_f16x8_update_haswell, nk_dot_f16x8_finalize_haswell, nk_partial_store_b32x4_serial_,
                             /*k_tile=*/8)

// BF16 GEMM: k_tile=8 (8 bf16s = 16 bytes = 128-bit input)
nk_make_dots_pack_size_(haswell, bf16, f32)
nk_make_dots_pack_(haswell, bf16, f32)
nk_make_dots_packed_vectors_(bf16_haswell, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_haswell_t, nk_b128_vec_t,
                             nk_dot_bf16x8_init_haswell, nk_load_b128_haswell_, nk_partial_load_b16x8_serial_,
                             nk_dot_bf16x8_update_haswell, nk_dot_bf16x8_finalize_haswell,
                             nk_partial_store_b32x4_serial_,
                             /*k_tile=*/8)

// E4M3 GEMM: k_tile=16 (16 e4m3s = 16 bytes = 128-bit input)
nk_make_dots_pack_size_(haswell, e4m3, f32)
nk_make_dots_pack_(haswell, e4m3, f32)
nk_make_dots_packed_vectors_(e4m3_haswell, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_haswell_t, nk_b128_vec_t,
                             nk_dot_e4m3x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_u1x16_serial_,
                             nk_dot_e4m3x16_update_haswell, nk_dot_e4m3x16_finalize_haswell,
                             nk_partial_store_b32x4_serial_,
                             /*k_tile=*/16)

// E5M2 GEMM: k_tile=16 (16 e5m2s = 16 bytes = 128-bit input)
nk_make_dots_pack_size_(haswell, e5m2, f32)
nk_make_dots_pack_(haswell, e5m2, f32)
nk_make_dots_packed_vectors_(e5m2_haswell, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_haswell_t, nk_b128_vec_t,
                             nk_dot_e5m2x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_u1x16_serial_,
                             nk_dot_e5m2x16_update_haswell, nk_dot_e5m2x16_finalize_haswell,
                             nk_partial_store_b32x4_serial_,
                             /*k_tile=*/16)

// I8 GEMM: k_tile=16 (16 i8s = 16 bytes = 128-bit input)
nk_make_dots_pack_size_(haswell, i8, i32)
nk_make_dots_pack_(haswell, i8, i32)
nk_make_dots_packed_vectors_(i8_haswell, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_haswell_t, nk_b128_vec_t,
                             nk_dot_i8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_u1x16_serial_,
                             nk_dot_i8x16_update_haswell, nk_dot_i8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                             /*k_tile=*/16)

// U8 GEMM: k_tile=16 (16 u8s = 16 bytes = 128-bit input)
nk_make_dots_pack_size_(haswell, u8, u32)
nk_make_dots_pack_(haswell, u8, u32)
nk_make_dots_packed_vectors_(u8_haswell, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_haswell_t, nk_b128_vec_t,
                             nk_dot_u8x16_init_haswell, nk_load_b128_haswell_, nk_partial_load_u1x16_serial_,
                             nk_dot_u8x16_update_haswell, nk_dot_u8x16_finalize_haswell, nk_partial_store_b32x4_serial_,
                             /*k_tile=*/16)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_DOTS_HASWELL_H
