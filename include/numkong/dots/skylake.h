/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/dots/skylake.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_SKYLAKE_H
#define NK_DOTS_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "bmi2")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,bmi2"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// F64 GEMM: k_tile=8 (8 f64s = 64 bytes = 1 cache line)
nk_make_dots_pack_size_(skylake, f64, f64)
nk_make_dots_pack_(skylake, f64, f64)
nk_make_dots_inner_vectors_(f64f64f64_skylake, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                            nk_dot_f64x8_init_skylake, nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_,
                            nk_dot_f64x8_update_skylake, nk_dot_f64x8_finalize_skylake, nk_partial_store_b64x4_skylake_,
                            /*k_tile=*/8)

// F32 GEMM: k_tile=8 (8 f32s = 32 bytes = half cache line)
nk_make_dots_pack_size_(skylake, f32, f32)
nk_make_dots_pack_(skylake, f32, f32)
nk_make_dots_inner_vectors_(f32f32f32_skylake, f32, f32, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b128_vec_t,
                            nk_dot_f32x8_init_skylake, nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_,
                            nk_dot_f32x8_update_skylake, nk_dot_f32x8_finalize_skylake, nk_partial_store_b32x4_skylake_,
                            /*k_tile=*/8)

// E4M3 GEMM: k_tile=16 (16 e4m3s = 16 bytes = quarter cache line), F32 accumulator
nk_make_dots_pack_size_(skylake, e4m3, f32)
nk_make_dots_pack_(skylake, e4m3, f32)
nk_make_dots_inner_vectors_(e4m3e4m3f32_skylake, e4m3, f32, nk_b128_vec_t, nk_dot_e4m3x16_state_skylake_t,
                            nk_b128_vec_t, nk_dot_e4m3x16_init_skylake, nk_load_b128_haswell_,
                            nk_partial_load_b8x16_skylake_, nk_dot_e4m3x16_update_skylake,
                            nk_dot_e4m3x16_finalize_skylake, nk_partial_store_b32x4_skylake_,
                            /*k_tile=*/16)

// E5M2 GEMM: k_tile=16 (16 e5m2s = 16 bytes = quarter cache line), F32 accumulator
nk_make_dots_pack_size_(skylake, e5m2, f32)
nk_make_dots_pack_(skylake, e5m2, f32)
nk_make_dots_inner_vectors_(e5m2e5m2f32_skylake, e5m2, f32, nk_b128_vec_t, nk_dot_e5m2x16_state_skylake_t,
                            nk_b128_vec_t, nk_dot_e5m2x16_init_skylake, nk_load_b128_haswell_,
                            nk_partial_load_b8x16_skylake_, nk_dot_e5m2x16_update_skylake,
                            nk_dot_e5m2x16_finalize_skylake, nk_partial_store_b32x4_skylake_,
                            /*k_tile=*/16)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SKYLAKE_H
