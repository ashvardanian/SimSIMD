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
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(skylake, f64, f64, NK_DOTS_SERIAL_TILE_K_F64)
NK_MAKE_DOTS_SERIAL_PACK(skylake, f64, f64, NK_DOTS_SERIAL_TILE_K_F64)
NK_MAKE_DOTS_INNER(f64f64f64_skylake, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_dot_f64x8_init_skylake,
                   nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_, nk_dot_f64x8_update_skylake,
                   nk_dot_f64x8_finalize_skylake,
                   /*k_tile=*/8, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// F32 GEMM: k_tile=16 (16 f32s = 64 bytes = 1 cache line)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(skylake, f32, f32, NK_DOTS_SERIAL_TILE_K_F32)
NK_MAKE_DOTS_SERIAL_PACK(skylake, f32, f32, NK_DOTS_SERIAL_TILE_K_F32)
NK_MAKE_DOTS_INNER(f32f32f32_skylake, f32, f32, nk_b512_vec_t, nk_dot_f32x16_state_skylake_t,
                   nk_dot_f32x16_init_skylake, nk_load_b512_skylake_, nk_partial_load_b32x16_skylake_,
                   nk_dot_f32x16_update_skylake, nk_dot_f32x16_finalize_skylake,
                   /*k_tile=*/16, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// E4M3 GEMM: k_tile=64 (64 e4m3s = 64 bytes = 1 cache line), F32 accumulator
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(skylake, e4m3, f32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_SERIAL_PACK(skylake, e4m3, f32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_INNER(e4m3e4m3f32_skylake, e4m3, f32, nk_b512_vec_t, nk_dot_e4m3x64_state_skylake_t,
                   nk_dot_e4m3x64_init_skylake, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                   nk_dot_e4m3x64_update_skylake, nk_dot_e4m3x64_finalize_skylake,
                   /*k_tile=*/64, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// E5M2 GEMM: k_tile=64 (64 e5m2s = 64 bytes = 1 cache line), F32 accumulator
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(skylake, e5m2, f32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_SERIAL_PACK(skylake, e5m2, f32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_INNER(e5m2e5m2f32_skylake, e5m2, f32, nk_b512_vec_t, nk_dot_e5m2x64_state_skylake_t,
                   nk_dot_e5m2x64_init_skylake, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                   nk_dot_e5m2x64_update_skylake, nk_dot_e5m2x64_finalize_skylake,
                   /*k_tile=*/64, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SKYLAKE_H