/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/dots/ice.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_ICE_H
#define NK_DOTS_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512vnni")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512vnni"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// I8 GEMM: k_tile=64 (64 i8s = 64 bytes = 1 cache line)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(ice, i8, i32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_SERIAL_PACK(ice, i8, i32, NK_DOTS_SERIAL_TILE_K_I8)
NK_MAKE_DOTS_INNER(i8i8i32_ice, i8, i32, nk_b512_vec_t, nk_dot_i8x64_state_ice_t, nk_dot_i8x64_init_ice,
                   nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_, nk_dot_i8x64_update_ice,
                   nk_dot_i8x64_finalize_ice,
                   /*k_tile=*/64, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// U8 GEMM: k_tile=64 (64 u8s = 64 bytes = 1 cache line)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(ice, u8, i32, NK_DOTS_SERIAL_TILE_K_U8)
NK_MAKE_DOTS_SERIAL_PACK(ice, u8, i32, NK_DOTS_SERIAL_TILE_K_U8)
NK_MAKE_DOTS_INNER(u8u8i32_ice, u8, u32, nk_b512_vec_t, nk_dot_u8x64_state_ice_t, nk_dot_u8x64_init_ice,
                   nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_, nk_dot_u8x64_update_ice,
                   nk_dot_u8x64_finalize_ice,
                   /*k_tile=*/64, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_DOTS_ICE_H