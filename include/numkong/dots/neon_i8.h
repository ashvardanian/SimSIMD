/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dots/neonsdot.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_NEONSDOT_H
#define NK_DOTS_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// I8 GEMM: k_tile=16 (16 i8s = 16 bytes = NEON register width)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(neon, i8, i32, 16)
NK_MAKE_DOTS_SERIAL_PACK(neon, i8, i32, 16)
NK_MAKE_DOTS_INNER(i8i8i32_neon, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_neon_t, nk_dot_i8x16_init_neon,
                   nk_load_b128_neon_, nk_partial_load_b8x16_neon_, nk_dot_i8x16_update_neon,
                   nk_dot_i8x16_finalize_neon,
                   /*k_tile=*/16, /*k_unroll=*/1, /*MR=*/4, /*MC=*/64, /*NC=*/1024, /*KC=*/256)

// U8 GEMM: k_tile=16 (16 u8s = 16 bytes = NEON register width)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(neon, u8, i32, 16)
NK_MAKE_DOTS_SERIAL_PACK(neon, u8, i32, 16)
NK_MAKE_DOTS_INNER(u8u8i32_neon, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_neon_t, nk_dot_u8x16_init_neon,
                   nk_load_b128_neon_, nk_partial_load_b8x16_neon_, nk_dot_u8x16_update_neon,
                   nk_dot_u8x16_finalize_neon,
                   /*k_tile=*/16, /*k_unroll=*/1, /*MR=*/4, /*MC=*/64, /*NC=*/1024, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEONSDOT_H