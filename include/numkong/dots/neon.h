/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dots/neon.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_NEON_H
#define NK_DOTS_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// F32 GEMM: k_tile=4 (4 f32s = 16 bytes = NEON register width)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(neon, f32, f32, 4)
NK_MAKE_DOTS_SERIAL_PACK(neon, f32, f32, 4)
NK_MAKE_DOTS_INNER(f32f32f32_neon, f32, f32, nk_b128_vec_t, nk_dot_f32x4_state_neon_t, nk_dot_f32x4_init_neon,
                   nk_load_b128_neon_, nk_partial_load_b32x4_neon_, nk_dot_f32x4_update_neon,
                   nk_dot_f32x4_finalize_neon,
                   /*k_tile=*/4, /*MR=*/4, /*MC=*/64, /*NC=*/1024, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEON_H