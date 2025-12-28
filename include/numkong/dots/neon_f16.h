/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dots/neon_f16.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_NEON_F16_H
#define NK_DOTS_NEON_F16_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// F16 GEMM: k_tile=8 (8 f16s = 16 bytes = NEON register width)
NK_MAKE_DOTS_SERIAL_PACKED_SIZE(neon, f16, f32, 8)
NK_MAKE_DOTS_SERIAL_PACK(neon, f16, f32, 8)
NK_MAKE_DOTS_INNER(f16f16f32_neon, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neon_t, nk_dot_f16x8_init_neon,
                   nk_load_b128_neon_, nk_partial_load_b16x8_neon_, nk_dot_f16x8_update_neon,
                   nk_dot_f16x8_finalize_neon,
                   /*k_tile=*/8, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEON_F16
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEON_F16_H