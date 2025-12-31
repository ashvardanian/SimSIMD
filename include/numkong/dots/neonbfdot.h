/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON-capable CPUs.
 *  @file include/numkong/dots/neonbfdot.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_NEONBFDOT_H
#define NK_DOTS_NEONBFDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONBFDOT
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// BF16 GEMM: simd_width=8 (8 bf16s = 16 bytes = NEON register width)
nk_make_dots_pack_size_(neonbfdot, bf16, f32)
nk_make_dots_pack_(neonbfdot, bf16, f32)
nk_make_dots_inner_vectors_(bf16bf16f32_neonbfdot, bf16, f32, nk_b128_vec_t, nk_dot_bf16x8_state_neonbfdot_t,
                            nk_b128_vec_t, nk_dot_bf16x8_init_neonbfdot, nk_load_b128_neon_,
                            nk_partial_load_b16x8_neon_, nk_dot_bf16x8_update_neonbfdot,
                            nk_dot_bf16x8_finalize_neonbfdot, nk_partial_store_b32x4_neon_,
                            /*k_tile=*/8)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEONBFDOT_H