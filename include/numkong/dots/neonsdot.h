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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* I8 GEMM: simd_width=16 (16 i8s = 16 bytes = NEON register width) */
nk_define_dots_pack_size_(neonsdot, i8, i8, i32)
nk_define_dots_pack_(neonsdot, i8, i8, i32, nk_assign_from_to_)
nk_define_dots_symmetric_(i8_neonsdot, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_neonsdot_t, nk_b128_vec_t,
                          nk_dot_i8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                          nk_dot_i8x16_update_neonsdot, nk_dot_i8x16_finalize_neonsdot,
                          /*simd_width=*/16)
nk_define_dots_packed_(i8_neonsdot, i8, i8, i32, nk_b128_vec_t, nk_dot_i8x16_state_neonsdot_t, nk_b128_vec_t,
                       nk_dot_i8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_i8x16_update_neonsdot,
                       nk_dot_i8x16_finalize_neonsdot, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

/* U8 GEMM: simd_width=16 (16 u8s = 16 bytes = NEON register width) */
nk_define_dots_pack_size_(neonsdot, u8, u8, u32)
nk_define_dots_pack_(neonsdot, u8, u8, u32, nk_assign_from_to_)
nk_define_dots_symmetric_(u8_neonsdot, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_neonsdot_t, nk_b128_vec_t,
                          nk_dot_u8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                          nk_dot_u8x16_update_neonsdot, nk_dot_u8x16_finalize_neonsdot,
                          /*simd_width=*/16)
nk_define_dots_packed_(u8_neonsdot, u8, u8, u32, nk_b128_vec_t, nk_dot_u8x16_state_neonsdot_t, nk_b128_vec_t,
                       nk_dot_u8x16_init_neonsdot, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_u8x16_update_neonsdot,
                       nk_dot_u8x16_finalize_neonsdot, nk_partial_store_b32x4_serial_,
                       /*simd_width=*/16)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEONSDOT_H
