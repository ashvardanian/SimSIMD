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
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/dot/neon.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* F32 GEMM: depth_simd_step=2 (2 f32s = 8 bytes = 64-bit input for f64 upcast accumulation) */
nk_define_dots_pack_size_(f32, neon, f32, f32, /*depth_simd_step=*/2)
nk_define_dots_pack_(f32, neon, f32, f32, nk_assign_from_to_, /*depth_simd_step=*/2)
nk_define_dots_symmetric_(f32, neon, f32, f32, nk_b64_vec_t, nk_dot_f32x2_state_neon_t, nk_b128_vec_t,
                          nk_dot_f32x2_init_neon, nk_load_b64_neon_, nk_partial_load_b32x2_serial_,
                          nk_dot_f32x2_update_neon, nk_dot_f32x2_finalize_neon, /*depth_simd_step=*/2)
nk_define_dots_packed_(f32, neon, f32, f32, f32, nk_b64_vec_t, nk_dot_f32x2_state_neon_t, nk_b128_vec_t,
                       nk_dot_f32x2_init_neon, nk_load_b64_neon_, nk_partial_load_b32x2_serial_, nk_load_b64_neon_,
                       nk_partial_load_b32x2_serial_, nk_dot_f32x2_update_neon, nk_dot_f32x2_finalize_neon,
                       nk_partial_store_b32x4_serial_,
                       /*depth_simd_step=*/2)

/* F64 GEMM: depth_simd_step=2 (2 f64s = 16 bytes = NEON register width) */
nk_define_dots_pack_size_(f64, neon, f64, f64, /*depth_simd_step=*/2)
nk_define_dots_pack_(f64, neon, f64, f64, nk_assign_from_to_, /*depth_simd_step=*/2)
nk_define_dots_symmetric_(f64, neon, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_neon_t, nk_b256_vec_t,
                          nk_dot_f64x2_init_neon, nk_load_b128_neon_, nk_partial_load_b64x2_serial_,
                          nk_dot_f64x2_update_neon, nk_dot_f64x2_finalize_neon, /*depth_simd_step=*/2)
nk_define_dots_packed_(f64, neon, f64, f64, f64, nk_b128_vec_t, nk_dot_f64x2_state_neon_t, nk_b256_vec_t,
                       nk_dot_f64x2_init_neon, nk_load_b128_neon_, nk_partial_load_b64x2_serial_, nk_load_b128_neon_,
                       nk_partial_load_b64x2_serial_, nk_dot_f64x2_update_neon, nk_dot_f64x2_finalize_neon,
                       nk_partial_store_b64x4_serial_,
                       /*depth_simd_step=*/2)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEON_H
