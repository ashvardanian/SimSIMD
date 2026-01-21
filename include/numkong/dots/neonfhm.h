/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Arm NEON CPUs with FMLAL.
 *  @file include/numkong/dots/neonfhm.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  Uses FMLAL (FEAT_FHM) for widening fp16->f32 multiply-accumulate, which is 20-48% faster
 *  than the convert-then-FMA approach used in neonhalf.h.
 */
#ifndef NK_DOTS_NEONFHM_H
#define NK_DOTS_NEONFHM_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

#include "numkong/dot/neonfhm.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* F16 GEMM using FMLAL: depth_simd_dimensions=8 (8 f16s = 16 bytes = NEON register width) */
nk_define_dots_pack_size_(f16, neonfhm, f16, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f16, neonfhm, f16, f16, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f16, neonfhm, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                          nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                          nk_dot_f16x8_update_neonfhm, nk_dot_f16x8_finalize_neonfhm,
                          /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_packed_(f16, neonfhm, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                       nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_load_b128_neon_,
                       nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_neonfhm, nk_dot_f16x8_finalize_neonfhm,
                       nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E2M3FN GEMM using FMLAL: depth_simd_dimensions=8 (8 e2m3s = 8 bytes) */
nk_define_dots_pack_size_(e2m3, neonfhm, e2m3, e2m3, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e2m3, neonfhm, e2m3, e2m3, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e2m3, neonfhm, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x8_state_neonfhm_t, nk_b128_vec_t,
                          nk_dot_e2m3x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                          nk_dot_e2m3x8_update_neonfhm, nk_dot_e2m3x8_finalize_neonfhm,
                          /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_packed_(e2m3, neonfhm, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x8_state_neonfhm_t, nk_b128_vec_t,
                       nk_dot_e2m3x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x8_update_neonfhm,
                       nk_dot_e2m3x8_finalize_neonfhm, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E3M2FN GEMM using FMLAL: depth_simd_dimensions=8 (8 e3m2s = 8 bytes) */
nk_define_dots_pack_size_(e3m2, neonfhm, e3m2, e3m2, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e3m2, neonfhm, e3m2, e3m2, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e3m2, neonfhm, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x8_state_neonfhm_t, nk_b128_vec_t,
                          nk_dot_e3m2x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                          nk_dot_e3m2x8_update_neonfhm, nk_dot_e3m2x8_finalize_neonfhm,
                          /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_packed_(e3m2, neonfhm, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x8_state_neonfhm_t, nk_b128_vec_t,
                       nk_dot_e3m2x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                       nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x8_update_neonfhm,
                       nk_dot_e3m2x8_finalize_neonfhm, nk_partial_store_b32x4_serial_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#endif // NK_DOTS_NEONFHM_H
