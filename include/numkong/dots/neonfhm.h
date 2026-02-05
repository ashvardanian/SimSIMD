/**
 *  @brief SIMD-accelerated Batched Dot Products for NEON FHM.
 *  @file include/numkong/dots/neonfhm.h
 *  @author Ash Vardanian
 *  @date December 28, 2025
 *
 *  @sa include/numkong/dots.h
 *
 *  Uses FMLAL (FEAT_FHM) for widening fp16->f32 multiply-accumulate, which is 20-48% faster
 *  than the convert-then-FMA approach used in neonhalf.h.
 */
#ifndef NK_DOTS_NEONFHM_H
#define NK_DOTS_NEONFHM_H

#if defined(__cplusplus)
extern "C" {
#endif

#if NK_TARGET_ARM_
#if NK_TARGET_NEONFHM
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

#include "numkong/dot/neonfhm.h"

/* F16 GEMM using FMLAL: depth_simd_dimensions=8 (8 f16s = 16 bytes = NEON register width) */
nk_define_cross_pack_size_(dots, f16, neonfhm, f16, f16, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, f16, neonfhm, f16, f16, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, f16, neonfhm, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                           nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                           nk_dot_f16x8_update_neonfhm, nk_dot_f16x8_finalize_neonfhm, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, f16, neonfhm, f16, f16, f32, nk_b128_vec_t, nk_dot_f16x8_state_neonfhm_t, nk_b128_vec_t,
                        nk_dot_f16x8_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b16x8_serial_,
                        nk_load_b128_neon_, nk_partial_load_b16x8_serial_, nk_dot_f16x8_update_neonfhm,
                        nk_dot_f16x8_finalize_neonfhm, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* E2M3FN GEMM using FMLAL with TBL: depth_simd_dimensions=16 (16 e2m3s = 16 bytes) */
nk_define_cross_pack_size_(dots, e2m3, neonfhm, e2m3, e2m3, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e2m3, neonfhm, e2m3, e2m3, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e2m3, neonfhm, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_neonfhm_t, nk_b128_vec_t,
                           nk_dot_e2m3x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e2m3x16_update_neonfhm, nk_dot_e2m3x16_finalize_neonfhm,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e2m3, neonfhm, e2m3, e2m3, f32, nk_b128_vec_t, nk_dot_e2m3x16_state_neonfhm_t,
                        nk_b128_vec_t, nk_dot_e2m3x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e2m3x16_update_neonfhm,
                        nk_dot_e2m3x16_finalize_neonfhm, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E3M2FN GEMM using FMLAL with TBL: depth_simd_dimensions=16 (16 e2m3s = 16 bytes) */
nk_define_cross_pack_size_(dots, e3m2, neonfhm, e3m2, e3m2, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, e3m2, neonfhm, e3m2, e3m2, nk_assign_from_to_, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, e3m2, neonfhm, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_neonfhm_t, nk_b128_vec_t,
                           nk_dot_e3m2x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                           nk_dot_e3m2x16_update_neonfhm, nk_dot_e3m2x16_finalize_neonfhm,
                           nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, e3m2, neonfhm, e3m2, e3m2, f32, nk_b128_vec_t, nk_dot_e3m2x16_state_neonfhm_t,
                        nk_b128_vec_t, nk_dot_e3m2x16_init_neonfhm, nk_load_b128_neon_, nk_partial_load_b8x16_serial_,
                        nk_load_b128_neon_, nk_partial_load_b8x16_serial_, nk_dot_e3m2x16_update_neonfhm,
                        nk_dot_e3m2x16_finalize_neonfhm, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_DOTS_NEONFHM_H
