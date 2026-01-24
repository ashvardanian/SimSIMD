/**
 *  @brief SIMD-accelerated GEMM for Integer Datatypes optimized for Intel Sierra Forest CPUs.
 *  @file include/numkong/dots/sierra.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  Uses AVX-VNNI (256-bit) for integer GEMM:
 *  - _mm256_dpbssds_epi32: i8 × i8 → i32 with saturation
 *  - _mm256_dpbuud_epi32: u8 × u8 → u32 without saturation
 */
#ifndef NK_DOTS_SIERRA_H
#define NK_DOTS_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

#include "numkong/dot/sierra.h"  // Sierra-specific dot product helpers
#include "numkong/dot/haswell.h" // Haswell partial load functions
#include "numkong/dots/serial.h" // GEMM macro definitions

#if defined(__cplusplus)
extern "C" {
#endif

/* I8 GEMM: depth_simd_dimensions=32 (32 i8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, i8, sierra, i8, i8, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, i8, sierra, i8, i8, nk_assign_from_to_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, i8, sierra, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_i8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                           nk_dot_i8x32_update_sierra, nk_dot_i8x32_finalize_sierra, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, i8, sierra, i8, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_i8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_i8x32_update_sierra,
                        nk_dot_i8x32_finalize_sierra, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

/* U8 GEMM: depth_simd_dimensions=32 (32 u8s = 32 bytes = AVX2 register width) */
nk_define_cross_pack_size_(dots, u8, sierra, u8, u8, /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_pack_(dots, u8, sierra, u8, u8, nk_assign_from_to_, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/1)
nk_define_cross_symmetric_(dots, u8, sierra, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                           nk_dot_u8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                           nk_dot_u8x32_update_sierra, nk_dot_u8x32_finalize_sierra, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)
nk_define_cross_packed_(dots, u8, sierra, u8, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_b128_vec_t,
                        nk_dot_u8x32_init_sierra, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                        nk_load_b256_haswell_, nk_partial_load_b8x32_serial_, nk_dot_u8x32_update_sierra,
                        nk_dot_u8x32_finalize_sierra, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/1)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SIERRA_H
