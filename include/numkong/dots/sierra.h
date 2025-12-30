/**
 *  @brief SIMD-accelerated GEMM for Integer Datatypes optimized for Intel Sierra Forest CPUs.
 *  @file include/numkong/dots/sierra.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  Uses AVX-VNNI (256-bit) for efficient integer GEMM:
 *  - _mm256_dpbssds_epi32: i8×i8→i32 with saturation
 *  - _mm256_dpbuud_epi32: u8×u8→u32 without saturation
 */
#ifndef NK_DOTS_SIERRA_H
#define NK_DOTS_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA
#pragma GCC push_options
#pragma GCC target("avx2", "bmi2", "f16c", "fma", "avxvnni", "avxvnniint8")
#pragma clang attribute push(__attribute__((target("avx2,bmi2,f16c,fma,avxvnni,avxvnniint8"))), apply_to = function)

#include "numkong/dot/sierra.h"   // Sierra-specific dot product helpers
#include "numkong/dot/haswell.h"  // Haswell partial load functions
#include "numkong/dots/serial.h"  // GEMM macro definitions

#if defined(__cplusplus)
extern "C" {
#endif

// I8 GEMM: k_tile=32 (32 i8s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(sierra, i8, i32)
NK_MAKE_DOTS_PACK(sierra, i8, i32)
NK_MAKE_DOTS_VECTORS(i8i8i32_sierra, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_sierra_t, nk_dot_i8x32_init_sierra,
                     nk_load_b256_sierra_, nk_partial_load_b8x32_sierra_, nk_dot_i8x32_update_sierra,
                     nk_dot_i8x32_finalize_sierra,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// U8 GEMM: k_tile=32 (32 u8s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(sierra, u8, i32)
NK_MAKE_DOTS_PACK(sierra, u8, i32)
NK_MAKE_DOTS_VECTORS(u8u8i32_sierra, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_sierra_t, nk_dot_u8x32_init_sierra,
                     nk_load_b256_sierra_, nk_partial_load_b8x32_sierra_, nk_dot_u8x32_update_sierra,
                     nk_dot_u8x32_finalize_sierra,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SIERRA_H