/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Haswell CPUs.
 *  @file include/numkong/dots/haswell.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_HASWELL_H
#define NK_DOTS_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma")
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma"))), apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// F32 GEMM: k_tile=8 (8 f32s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, f32, f32)
NK_MAKE_DOTS_PACK(haswell, f32, f32)
NK_MAKE_DOTS_VECTORS(f32f32f32_haswell, f32, f32, nk_b256_vec_t, nk_dot_f32x8_state_haswell_t,
                     nk_dot_f32x8_init_haswell, nk_load_b256_haswell_, nk_partial_load_b32x8_haswell_,
                     nk_dot_f32x8_update_haswell, nk_dot_f32x8_finalize_haswell,
                     /*k_tile=*/8, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// F16 GEMM: k_tile=16 (16 f16s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, f16, f32)
NK_MAKE_DOTS_PACK(haswell, f16, f32)
NK_MAKE_DOTS_VECTORS(f16f16f32_haswell, f16, f32, nk_b256_vec_t, nk_dot_f16x16_state_haswell_t,
                     nk_dot_f16x16_init_haswell, nk_load_b256_haswell_, nk_partial_load_b16x16_haswell_,
                     nk_dot_f16x16_update_haswell, nk_dot_f16x16_finalize_haswell,
                     /*k_tile=*/16, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// BF16 GEMM: k_tile=16 (16 bf16s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, bf16, f32)
NK_MAKE_DOTS_PACK(haswell, bf16, f32)
NK_MAKE_DOTS_VECTORS(bf16bf16f32_haswell, bf16, f32, nk_b256_vec_t, nk_dot_bf16x16_state_haswell_t,
                     nk_dot_bf16x16_init_haswell, nk_load_b256_haswell_, nk_partial_load_b16x16_haswell_,
                     nk_dot_bf16x16_update_haswell, nk_dot_bf16x16_finalize_haswell,
                     /*k_tile=*/16, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// E4M3 GEMM: k_tile=32 (32 e4m3s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, e4m3, f32)
NK_MAKE_DOTS_PACK(haswell, e4m3, f32)
NK_MAKE_DOTS_VECTORS(e4m3e4m3f32_haswell, e4m3, f32, nk_b256_vec_t, nk_dot_e4m3x32_state_haswell_t,
                     nk_dot_e4m3x32_init_haswell, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                     nk_dot_e4m3x32_update_haswell, nk_dot_e4m3x32_finalize_haswell,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// E5M2 GEMM: k_tile=32 (32 e5m2s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, e5m2, f32)
NK_MAKE_DOTS_PACK(haswell, e5m2, f32)
NK_MAKE_DOTS_VECTORS(e5m2e5m2f32_haswell, e5m2, f32, nk_b256_vec_t, nk_dot_e5m2x32_state_haswell_t,
                     nk_dot_e5m2x32_init_haswell, nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_,
                     nk_dot_e5m2x32_update_haswell, nk_dot_e5m2x32_finalize_haswell,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// I8 GEMM: k_tile=32 (32 i8s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, i8, i32)
NK_MAKE_DOTS_PACK(haswell, i8, i32)
NK_MAKE_DOTS_VECTORS(i8i8i32_haswell, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_haswell_t, nk_dot_i8x32_init_haswell,
                     nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_i8x32_update_haswell,
                     nk_dot_i8x32_finalize_haswell,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// U8 GEMM: k_tile=32 (32 u8s = 32 bytes = AVX2 register width)
NK_MAKE_DOTS_PACK_SIZE(haswell, u8, i32)
NK_MAKE_DOTS_PACK(haswell, u8, i32)
NK_MAKE_DOTS_VECTORS(u8u8i32_haswell, u8, u32, nk_b256_vec_t, nk_dot_u8x32_state_haswell_t, nk_dot_u8x32_init_haswell,
                     nk_load_b256_haswell_, nk_partial_load_b8x32_haswell_, nk_dot_u8x32_update_haswell,
                     nk_dot_u8x32_finalize_haswell,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_

#endif // NK_DOTS_HASWELL_H