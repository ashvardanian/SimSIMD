/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for AMD Genoa CPUs.
 *  @file include/numkong/dots/genoa.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_GENOA_H
#define NK_DOTS_GENOA_H

#if NK_TARGET_X86_
#if NK_TARGET_GENOA
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512bf16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512bf16"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// BF16 GEMM: k_tile=32 (32 bf16s = 64 bytes = 1 cache line)
NK_MAKE_DOTS_PACK_SIZE(genoa, bf16, f32)
NK_MAKE_DOTS_PACK(genoa, bf16, f32)
NK_MAKE_DOTS_VECTORS(bf16bf16f32_genoa, bf16, f32, nk_b512_vec_t, nk_dot_bf16x32_state_genoa_t,
                     nk_dot_bf16x32_init_genoa, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                     nk_dot_bf16x32_update_genoa, nk_dot_bf16x32_finalize_genoa,
                     /*k_tile=*/32, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// E4M3 GEMM: k_tile=64 (64 e4m3s = 64 bytes = 1 cache line), F32 accumulator
NK_MAKE_DOTS_PACK_SIZE(genoa, e4m3, f32)
NK_MAKE_DOTS_PACK(genoa, e4m3, f32)
NK_MAKE_DOTS_VECTORS(e4m3e4m3f32_genoa, e4m3, f32, nk_b512_vec_t, nk_dot_e4m3x64_state_genoa_t,
                     nk_dot_e4m3x64_init_genoa, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                     nk_dot_e4m3x64_update_genoa, nk_dot_e4m3x64_finalize_genoa,
                     /*k_tile=*/64, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// E5M2 GEMM: k_tile=64 (64 e5m2s = 64 bytes = 1 cache line), F32 accumulator
NK_MAKE_DOTS_PACK_SIZE(genoa, e5m2, f32)
NK_MAKE_DOTS_PACK(genoa, e5m2, f32)
NK_MAKE_DOTS_VECTORS(e5m2e5m2f32_genoa, e5m2, f32, nk_b512_vec_t, nk_dot_e5m2x64_state_genoa_t,
                     nk_dot_e5m2x64_init_genoa, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                     nk_dot_e5m2x64_update_genoa, nk_dot_e5m2x64_finalize_genoa,
                     /*k_tile=*/64, /*k_unroll=*/1, /*MR=*/4, /*MC=*/128, /*NC=*/2048, /*KC=*/256)

// Compact function: F32→BF16 conversion (reuses serial implementation logic)
NK_PUBLIC void nk_dots_bf16bf16bf16_genoa(void *c, nk_size_t m, nk_size_t n, nk_size_t c_stride) {
    nk_dots_bf16bf16bf16_serial(c, m, n, c_stride);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_

#endif // NK_DOTS_GENOA_H