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
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// BF16 GEMM: k_tile=32 (32 bf16s = 64 bytes = 1 cache line)
nk_make_dots_pack_size_(genoa, bf16, f32)
nk_make_dots_pack_(genoa, bf16, f32)
nk_make_dots_packed_vectors_(bf16_genoa, bf16, f32, nk_b512_vec_t, nk_dot_bf16x32_state_genoa_t, nk_b128_vec_t,
                             nk_dot_bf16x32_init_genoa, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                             nk_dot_bf16x32_update_genoa, nk_dot_bf16x32_finalize_genoa,
                             nk_partial_store_b32x4_skylake_,
                             /*k_tile=*/32)

// E4M3 GEMM: k_tile=32 (32 e4m3s = 32 bytes = half cache line), F32 accumulator
nk_make_dots_pack_size_(genoa, e4m3, f32)
nk_make_dots_pack_(genoa, e4m3, f32)
nk_make_dots_packed_vectors_(e4m3_genoa, e4m3, f32, nk_b256_vec_t, nk_dot_e4m3x32_state_genoa_t, nk_b128_vec_t,
                             nk_dot_e4m3x32_init_genoa, nk_load_b256_haswell_, nk_partial_load_u1x32_serial_,
                             nk_dot_e4m3x32_update_genoa, nk_dot_e4m3x32_finalize_genoa,
                             nk_partial_store_b32x4_skylake_,
                             /*k_tile=*/32)

// E5M2 GEMM: k_tile=32 (32 e5m2s = 32 bytes = half cache line), F32 accumulator
nk_make_dots_pack_size_(genoa, e5m2, f32)
nk_make_dots_pack_(genoa, e5m2, f32)
nk_make_dots_packed_vectors_(e5m2_genoa, e5m2, f32, nk_b256_vec_t, nk_dot_e5m2x32_state_genoa_t, nk_b128_vec_t,
                             nk_dot_e5m2x32_init_genoa, nk_load_b256_haswell_, nk_partial_load_u1x32_serial_,
                             nk_dot_e5m2x32_update_genoa, nk_dot_e5m2x32_finalize_genoa,
                             nk_partial_store_b32x4_skylake_,
                             /*k_tile=*/32)

// Compact function: F32 → BF16 conversion (reuses serial implementation logic)
NK_PUBLIC void nk_dots_compact_bf16_genoa(void *c, nk_size_t row_count, nk_size_t column_count, nk_size_t c_stride) {
    nk_dots_compact_bf16_serial(c, row_count, column_count, c_stride);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_

#endif // NK_DOTS_GENOA_H
