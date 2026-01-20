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
#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* BF16 GEMM: depth_simd_step=32 (32 bf16s = 64 bytes = 1 cache line) */
nk_define_dots_pack_size_(bf16, genoa, bf16, bf16, f32, /*depth_simd_step=*/32)
nk_define_dots_pack_(bf16, genoa, bf16, bf16, f32, nk_assign_from_to_, /*depth_simd_step=*/32)
nk_define_dots_symmetric_(bf16, genoa, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_, nk_b128_vec_t,
                          nk_dot_through_bf16_init_genoa_, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                          nk_dot_through_bf16_update_genoa_, nk_dot_through_bf16_finalize_genoa_,
                          /*depth_simd_step=*/32)
nk_define_dots_packed_(bf16, genoa, bf16, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_, nk_b128_vec_t,
                       nk_dot_through_bf16_init_genoa_, nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_,
                       nk_load_b512_skylake_, nk_partial_load_b16x32_skylake_, nk_dot_through_bf16_update_genoa_,
                       nk_dot_through_bf16_finalize_genoa_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_step=*/32)

/* E4M3 GEMM: depth_simd_step=32 (32 e4m3s = 32 bytes = half cache line), F32 accumulator */
nk_define_dots_pack_size_(e4m3, genoa, e4m3, e4m3, f32, /*depth_simd_step=*/32)
nk_define_dots_pack_(e4m3, genoa, e4m3, bf16, f32, nk_e4m3_to_bf16, /*depth_simd_step=*/32)
nk_define_dots_symmetric_(e4m3, genoa, e4m3, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_, nk_b128_vec_t,
                          nk_dot_through_bf16_init_genoa_, nk_load_e4m3x32_to_bf16x32_genoa_,
                          nk_partial_load_e4m3x32_to_bf16x32_genoa_, nk_dot_through_bf16_update_genoa_,
                          nk_dot_through_bf16_finalize_genoa_,
                          /*depth_simd_step=*/32)
nk_define_dots_packed_(e4m3, genoa, e4m3, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_, nk_b128_vec_t,
                       nk_dot_through_bf16_init_genoa_, nk_load_e4m3x32_to_bf16x32_genoa_,
                       nk_partial_load_e4m3x32_to_bf16x32_genoa_, nk_load_b512_skylake_,
                       nk_partial_load_b16x32_skylake_, nk_dot_through_bf16_update_genoa_,
                       nk_dot_through_bf16_finalize_genoa_, nk_partial_store_b32x4_serial_,
                       /*depth_simd_step=*/32)

/* E5M2 GEMM: depth_simd_step=32 (32 e5m2s = 32 bytes = half cache line), F32 accumulator */
nk_define_dots_pack_size_(e5m2, genoa, e5m2, e5m2, f32, /*depth_simd_step=*/32)
nk_define_dots_pack_(e5m2, genoa, e5m2, bf16, f32, nk_e5m2_to_bf16, /*depth_simd_step=*/32)
nk_define_dots_symmetric_(e5m2, genoa, e5m2, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_, nk_b128_vec_t,
                          nk_dot_through_bf16_init_genoa_, nk_load_e5m2x32_to_bf16x32_genoa_,
                          nk_partial_load_e5m2x32_to_bf16x32_genoa_, nk_dot_through_bf16_update_genoa_,
                          nk_dot_through_bf16_finalize_genoa_,
                          /*depth_simd_step=*/32)
nk_define_dots_packed_(e5m2, genoa, e5m2, bf16, f32, nk_b512_vec_t, nk_dot_through_bf16_state_genoa_t_, nk_b128_vec_t,
                       nk_dot_through_bf16_init_genoa_, nk_load_e5m2x32_to_bf16x32_genoa_,
                       nk_partial_load_e5m2x32_to_bf16x32_genoa_, nk_load_b512_skylake_,
                       nk_partial_load_b16x32_skylake_, nk_dot_through_bf16_update_genoa_,
                       nk_dot_through_bf16_finalize_genoa_, nk_partial_store_b32x4_serial_,
                       /*depth_simd_step=*/32)

// Compact function: F32 → BF16 conversion (reuses serial implementation logic)
NK_PUBLIC void nk_dots_compact_bf16_genoa(void *c, nk_size_t row_count, nk_size_t column_count, nk_size_t c_stride) {
    nk_dots_compact_bf16_serial(c, row_count, column_count, c_stride);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_GENOA
#endif // NK_TARGET_X86_

#endif // NK_DOTS_GENOA_H
