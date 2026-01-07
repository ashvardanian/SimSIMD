/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/dots/ice.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_DOTS_ICE_H
#define NK_DOTS_ICE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

// I8 GEMM: k_tile=32 (32 i8s = 32 bytes = half cache line)
nk_make_dots_pack_size_(ice, i8, i32)
nk_make_dots_pack_(ice, i8, i32)
nk_make_dots_packed_vectors_(i8_ice, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_ice_t, nk_b128_vec_t,
                             nk_dot_i8x32_init_ice, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                             nk_dot_i8x32_update_ice, nk_dot_i8x32_finalize_ice, nk_partial_store_b32x4_skylake_,
                             /*k_tile=*/32)

// U8 GEMM: k_tile=64 (64 u8s = 64 bytes = 1 cache line)
nk_make_dots_pack_size_(ice, u8, u32)
nk_make_dots_pack_(ice, u8, u32)
nk_make_dots_packed_vectors_(u8_ice, u8, u32, nk_b512_vec_t, nk_dot_u8x64_state_ice_t, nk_b128_vec_t,
                             nk_dot_u8x64_init_ice, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                             nk_dot_u8x64_update_ice, nk_dot_u8x64_finalize_ice, nk_partial_store_b32x4_skylake_,
                             /*k_tile=*/64)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_ICE
#endif // NK_TARGET_X86_

#endif // NK_DOTS_ICE_H
