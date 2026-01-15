/**
 *  @brief SIMD-accelerated batch dot products (GEMM micro-kernels) optimized for Intel Ice Lake CPUs.
 *  @file include/numkong/dots/ice.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section ice_dots_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_dpbusd_epi32         VPDPBUSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_dpwssd_epi32         VPDPWSSD (ZMM, ZMM, ZMM)        5cy @ p0    4cy @ p01
 *      _mm512_cvtepi8_epi32        VPMOVSXBD (ZMM, XMM)            3cy @ p5    3cy @ p12
 *      _mm512_loadu_si512          VMOVDQU64 (ZMM, M512)           7cy @ p23   7cy @ p23
 *
 *  Ice Lake's VNNI instructions accelerate int8 GEMM by computing 4-element dot products per lane.
 *  VPDPBUSD/VPDPWSSD bottleneck on port 0, limiting throughput to 1/cy. AMD Genoa achieves 0.5/cy
 *  via dual-issue on ports 0-1, making it significantly faster for quantized inference workloads.
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

/* I8 GEMM: simd_width=32 (32 i8s = 32 bytes = half cache line) */
nk_define_dots_pack_size_(ice, i8, i32)
nk_define_dots_pack_(ice, i8, i32)
nk_define_dots_packed_vectors_(i8_ice, i8, i32, nk_b256_vec_t, nk_dot_i8x32_state_ice_t, nk_b128_vec_t,
                               nk_dot_i8x32_init_ice, nk_load_b256_haswell_, nk_partial_load_b8x32_serial_,
                               nk_dot_i8x32_update_ice, nk_dot_i8x32_finalize_ice, nk_partial_store_b32x4_skylake_,
                               /*simd_width=*/32)

/* U8 GEMM: simd_width=64 (64 u8s = 64 bytes = 1 cache line) */
nk_define_dots_pack_size_(ice, u8, u32)
nk_define_dots_pack_(ice, u8, u32)
nk_define_dots_packed_vectors_(u8_ice, u8, u32, nk_b512_vec_t, nk_dot_u8x64_state_ice_t, nk_b128_vec_t,
                               nk_dot_u8x64_init_ice, nk_load_b512_skylake_, nk_partial_load_b8x64_skylake_,
                               nk_dot_u8x64_update_ice, nk_dot_u8x64_finalize_ice, nk_partial_store_b32x4_skylake_,
                               /*simd_width=*/64)

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
