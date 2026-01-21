/**
 *  @brief SIMD-accelerated batch dot products (GEMM micro-kernels) optimized for Intel Skylake-X CPUs.
 *  @file include/numkong/dots/skylake.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @section skylake_dots_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     SKL         ICL         Genoa
 *      _mm512_fmadd_ps             VFMADD132PS (ZMM, ZMM, ZMM)     4cy @ p05   4cy @ p05   4cy @ p01
 *      _mm512_fmadd_pd             VFMADD132PD (ZMM, ZMM, ZMM)     4cy @ p05   4cy @ p05   4cy @ p01
 *      _mm512_cvtph_ps             VCVTPH2PS (ZMM, YMM)            5cy @ p05   5cy @ p05   5cy @ p01
 *      _mm512_loadu_ps             VMOVUPS (ZMM, M512)             7cy @ p23   7cy @ p23   7cy @ p23
 *
 *  GEMM micro-kernels tile the K dimension to maximize FMA throughput. Skylake-X server chips with
 *  dual FMA units achieve 0.5cy throughput, enabling 32 FLOPs/cycle for f32 or 16 FLOPs/cycle for f64.
 *  FP8 types (E4M3, E5M2) convert to f32 for accumulation, adding ~5cy latency per conversion.
 */
#ifndef NK_DOTS_SKYLAKE_H
#define NK_DOTS_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* F64 GEMM: depth_simd_dimensions=8 (8 f64s = 64 bytes = 1 cache line) */
nk_define_dots_pack_size_(f64, skylake, f64, f64, f64, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f64, skylake, f64, f64, f64, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f64, skylake, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                          nk_dot_f64x8_init_skylake, nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_,
                          nk_dot_f64x8_update_skylake, nk_dot_f64x8_finalize_skylake,
                          /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_packed_(f64, skylake, f64, f64, f64, nk_b512_vec_t, nk_dot_f64x8_state_skylake_t, nk_b256_vec_t,
                       nk_dot_f64x8_init_skylake, nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_,
                       nk_load_b512_skylake_, nk_partial_load_b64x8_skylake_, nk_dot_f64x8_update_skylake,
                       nk_dot_f64x8_finalize_skylake, nk_partial_store_b64x4_skylake_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* F32 GEMM: depth_simd_dimensions=8 (8 f32s = 32 bytes = half cache line) */
nk_define_dots_pack_size_(f32, skylake, f32, f32, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f32, skylake, f32, f32, f32, nk_assign_from_to_, /*depth_simd_dimensions=*/8,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f32, skylake, f32, f32, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b128_vec_t,
                          nk_dot_f32x8_init_skylake, nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_,
                          nk_dot_f32x8_update_skylake, nk_dot_f32x8_finalize_skylake,
                          /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_dots_packed_(f32, skylake, f32, f32, f32, nk_b256_vec_t, nk_dot_f32x8_state_skylake_t, nk_b128_vec_t,
                       nk_dot_f32x8_init_skylake, nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_,
                       nk_load_b256_haswell_, nk_partial_load_b32x8_skylake_, nk_dot_f32x8_update_skylake,
                       nk_dot_f32x8_finalize_skylake, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)

/* BF16 GEMM: depth_simd_dimensions=16 (16 bf16s = 32 bytes = half cache line), F32 accumulator */
nk_define_dots_pack_size_(bf16, skylake, bf16, f32, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(bf16, skylake, bf16, f32, f32, nk_bf16_to_f32, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(bf16, skylake, bf16, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_skylake_, nk_load_bf16x16_to_f32x16_skylake_,
                          nk_partial_load_bf16x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                          nk_dot_through_f32_finalize_skylake_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_packed_(bf16, skylake, bf16, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_skylake_, nk_load_bf16x16_to_f32x16_skylake_,
                       nk_partial_load_bf16x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                       nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                       nk_dot_through_f32_finalize_skylake_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* F16 GEMM: depth_simd_dimensions=16 (16 f16s = 32 bytes = half cache line), F32 accumulator */
nk_define_dots_pack_size_(f16, skylake, f16, f32, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(f16, skylake, f16, f32, f32, nk_f16_to_f32, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(f16, skylake, f16, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_skylake_, nk_load_f16x16_to_f32x16_skylake_,
                          nk_partial_load_f16x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                          nk_dot_through_f32_finalize_skylake_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_packed_(f16, skylake, f16, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_skylake_, nk_load_f16x16_to_f32x16_skylake_,
                       nk_partial_load_f16x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                       nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                       nk_dot_through_f32_finalize_skylake_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E4M3 GEMM: depth_simd_dimensions=16 (16 e4m3s = 16 bytes = quarter cache line), F32 accumulator */
nk_define_dots_pack_size_(e4m3, skylake, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e4m3, skylake, e4m3, f32, f32, nk_e4m3_to_f32, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e4m3, skylake, e4m3, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_skylake_, nk_load_e4m3x16_to_f32x16_skylake_,
                          nk_partial_load_e4m3x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                          nk_dot_through_f32_finalize_skylake_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_packed_(e4m3, skylake, e4m3, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_skylake_, nk_load_e4m3x16_to_f32x16_skylake_,
                       nk_partial_load_e4m3x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                       nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                       nk_dot_through_f32_finalize_skylake_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E5M2 GEMM: depth_simd_dimensions=16 (16 e5m2s = 16 bytes = quarter cache line), F32 accumulator */
nk_define_dots_pack_size_(e5m2, skylake, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e5m2, skylake, e5m2, f32, f32, nk_e5m2_to_f32, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e5m2, skylake, e5m2, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_skylake_, nk_load_e5m2x16_to_f32x16_skylake_,
                          nk_partial_load_e5m2x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                          nk_dot_through_f32_finalize_skylake_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_packed_(e5m2, skylake, e5m2, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_skylake_, nk_load_e5m2x16_to_f32x16_skylake_,
                       nk_partial_load_e5m2x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                       nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                       nk_dot_through_f32_finalize_skylake_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E2M3 GEMM: depth_simd_dimensions=16 (16 e2m3s = 16 bytes = quarter cache line), F32 accumulator */
nk_define_dots_pack_size_(e2m3, skylake, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e2m3, skylake, e2m3, f32, f32, nk_e2m3_to_f32, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e2m3, skylake, e2m3, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_skylake_, nk_load_e2m3x16_to_f32x16_skylake_,
                          nk_partial_load_e2m3x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                          nk_dot_through_f32_finalize_skylake_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_packed_(e2m3, skylake, e2m3, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_skylake_, nk_load_e2m3x16_to_f32x16_skylake_,
                       nk_partial_load_e2m3x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                       nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                       nk_dot_through_f32_finalize_skylake_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

/* E3M2 GEMM: depth_simd_dimensions=16 (16 e3m2s = 16 bytes = quarter cache line), F32 accumulator */
nk_define_dots_pack_size_(e3m2, skylake, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_pack_(e3m2, skylake, e3m2, f32, f32, nk_e3m2_to_f32, /*depth_simd_dimensions=*/16,
                     /*dimensions_per_value=*/1)
nk_define_dots_symmetric_(e3m2, skylake, e3m2, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                          nk_dot_through_f32_init_skylake_, nk_load_e3m2x16_to_f32x16_skylake_,
                          nk_partial_load_e3m2x16_to_f32x16_skylake_, nk_dot_through_f32_update_skylake_,
                          nk_dot_through_f32_finalize_skylake_,
                          /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_dots_packed_(e3m2, skylake, e3m2, f32, f32, nk_b512_vec_t, nk_dot_through_f32_state_skylake_t_, nk_b128_vec_t,
                       nk_dot_through_f32_init_skylake_, nk_load_e3m2x16_to_f32x16_skylake_,
                       nk_partial_load_e3m2x16_to_f32x16_skylake_, nk_load_b512_skylake_,
                       nk_partial_load_b32x16_skylake_, nk_dot_through_f32_update_skylake_,
                       nk_dot_through_f32_finalize_skylake_, nk_partial_store_b32x4_skylake_,
                       /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SKYLAKE_H
