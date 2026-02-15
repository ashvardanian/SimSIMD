/**
 *  @brief SIMD-accelerated Batched Set Distances for Ice Lake.
 *  @file include/numkong/sets/icelake.h
 *  @author Ash Vardanian
 *  @date January 25, 2026
 *
 *  @sa include/numkong/sets.h
 *
 *  @section sets_icelake_instructions Key AVX-512 Set Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm512_xor_si512            VPXORQ (ZMM, ZMM, ZMM)          1cy         0.33/cy     p05
 *      _mm512_and_si512            VPANDQ (ZMM, ZMM, ZMM)          1cy         0.33/cy     p05
 *      _mm512_popcnt_epi64         VPOPCNTQ (ZMM, ZMM)             3cy         1/cy        p5
 *      _mm512_maskz_loadu_epi8     VMOVDQU8 (ZMM, mem, k1)         7cy         0.5/cy      p23
 *
 *  Hamming distance kernels use XOR + hardware VPOPCNTQ for bit difference counting.
 *  Jaccard distance kernels use AND/OR + hardware VPOPCNTQ for intersection/union.
 *  Ice Lake processes 512 bits (64 bytes) per iteration using full ZMM registers with
 *  Ice Lake-specific state objects that leverage AVX-512 VPOPCNTQ for efficient popcount.
 *  Masked loads handle partial vector fills for non-multiple-of-512 bit depths.
 */

#ifndef NK_SETS_ICELAKE_H
#define NK_SETS_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"
#include "numkong/set/icelake.h"  // `nk_hamming_u1x512_state_icelake_t`
#include "numkong/cast/skylake.h" // `nk_partial_load_b1x512_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vpopcntdq", "f16c", \
                   "fma", "bmi", "bmi2")
#endif

// Four macro invocations for u1 - Ice Lake processes 512 bits at a time!
nk_define_cross_pack_size_(hammings, u1, icelake, u1x8, u32,
                           /*depth_simd_dimensions=*/512,
                           /*dimensions_per_value=*/8)

nk_define_cross_pack_(hammings, u1, icelake, u1x8, u32, nk_assign_from_to_,
                      /*depth_simd_dimensions=*/512,
                      /*dimensions_per_value=*/8)

nk_define_cross_symmetric_(hammings, u1, icelake, u1x8, u32, nk_b512_vec_t, nk_hamming_u1x512_state_icelake_t,
                           nk_b128_vec_t, nk_hamming_u1x512_init_icelake, nk_load_b512_skylake_,
                           nk_partial_load_b1x512_skylake_, nk_hamming_u1x512_update_icelake,
                           nk_hamming_u1x512_finalize_icelake, nk_partial_store_b32x4_skylake_,
                           /*depth_simd_dimensions=*/512,
                           /*dimensions_per_value=*/8)

nk_define_cross_packed_(hammings, u1, icelake, u1x8, u32, u32, nk_b512_vec_t, nk_hamming_u1x512_state_icelake_t,
                        nk_b128_vec_t, nk_hamming_u1x512_init_icelake, nk_load_b512_skylake_,
                        nk_partial_load_b1x512_skylake_, nk_load_b512_skylake_, nk_partial_load_b1x512_skylake_,
                        nk_hamming_u1x512_update_icelake, nk_hamming_u1x512_finalize_icelake,
                        nk_partial_store_b32x4_skylake_,
                        /*depth_simd_dimensions=*/512,
                        /*dimensions_per_value=*/8)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_SETS_ICELAKE_H
