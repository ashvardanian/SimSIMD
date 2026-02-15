/**
 *  @brief SIMD-accelerated Batched Set Distances for Haswell.
 *  @file include/numkong/sets/haswell.h
 *  @author Ash Vardanian
 *  @date January 25, 2026
 *
 *  @sa include/numkong/sets.h
 *
 *  @section sets_haswell_instructions Key AVX2 Set Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm256_xor_si256            VPXOR (YMM, YMM, YMM)           1cy         0.33/cy     p015
 *      _mm256_and_si256            VPAND (YMM, YMM, YMM)           1cy         0.33/cy     p015
 *      _mm_popcnt_u64              POPCNT (r64, r64)               3cy         1/cy        p1
 *
 *  Hamming distance kernels use XOR + POPCNT for bit difference counting.
 *  Jaccard distance kernels use AND/OR + POPCNT for intersection/union computation.
 *  Haswell processes 128 bits (16 bytes) per iteration using AVX2 128-bit loads with
 *  serial state objects for accumulation. Uses software popcount via lookup tables
 *  or POPCNT instruction on extracted 64-bit chunks.
 */

#ifndef NK_SETS_HASWELL_H
#define NK_SETS_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/set/serial.h"   // `nk_hamming_u1x128_state_serial_t`
#include "numkong/cast/haswell.h" // `nk_partial_load_b8x32_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "popcnt")
#endif

// Four macro invocations for u1 - matching serial pattern
nk_define_cross_pack_size_(hammings, u1, haswell, u1x8, u32,
                           /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/128)

nk_define_cross_pack_(hammings, u1, haswell, u1x8, u32, nk_assign_from_to_,
                      /*depth_simd_dimensions=*/128,
                      /*dimensions_per_value=*/128)

nk_define_cross_symmetric_(hammings, u1, haswell, u1x8, u32, nk_b128_vec_t, nk_hamming_u1x128_state_serial_t,
                           nk_b128_vec_t, nk_hamming_u1x128_init_serial, nk_load_b128_haswell_,
                           nk_partial_load_b32x4_serial_, nk_hamming_u1x128_update_serial,
                           nk_hamming_u1x128_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/128)

nk_define_cross_packed_(hammings, u1, haswell, u1x8, u32, u32, nk_b128_vec_t, nk_hamming_u1x128_state_serial_t,
                        nk_b128_vec_t, nk_hamming_u1x128_init_serial, nk_load_b128_haswell_,
                        nk_partial_load_b32x4_serial_, nk_load_b128_haswell_, nk_partial_load_b32x4_serial_,
                        nk_hamming_u1x128_update_serial, nk_hamming_u1x128_finalize_serial,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128,
                        /*dimensions_per_value=*/128)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_SETS_HASWELL_H
