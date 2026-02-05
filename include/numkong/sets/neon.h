/**
 *  @brief SIMD-accelerated Batched Set Distances for NEON.
 *  @file include/numkong/sets/neon.h
 *  @author Ash Vardanian
 *  @date January 25, 2026
 *
 *  @sa include/numkong/sets.h
 *
 *  @section sets_neon_instructions Key NEON Set Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      veorq_u8                    EOR (Vd.16B, Vn.16B, Vm.16B)    2cy         2/cy        -
 *      vandq_u8                    AND (Vd.16B, Vn.16B, Vm.16B)    2cy         2/cy        -
 *      vcntq_u8                    CNT (Vd.16B, Vn.16B)            2cy         2/cy        -
 *      vld1q_u8                    LD1 {Vt.16B}, [Xn]              4cy         2/cy        -
 *
 *  Hamming distance kernels use EOR + CNT for bit difference counting.
 *  Jaccard distance kernels use AND/ORR + CNT for intersection/union computation.
 *  NEON processes 128 bits (16 bytes) per iteration using 128-bit vector registers with
 *  serial state objects for accumulation. Uses ARMv8 CNT instruction for 8-bit popcount
 *  followed by horizontal sum via ADDV.
 */

#ifndef NK_SETS_NEON_H
#define NK_SETS_NEON_H

#if defined(__cplusplus)
extern "C" {
#endif

#if NK_TARGET_ARM_
#if NK_TARGET_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

#include "numkong/types.h"
#include "numkong/set/serial.h" // For nk_hamming_u1x128_state_serial_t
#include "numkong/cast/neon.h"  // For load functions

// Four macro invocations for u1 - matching serial pattern
nk_define_cross_pack_size_(hammings, u1, neon, u1x8, u32,
                           /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/128)

nk_define_cross_pack_(hammings, u1, neon, u1x8, u32, nk_assign_from_to_,
                      /*depth_simd_dimensions=*/128,
                      /*dimensions_per_value=*/128)

nk_define_cross_symmetric_(hammings, u1, neon, u1x8, u32, nk_b128_vec_t, nk_hamming_u1x128_state_serial_t,
                           nk_b128_vec_t, nk_hamming_u1x128_init_serial, nk_load_b128_neon_,
                           nk_partial_load_b32x4_serial_, nk_hamming_u1x128_update_serial,
                           nk_hamming_u1x128_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/128)

nk_define_cross_packed_(hammings, u1, neon, u1x8, u32, u32, nk_b128_vec_t, nk_hamming_u1x128_state_serial_t,
                        nk_b128_vec_t, nk_hamming_u1x128_init_serial, nk_load_b128_neon_, nk_partial_load_b32x4_serial_,
                        nk_load_b128_neon_, nk_partial_load_b32x4_serial_, nk_hamming_u1x128_update_serial,
                        nk_hamming_u1x128_finalize_serial, nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128,
                        /*dimensions_per_value=*/128)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SETS_NEON_H
