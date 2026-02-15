/**
 *  @brief SWAR-accelerated Batched Set Distances for SIMD-free CPUs.
 *  @file include/numkong/sets/serial.h
 *  @author Ash Vardanian
 *  @date January 24, 2026
 *
 *  @sa include/numkong/sets.h for API overview and use cases
 *
 *  This file provides macro families for generating batched Hamming and Jaccard distance kernels:
 *
 *  - nk_define_hammings_packed_: Hamming distances between rows of A and packed B
 *  - nk_define_hammings_symmetric_: Hamming distances within rows of A (Gram matrix style)
 *  - nk_define_jaccards_packed_: Jaccard distances between rows of A and packed B
 *  - nk_define_jaccards_symmetric_: Jaccard distances within rows of A
 *
 *  All macros are parameterizable with custom load, update, and finalize functions
 *  to support different SIMD backends (serial, NEON, AVX-512, SVE).
 *
 *  @section sets_packing B Matrix Packing Format
 *
 *  Binary vectors are packed row-by-row with depth rounded up to SIMD width.
 *  For Jaccard, precomputed population counts (norms) are appended after packed data.
 *
 *  Memory layout: header + packed_data[n × depth_padded_bytes] + norms[n] (Jaccard only)
 *
 *  @section sets_math Mathematical Foundation
 *
 *  Hamming distance: count of differing bits = popcount(a XOR b)
 *
 *  Jaccard distance: 1 - |a ∩ b| / |a ∪ b|
 *                  = 1 - popcount(a AND b) / popcount(a OR b)
 *                  = 1 - popcount(a AND b) / (popcount(a) + popcount(b) - popcount(a AND b))
 */

#ifndef NK_SETS_SERIAL_H
#define NK_SETS_SERIAL_H

#include "numkong/dots/serial.h" // `nk_define_cross_symmetric_`
#include "numkong/set/serial.h"  // `nk_hamming_u1x128_state_serial_t`
#include "numkong/cast/serial.h" // `nk_load_b128_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_pack_size_(hammings, u1, serial, u1x8, u32, /*depth_simd_dimensions=*/128,
                           /*dimensions_per_value=*/128)
nk_define_cross_pack_(hammings, u1, serial, u1x8, u32, nk_assign_from_to_, /*depth_simd_dimensions=*/128,
                      /*dimensions_per_value=*/128)
nk_define_cross_symmetric_(hammings, u1, serial, u1x8, u32, nk_b128_vec_t, nk_hamming_u1x128_state_serial_t,
                           nk_b128_vec_t, nk_hamming_u1x128_init_serial, nk_load_b128_serial_,
                           nk_partial_load_b32x4_serial_, nk_hamming_u1x128_update_serial,
                           nk_hamming_u1x128_finalize_serial, nk_partial_store_b32x4_serial_,
                           /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/128)
nk_define_cross_packed_(hammings, u1, serial, u1x8, u32, u32, nk_b128_vec_t, nk_hamming_u1x128_state_serial_t,
                        nk_b128_vec_t, nk_hamming_u1x128_init_serial, nk_load_b128_serial_,
                        nk_partial_load_b32x4_serial_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                        nk_hamming_u1x128_update_serial, nk_hamming_u1x128_finalize_serial,
                        nk_partial_store_b32x4_serial_,
                        /*depth_simd_dimensions=*/128, /*dimensions_per_value=*/128)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SETS_SERIAL_H
