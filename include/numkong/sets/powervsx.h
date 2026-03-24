/**
 *  @brief Batched Set Operations for Power VSX.
 *  @file include/numkong/sets/powervsx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/sets.h
 */
#ifndef NK_SETS_POWERVSX_H
#define NK_SETS_POWERVSX_H

#if NK_TARGET_POWER_
#if NK_TARGET_POWERVSX

#include "numkong/set/powervsx.h"
#include "numkong/dots/powervsx.h"

#if defined(__cplusplus)
extern "C" {
#endif

nk_define_cross_normalized_packed_(hamming, u1, powervsx, u1x8, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                   nk_dots_packed_u1_powervsx, nk_hamming_u32x4_from_dot_powervsx_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_packed_(jaccard, u1, powervsx, u1x8, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_u1_powervsx, nk_jaccard_f32x4_from_dot_powervsx_,
                                   nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(hamming, u1, powervsx, u1x8, u32, /*norm_value_type=*/u32, u32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_powervsx, nk_hamming_u32x4_from_dot_powervsx_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                                      /*dimensions_per_value=*/8)

nk_define_cross_normalized_symmetric_(jaccard, u1, powervsx, u1x8, u32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_u1_powervsx, nk_jaccard_f32x4_from_dot_powervsx_,
                                      nk_dots_reduce_sum_u1_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_,
                                      /*dimensions_per_value=*/8)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER_
#endif // NK_SETS_POWERVSX_H
