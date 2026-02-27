/**
 *  @brief Batched Spatial Distances for Sierra Forest (AVX-VNNI-INT16).
 *  @file include/numkong/spatials/sierra.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SIERRA_H
#define NK_SPATIALS_SIERRA_H

#if NK_TARGET_X86_
#if NK_TARGET_SIERRA

#include "numkong/spatial/serial.h"
#include "numkong/dots/sierra.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni,avxvnniint8"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni", "avxvnniint8")
#endif

nk_define_cross_normalized_packed_(angular, i8, sierra, i8, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                   nk_dots_packed_i8_sierra, nk_angular_through_i32_from_dot_serial_,
                                   nk_dots_reduce_sumsq_i8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                   nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)
nk_define_cross_normalized_symmetric_(angular, i8, sierra, i8, i32, /*norm_value_type=*/u32, f32, nk_b128_vec_t,
                                      nk_dots_symmetric_i8_sierra, nk_angular_through_i32_from_dot_serial_,
                                      nk_dots_reduce_sumsq_i8_, nk_load_b128_serial_, nk_partial_load_b32x4_serial_,
                                      nk_store_b128_serial_, nk_partial_store_b32x4_serial_, 1)

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SIERRA
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_SIERRA_H
