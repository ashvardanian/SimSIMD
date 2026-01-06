/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm SVE-capable CPUs.
 *  @file include/numkong/reduce/sve.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_SVE_H
#define NK_REDUCE_SVE_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_popcount_u1`
#include "numkong/reduce/neon.h"   // `nk_reduce_add_u8x16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_SVE_H
