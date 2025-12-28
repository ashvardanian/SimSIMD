/**
 *  @brief SIMD-accelerated horizontal reduction operations for Arm SVE-capable CPUs.
 *  @file include/numkong/reduce/sve.h
 *  @sa include/numkong/reduce.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 */
#ifndef NK_REDUCE_SVE_F16_H
#define NK_REDUCE_SVE_F16_H

#if NK_TARGET_ARM_
#if NK_TARGET_SVE_F16
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+fp16")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+fp16"))), apply_to = function)

#include "numkong/types.h"
#include "numkong/reduce/serial.h" // `nk_popcount_b8`
#include "numkong/reduce/neon.h"   // `nk_reduce_add_u8x16_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SVE_F16
#endif // NK_TARGET_ARM_

#endif // NK_REDUCE_SVE_F16_H